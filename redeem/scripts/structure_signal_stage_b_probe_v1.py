#!/usr/bin/env python3
"""Tiny deterministic Stage B whole-peptide/conformer engine probe.

This is a bounded TRAIN-only feasibility probe for ReDeeM's
``structure_signal_feasibility_v1`` lane. It consumes the already-frozen Stage B
manifest and uses no measured CCS or mobility label. The probe deliberately does
not train a model and does not process the full 8k manifest.

Scientific boundary
-------------------
* Whole-peptide connection tables are constructed with RDKit ``MolFromSequence``.
* Only the common PTM families pre-qualified by Stage A are implemented:
  UniMod:4 carbamidomethyl C, UniMod:35 oxidation M, UniMod:7 deamidation N/Q,
  UniMod:21 phospho S/T/Y, and UniMod:1 acetyl K/N-terminus.
* Every built neutral molecule must reproduce Stage A's theoretical neutral mass
  within a tight tolerance before any 3D work is allowed.
* Charge conditioning is an explicitly heuristic, deterministic protonation-site
  policy. It is a feasibility prior, not a claim that the selected microstate is
  the experimentally occupied gas-phase protomer.
* Two charge-site microstates are attempted where possible, followed by fixed-seed
  ETKDGv3 conformer generation and MMFF94s (UFF fallback) minimization.
* Structural descriptors are independent of measured CCS.

The output is intended to decide whether a larger 5k-10k Stage B conformer pilot
is scientifically and computationally worthwhile. It is not a v0.70 input file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdFreeSASA, rdMolDescriptors
except ImportError as exc:  # pragma: no cover - exercised by container gate
    raise SystemExit(
        "RDKit is required for structure_signal_stage_b_probe_v1; "
        "use the dedicated pinned probe container"
    ) from exc

AUDIT_VERSION = "structure_signal_feasibility_v1_stage_b_probe_v1"
SELECTION_SALT = "redeem_structure_signal_stage_b_probe_v1_sha256"
DEFAULT_PROBE_SIZE = 42
DEFAULT_MATCHED_CHARGE_PAIRS = 4
DEFAULT_MIN_PER_PTM_CLASS = 3
DEFAULT_NUM_CONFORMERS = 3
DEFAULT_MICROSTATES = 2
DEFAULT_MAX_MINIMIZE_ITERS = 250
DEFAULT_REPRODUCIBILITY_CHECKS = 6
DEFAULT_MASS_TOLERANCE_DA = 0.002

SUPPORTED_PTM_CLASSES = {
    "unmodified",
    "carbamidomethyl",
    "oxidation",
    "deamidation",
    "phospho",
    "acetyl",
    "common_ptm_combination",
}

FORBIDDEN_LABEL_COLUMNS = {
    "ccs",
    "target_ccs",
    "measured_ccs",
    "predicted_ccs",
    "mobility",
    "target_mobility",
    "measured_mobility",
    "predicted_mobility",
    "ion_mobility",
    "target_ion_mobility",
}

REQUIRED_MANIFEST_COLUMNS = {
    "stage_b_order",
    "stratum_key",
    "ptm_class",
    "selection_hash_sha256",
    "identity_key",
    "peptidoform",
    "sequence",
    "charge",
    "sequence_length",
    "length_bin",
    "modified",
    "modification_count",
    "theoretical_neutral_mass_da",
    "chemistry_status",
    "existing_attachment_template_candidate",
}

HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "PRO"}


@dataclass(frozen=True)
class Modification:
    site: str
    unimod_id: int


@dataclass(frozen=True)
class ChargeSite:
    atom_idx: int
    label: str
    kind: str
    priority: int
    residue_index: int


@dataclass
class BuildResult:
    mol: Chem.Mol
    blocked_basic_atoms: set[int]
    phosphate_acidic_atoms: set[int]
    neutral_exact_mass: float
    mass_error_da: float
    canonical_smiles: str


@dataclass
class SelectedProbeRow:
    row: dict[str, str]
    reasons: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-manifest-fingerprint")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--probe-size", type=int, default=DEFAULT_PROBE_SIZE)
    parser.add_argument(
        "--matched-charge-pairs", type=int, default=DEFAULT_MATCHED_CHARGE_PAIRS
    )
    parser.add_argument(
        "--minimum-per-ptm-class", type=int, default=DEFAULT_MIN_PER_PTM_CLASS
    )
    parser.add_argument("--num-conformers", type=int, default=DEFAULT_NUM_CONFORMERS)
    parser.add_argument("--microstates", type=int, default=DEFAULT_MICROSTATES)
    parser.add_argument(
        "--max-minimize-iters", type=int, default=DEFAULT_MAX_MINIMIZE_ITERS
    )
    parser.add_argument(
        "--reproducibility-checks",
        type=int,
        default=DEFAULT_REPRODUCIBILITY_CHECKS,
    )
    parser.add_argument(
        "--mass-tolerance-da", type=float, default=DEFAULT_MASS_TOLERANCE_DA
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def stable_digest(text: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}\t{text}".encode("utf-8")).hexdigest()


def seed_for(peptidoform: str, microstate_index: int) -> int:
    digest = hashlib.sha256(
        f"{AUDIT_VERSION}\t{peptidoform}\tmicrostate={microstate_index}".encode("utf-8")
    ).digest()
    # RDKit expects a signed 32-bit-ish seed; keep it positive/nonzero.
    return 1 + int.from_bytes(digest[:4], "big") % 2_000_000_000


def parse_int(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not an integer: {value!r}") from exc


def parse_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} is not finite: {value!r}")
    return parsed


def normalize_yes(value: str) -> bool:
    return value.strip().upper() == "YES"


def validate_manifest_header(fieldnames: Iterable[str] | None) -> list[str]:
    if fieldnames is None:
        raise ValueError("Stage B manifest has no header")
    fields = list(fieldnames)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(fields))
    if missing:
        raise ValueError("Stage B manifest missing columns: " + ", ".join(missing))
    lower = {name.strip().lower() for name in fields}
    forbidden = sorted(FORBIDDEN_LABEL_COLUMNS & lower)
    if forbidden:
        raise ValueError(
            "refusing Stage B manifest containing CCS/mobility label columns: "
            + ", ".join(forbidden)
        )
    return fields


def manifest_fingerprint(rows: Sequence[Mapping[str, str]]) -> str:
    hasher = hashlib.sha256()
    expected_order = 1
    for row in rows:
        observed = parse_int(row["stage_b_order"], "stage_b_order")
        if observed != expected_order:
            raise ValueError(
                f"Stage B manifest order is not contiguous: expected={expected_order} observed={observed}"
            )
        expected_order += 1
        hasher.update(row["stratum_key"].encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(row["selection_hash_sha256"].encode("ascii"))
        hasher.update(b"\t")
        hasher.update(row["identity_key"].encode("utf-8"))
        hasher.update(b"\n")
    return "sha256:" + hasher.hexdigest()


def load_manifest(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = validate_manifest_header(reader.fieldnames)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("Stage B manifest is empty")
    fingerprint = manifest_fingerprint(rows)
    return fields, rows, fingerprint


def charge_bucket(charge: int) -> str:
    if charge >= 5:
        return "5+"
    return str(charge)


def selection_sort_key(row: Mapping[str, str]) -> tuple[str, str]:
    return (stable_digest(row["identity_key"]), row["identity_key"])


def choose_diverse_rows(
    candidates: Sequence[dict[str, str]], count: int, used: set[str]
) -> list[dict[str, str]]:
    if count <= 0:
        return []
    available = [row for row in candidates if row["identity_key"] not in used]
    available.sort(key=selection_sort_key)
    chosen: list[dict[str, str]] = []
    seen_cells: set[tuple[str, str]] = set()
    for row in available:
        cell = (row["charge"], row["length_bin"])
        if cell in seen_cells:
            continue
        chosen.append(row)
        seen_cells.add(cell)
        if len(chosen) >= count:
            return chosen
    for row in available:
        if row in chosen:
            continue
        chosen.append(row)
        if len(chosen) >= count:
            break
    return chosen


def select_probe_rows(
    rows: Sequence[dict[str, str]],
    probe_size: int,
    matched_charge_pairs: int,
    minimum_per_ptm_class: int,
) -> list[SelectedProbeRow]:
    if probe_size <= 0:
        raise ValueError("probe_size must be positive")
    if matched_charge_pairs < 0 or minimum_per_ptm_class < 0:
        raise ValueError("selection counts cannot be negative")

    eligible = [
        row
        for row in rows
        if row["ptm_class"] in SUPPORTED_PTM_CLASSES
        and normalize_yes(row["existing_attachment_template_candidate"])
    ]
    if len(eligible) < probe_size:
        raise ValueError(
            f"only {len(eligible)} probe-buildable rows for requested size {probe_size}"
        )

    selected: dict[str, SelectedProbeRow] = {}

    def add(row: dict[str, str], reason: str) -> None:
        key = row["identity_key"]
        entry = selected.get(key)
        if entry is None:
            selected[key] = SelectedProbeRow(row=row, reasons={reason})
        else:
            entry.reasons.add(reason)

    # Reserve a handful of naturally observed same-peptidoform / different-charge
    # pairs so the probe can test charge-conditioned descriptor variation without
    # inventing a new sequence/PTM identity.
    by_peptidoform: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        by_peptidoform[row["peptidoform"]].append(row)
    pair_groups: list[tuple[str, list[dict[str, str]]]] = []
    for peptidoform, group in by_peptidoform.items():
        by_charge: dict[int, dict[str, str]] = {}
        for row in sorted(group, key=selection_sort_key):
            by_charge.setdefault(parse_int(row["charge"], "charge"), row)
        if len(by_charge) >= 2:
            pair_groups.append((stable_digest(peptidoform), list(by_charge.values())))
    pair_groups.sort(key=lambda item: item[0])
    pair_count = 0
    for _, group in pair_groups:
        if pair_count >= matched_charge_pairs:
            break
        group = sorted(group, key=lambda row: (parse_int(row["charge"], "charge"), selection_sort_key(row)))
        # Prefer two charges from the central 2-4 range where possible.
        central = [row for row in group if 2 <= parse_int(row["charge"], "charge") <= 4]
        chosen_pair = central[:2] if len(central) >= 2 else group[:2]
        if len(chosen_pair) < 2:
            continue
        for row in chosen_pair:
            add(row, "matched_observed_charge_pair")
        pair_count += 1

    # Ensure every major PTM class is represented by multiple identities and by
    # more than one charge/length cell when the frozen manifest provides it.
    for ptm_class in sorted(SUPPORTED_PTM_CLASSES):
        current = sum(entry.row["ptm_class"] == ptm_class for entry in selected.values())
        need = max(0, minimum_per_ptm_class - current)
        class_rows = [row for row in eligible if row["ptm_class"] == ptm_class]
        for row in choose_diverse_rows(class_rows, need, set(selected)):
            add(row, "ptm_class_minimum")

    # Ensure all observed charge buckets and all six Stage A length bins have at
    # least one mechanical probe identity where possible.
    all_charge_buckets = sorted(
        {charge_bucket(parse_int(row["charge"], "charge")) for row in eligible},
        key=lambda value: (value == "5+", int(value.rstrip("+"))),
    )
    for bucket in all_charge_buckets:
        if any(
            charge_bucket(parse_int(entry.row["charge"], "charge")) == bucket
            for entry in selected.values()
        ):
            continue
        candidates = [
            row
            for row in eligible
            if charge_bucket(parse_int(row["charge"], "charge")) == bucket
            and row["identity_key"] not in selected
        ]
        if candidates:
            add(min(candidates, key=selection_sort_key), "charge_bucket_coverage")

    for length_bin in sorted({row["length_bin"] for row in eligible}):
        if any(entry.row["length_bin"] == length_bin for entry in selected.values()):
            continue
        candidates = [
            row
            for row in eligible
            if row["length_bin"] == length_bin and row["identity_key"] not in selected
        ]
        if candidates:
            add(min(candidates, key=selection_sort_key), "length_bin_coverage")

    # Include the most highly charged observed identity and one maximum-length
    # identity as explicit stress cases. These are descriptive stress probes; a
    # failure confined to such a rare extreme does not silently redefine the
    # major-population chemistry coverage established in Stage A.
    max_charge = max(parse_int(row["charge"], "charge") for row in eligible)
    max_charge_rows = [
        row for row in eligible if parse_int(row["charge"], "charge") == max_charge
    ]
    if max_charge_rows:
        add(min(max_charge_rows, key=selection_sort_key), "maximum_charge_stress")

    max_length = max(parse_int(row["sequence_length"], "sequence_length") for row in eligible)
    max_length_rows = [
        row
        for row in eligible
        if parse_int(row["sequence_length"], "sequence_length") == max_length
    ]
    if max_length_rows:
        add(min(max_length_rows, key=selection_sort_key), "maximum_length_stress")

    if len(selected) > probe_size:
        raise ValueError(
            f"mandatory probe diversity ({len(selected)}) exceeds probe_size={probe_size}; increase probe size"
        )

    # Fill remaining seats with a deterministic global hash rank.
    for row in sorted(eligible, key=selection_sort_key):
        if len(selected) >= probe_size:
            break
        if row["identity_key"] not in selected:
            add(row, "deterministic_fill")

    final = list(selected.values())
    if len(final) != probe_size:
        raise RuntimeError(f"probe selection mismatch: observed={len(final)} expected={probe_size}")
    final.sort(key=lambda entry: (int(entry.row["stage_b_order"]), entry.row["identity_key"]))
    return final


def parse_peptidoform(peptidoform: str, expected_sequence: str) -> tuple[str, list[Modification]]:
    parts = peptidoform.split("|")
    sequence = parts[0]
    if sequence != expected_sequence:
        raise ValueError(
            f"peptidoform sequence mismatch: peptidoform={sequence!r} sequence={expected_sequence!r}"
        )
    mods: list[Modification] = []
    for token in parts[1:]:
        if ":" not in token:
            raise ValueError(f"malformed modification token: {token!r}")
        site, identity = token.split(":", 1)
        if not identity.startswith("UniMod:"):
            raise ValueError(f"probe refuses non-UniMod modification: {token!r}")
        unimod_id = parse_int(identity.split(":", 1)[1], "unimod_id")
        mods.append(Modification(site=site, unimod_id=unimod_id))
    return sequence, mods


def pdb_atom_idx(mol: Chem.Mol, residue_index: int, atom_name: str) -> int:
    target_number = residue_index + 1
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        if info.GetResidueNumber() == target_number and info.GetName().strip() == atom_name:
            return atom.GetIdx()
    raise ValueError(
        f"cannot find residue atom R{residue_index}:{atom_name} in RDKit peptide"
    )


def residue_name(mol: Chem.Mol, residue_index: int) -> str:
    target_number = residue_index + 1
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is not None and info.GetResidueNumber() == target_number:
            return info.GetResidueName().strip()
    raise ValueError(f"cannot resolve residue R{residue_index}")


def residue_letter(sequence: str, site: str) -> tuple[int, str]:
    if not site.startswith("R"):
        raise ValueError(f"expected residue site, observed {site!r}")
    index = parse_int(site[1:], "residue_index")
    if index < 0 or index >= len(sequence):
        raise ValueError(f"residue modification index out of range: {site!r}")
    return index, sequence[index]


def add_atom(rw: Chem.RWMol, atomic_number: int) -> int:
    return rw.AddAtom(Chem.Atom(atomic_number))


def finalize_edit(rw: Chem.RWMol) -> Chem.Mol:
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def apply_carbamidomethyl(mol: Chem.Mol, residue_index: int) -> Chem.Mol:
    if residue_name(mol, residue_index) != "CYS":
        raise ValueError("UniMod:4 probe implementation requires cysteine")
    rw = Chem.RWMol(mol)
    sulfur = pdb_atom_idx(rw, residue_index, "SG")
    methylene = add_atom(rw, 6)
    carbonyl_c = add_atom(rw, 6)
    carbonyl_o = add_atom(rw, 8)
    amide_n = add_atom(rw, 7)
    rw.AddBond(sulfur, methylene, Chem.BondType.SINGLE)
    rw.AddBond(methylene, carbonyl_c, Chem.BondType.SINGLE)
    rw.AddBond(carbonyl_c, carbonyl_o, Chem.BondType.DOUBLE)
    rw.AddBond(carbonyl_c, amide_n, Chem.BondType.SINGLE)
    return finalize_edit(rw)


def apply_methionine_oxidation(mol: Chem.Mol, residue_index: int) -> Chem.Mol:
    if residue_name(mol, residue_index) != "MET":
        raise ValueError("UniMod:35 probe implementation requires methionine")
    rw = Chem.RWMol(mol)
    sulfur = pdb_atom_idx(rw, residue_index, "SD")
    oxygen = add_atom(rw, 8)
    rw.AddBond(sulfur, oxygen, Chem.BondType.DOUBLE)
    return finalize_edit(rw)


def apply_deamidation(mol: Chem.Mol, residue_index: int, residue: str) -> Chem.Mol:
    atom_name = {"N": "ND2", "Q": "NE2"}.get(residue)
    if atom_name is None:
        raise ValueError("UniMod:7 probe implementation requires N or Q")
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(pdb_atom_idx(rw, residue_index, atom_name))
    atom.SetAtomicNum(8)
    atom.SetFormalCharge(0)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(False)
    return finalize_edit(rw)


def apply_phosphorylation(
    mol: Chem.Mol, residue_index: int, residue: str
) -> tuple[Chem.Mol, set[int]]:
    atom_name = {"S": "OG", "T": "OG1", "Y": "OH"}.get(residue)
    if atom_name is None:
        raise ValueError("UniMod:21 probe implementation requires S, T or Y")
    rw = Chem.RWMol(mol)
    attachment_o = pdb_atom_idx(rw, residue_index, atom_name)
    phosphorus = add_atom(rw, 15)
    double_o = add_atom(rw, 8)
    acid_o1 = add_atom(rw, 8)
    acid_o2 = add_atom(rw, 8)
    rw.AddBond(attachment_o, phosphorus, Chem.BondType.SINGLE)
    rw.AddBond(phosphorus, double_o, Chem.BondType.DOUBLE)
    rw.AddBond(phosphorus, acid_o1, Chem.BondType.SINGLE)
    rw.AddBond(phosphorus, acid_o2, Chem.BondType.SINGLE)
    return finalize_edit(rw), {acid_o1, acid_o2}


def apply_acetylation(
    mol: Chem.Mol, sequence: str, site: str
) -> tuple[Chem.Mol, int]:
    if site == "N-term":
        target_n = pdb_atom_idx(mol, 0, "N")
    else:
        residue_index, residue = residue_letter(sequence, site)
        if residue != "K":
            raise ValueError("UniMod:1 residue probe implementation requires lysine")
        target_n = pdb_atom_idx(mol, residue_index, "NZ")
    rw = Chem.RWMol(mol)
    carbonyl_c = add_atom(rw, 6)
    carbonyl_o = add_atom(rw, 8)
    methyl_c = add_atom(rw, 6)
    rw.AddBond(target_n, carbonyl_c, Chem.BondType.SINGLE)
    rw.AddBond(carbonyl_c, carbonyl_o, Chem.BondType.DOUBLE)
    rw.AddBond(carbonyl_c, methyl_c, Chem.BondType.SINGLE)
    return finalize_edit(rw), target_n


def build_neutral_molecule(
    row: Mapping[str, str], mass_tolerance_da: float
) -> BuildResult:
    sequence, modifications = parse_peptidoform(row["peptidoform"], row["sequence"])
    mol = Chem.MolFromSequence(sequence)
    if mol is None:
        raise ValueError(f"RDKit MolFromSequence failed for {sequence!r}")
    blocked_basic_atoms: set[int] = set()
    phosphate_acidic_atoms: set[int] = set()

    for modification in modifications:
        if modification.unimod_id == 4:
            residue_index, residue = residue_letter(sequence, modification.site)
            if residue != "C":
                raise ValueError("UniMod:4 encountered outside cysteine")
            mol = apply_carbamidomethyl(mol, residue_index)
        elif modification.unimod_id == 35:
            residue_index, residue = residue_letter(sequence, modification.site)
            if residue != "M":
                raise ValueError("UniMod:35 encountered outside methionine")
            mol = apply_methionine_oxidation(mol, residue_index)
        elif modification.unimod_id == 7:
            residue_index, residue = residue_letter(sequence, modification.site)
            mol = apply_deamidation(mol, residue_index, residue)
        elif modification.unimod_id == 21:
            residue_index, residue = residue_letter(sequence, modification.site)
            mol, acidic = apply_phosphorylation(mol, residue_index, residue)
            phosphate_acidic_atoms.update(acidic)
        elif modification.unimod_id == 1:
            mol, target_n = apply_acetylation(mol, sequence, modification.site)
            blocked_basic_atoms.add(target_n)
        else:
            raise ValueError(
                f"tiny probe does not implement UniMod:{modification.unimod_id}"
            )

    observed_mass = rdMolDescriptors.CalcExactMolWt(mol)
    expected_mass = parse_float(row["theoretical_neutral_mass_da"], "theoretical_neutral_mass_da")
    mass_error = observed_mass - expected_mass
    if abs(mass_error) > mass_tolerance_da:
        raise ValueError(
            f"neutral molecule mass mismatch for {row['identity_key']}: "
            f"rdkit={observed_mass:.8f} stage_a={expected_mass:.8f} error={mass_error:+.8f}"
        )
    return BuildResult(
        mol=mol,
        blocked_basic_atoms=blocked_basic_atoms,
        phosphate_acidic_atoms=phosphate_acidic_atoms,
        neutral_exact_mass=observed_mass,
        mass_error_da=mass_error,
        canonical_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
    )


def candidate_charge_sites(mol: Chem.Mol, blocked_basic_atoms: set[int]) -> list[ChargeSite]:
    sites: list[ChargeSite] = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        residue_index = info.GetResidueNumber() - 1
        residue_name_3 = info.GetResidueName().strip()
        atom_name = info.GetName().strip()
        if atom.GetIdx() in blocked_basic_atoms:
            continue
        if residue_name_3 == "ARG" and atom_name == "NH1":
            sites.append(
                ChargeSite(atom.GetIdx(), f"R{residue_index}:Arg-NH1", "basic_n", 400, residue_index)
            )
        elif residue_name_3 == "LYS" and atom_name == "NZ":
            sites.append(
                ChargeSite(atom.GetIdx(), f"R{residue_index}:Lys-NZ", "basic_n", 350, residue_index)
            )
        elif residue_name_3 == "HIS" and atom_name == "ND1":
            sites.append(
                ChargeSite(atom.GetIdx(), f"R{residue_index}:His-ND1", "basic_n", 300, residue_index)
            )
        elif residue_index == 0 and atom_name == "N":
            sites.append(ChargeSite(atom.GetIdx(), "N-term", "basic_n", 250, residue_index))

    # Backbone carbonyl protonation is used only as a deterministic fallback when
    # the requested positive precursor charge exceeds the conventional basic-site
    # count. This is intentionally reported as a heuristic fallback.
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetName().strip() != "O":
            continue
        residue_index = info.GetResidueNumber() - 1
        sites.append(
            ChargeSite(
                atom.GetIdx(),
                f"R{residue_index}:backbone-carbonyl-O",
                "carbonyl_o_fallback",
                100,
                residue_index,
            )
        )
    sites.sort(key=lambda site: (-site.priority, site.residue_index, site.label))
    # Each heavy atom should appear at most once.
    unique: list[ChargeSite] = []
    seen: set[int] = set()
    for site in sites:
        if site.atom_idx not in seen:
            unique.append(site)
            seen.add(site.atom_idx)
    return unique


def microstate_sites(
    candidates: Sequence[ChargeSite], charge: int, microstate_index: int
) -> list[ChargeSite]:
    if charge <= 0:
        raise ValueError("precursor charge must be positive")
    if len(candidates) < charge:
        raise ValueError(
            f"only {len(candidates)} deterministic protonation candidates for z={charge}"
        )
    primary = list(candidates[:charge])
    if microstate_index == 0 or len(candidates) == charge:
        return primary
    # Alternate states progressively replace the lowest-ranked selected site with
    # the next available site. This is bounded and deterministic; it is not an
    # exhaustive protomer enumeration.
    replacement_index = charge - 1 + microstate_index
    if replacement_index >= len(candidates):
        replacement_index = charge
    if replacement_index < len(candidates):
        return list(candidates[: charge - 1]) + [candidates[replacement_index]]
    return primary


def apply_positive_charge(
    mol: Chem.Mol, sites: Sequence[ChargeSite], expected_charge: int
) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    for site in sites:
        atom = rw.GetAtomWithIdx(site.atom_idx)
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        if site.kind == "carbonyl_o_fallback":
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            atom.SetNoImplicit(True)
    charged = finalize_edit(rw)
    observed_charge = sum(atom.GetFormalCharge() for atom in charged.GetAtoms())
    if observed_charge != expected_charge:
        raise ValueError(
            f"formal charge mismatch: observed={observed_charge} expected={expected_charge}"
        )
    return charged


def embed_and_minimize(
    charged_mol: Chem.Mol,
    peptidoform: str,
    microstate_index: int,
    num_conformers: int,
    max_minimize_iters: int,
) -> tuple[Chem.Mol, list[int], str, list[tuple[int, float]], str, float]:
    if num_conformers <= 0:
        raise ValueError("num_conformers must be positive")
    started = time.perf_counter()
    mol_h = Chem.AddHs(charged_mol)
    seed = seed_for(peptidoform, microstate_index)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.pruneRmsThresh = -1.0
    embed_mode = "etkdgv3"
    conformer_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params))
    if not conformer_ids:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1
        params.pruneRmsThresh = -1.0
        params.useRandomCoords = True
        embed_mode = "etkdgv3_random_coords_fallback"
        conformer_ids = list(
            AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params)
        )
    if not conformer_ids:
        raise RuntimeError("ETKDGv3 produced zero conformers")

    ff_name = "none"
    optimization: list[tuple[int, float]] = [(1, float("nan")) for _ in conformer_ids]
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            ff_name = "MMFF94s"
            optimization = list(
                AllChem.MMFFOptimizeMoleculeConfs(
                    mol_h,
                    numThreads=1,
                    maxIters=max_minimize_iters,
                    mmffVariant="MMFF94s",
                )
            )
        elif AllChem.UFFHasAllMoleculeParams(mol_h):
            ff_name = "UFF"
            optimization = list(
                AllChem.UFFOptimizeMoleculeConfs(
                    mol_h, numThreads=1, maxIters=max_minimize_iters
                )
            )
    except Exception as exc:  # keep ETKDG coordinates for descriptor feasibility
        ff_name = f"optimization_failed:{type(exc).__name__}"
        optimization = [(1, float("nan")) for _ in conformer_ids]

    elapsed = time.perf_counter() - started
    return mol_h, conformer_ids, ff_name, optimization, embed_mode, elapsed


def distance(conf: Chem.Conformer, atom_a: int, atom_b: int) -> float:
    a = conf.GetAtomPosition(atom_a)
    b = conf.GetAtomPosition(atom_b)
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def mean_or_nan(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def extrema_or_nan(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return min(values), max(values)


def acidic_atom_indices(mol: Chem.Mol, phosphate_atoms: set[int]) -> list[int]:
    indices = set(phosphate_atoms)
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        res = info.GetResidueName().strip()
        name = info.GetName().strip()
        if res == "ASP" and name in {"OD1", "OD2"}:
            indices.add(atom.GetIdx())
        elif res == "GLU" and name in {"OE1", "OE2"}:
            indices.add(atom.GetIdx())
        elif name == "OXT":
            indices.add(atom.GetIdx())
    return sorted(indices)


def terminal_atom_indices(mol: Chem.Mol, sequence_length: int) -> tuple[int, int]:
    return pdb_atom_idx(mol, 0, "N"), pdb_atom_idx(mol, sequence_length - 1, "C")


def hydrophobic_atom_indices(mol: Chem.Mol) -> list[int]:
    indices = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        info = atom.GetPDBResidueInfo()
        if info is not None and info.GetResidueName().strip() in HYDROPHOBIC_RESIDUES:
            indices.append(atom.GetIdx())
    return indices


def coordinate_rg(conf: Chem.Conformer, atom_indices: Sequence[int]) -> float:
    if not atom_indices:
        return float("nan")
    coords = [conf.GetAtomPosition(index) for index in atom_indices]
    cx = statistics.fmean(point.x for point in coords)
    cy = statistics.fmean(point.y for point in coords)
    cz = statistics.fmean(point.z for point in coords)
    return math.sqrt(
        statistics.fmean(
            (point.x - cx) ** 2 + (point.y - cy) ** 2 + (point.z - cz) ** 2
            for point in coords
        )
    )


def conformer_descriptors(
    mol_h: Chem.Mol,
    conf_id: int,
    charge_site_indices: Sequence[int],
    acidic_indices: Sequence[int],
    nterm_idx: int,
    cterm_c_idx: int,
    hydrophobic_indices: Sequence[int],
) -> dict[str, float]:
    conf = mol_h.GetConformer(conf_id)
    charge_pair_distances = [
        distance(conf, charge_site_indices[i], charge_site_indices[j])
        for i in range(len(charge_site_indices))
        for j in range(i + 1, len(charge_site_indices))
    ]
    charge_acidic_distances = [
        distance(conf, charge_idx, acidic_idx)
        for charge_idx in charge_site_indices
        for acidic_idx in acidic_indices
        if charge_idx != acidic_idx
    ]
    charge_pair_min, charge_pair_max = extrema_or_nan(charge_pair_distances)
    charge_acidic_min, charge_acidic_max = extrema_or_nan(charge_acidic_distances)

    # FreeSASA writes per-atom SASA properties onto the molecule for this conformer.
    radii = rdFreeSASA.classifyAtoms(mol_h)
    total_sasa = rdFreeSASA.CalcSASA(mol_h, radii, confIdx=conf_id)
    charge_sasa = []
    for index in charge_site_indices:
        atom = mol_h.GetAtomWithIdx(index)
        if atom.HasProp("SASA"):
            charge_sasa.append(float(atom.GetProp("SASA")))
    charge_sasa_min, charge_sasa_max = extrema_or_nan(charge_sasa)

    return {
        "radius_of_gyration": rdMolDescriptors.CalcRadiusOfGyration(mol_h, confId=conf_id),
        "asphericity": rdMolDescriptors.CalcAsphericity(mol_h, confId=conf_id),
        "eccentricity": rdMolDescriptors.CalcEccentricity(mol_h, confId=conf_id),
        "inertial_shape_factor": rdMolDescriptors.CalcInertialShapeFactor(mol_h, confId=conf_id),
        "npr1": rdMolDescriptors.CalcNPR1(mol_h, confId=conf_id),
        "npr2": rdMolDescriptors.CalcNPR2(mol_h, confId=conf_id),
        "pmi1": rdMolDescriptors.CalcPMI1(mol_h, confId=conf_id),
        "pmi2": rdMolDescriptors.CalcPMI2(mol_h, confId=conf_id),
        "pmi3": rdMolDescriptors.CalcPMI3(mol_h, confId=conf_id),
        "spherocity_index": rdMolDescriptors.CalcSpherocityIndex(mol_h, confId=conf_id),
        "pbf": rdMolDescriptors.CalcPBF(mol_h, confId=conf_id),
        "molecular_volume": AllChem.ComputeMolVolume(mol_h, confId=conf_id),
        "total_sasa": total_sasa,
        "end_to_end_distance": distance(conf, nterm_idx, cterm_c_idx),
        "charge_pair_distance_min": charge_pair_min,
        "charge_pair_distance_mean": mean_or_nan(charge_pair_distances),
        "charge_pair_distance_max": charge_pair_max,
        "charge_acidic_distance_min": charge_acidic_min,
        "charge_acidic_distance_mean": mean_or_nan(charge_acidic_distances),
        "charge_acidic_distance_max": charge_acidic_max,
        "charge_site_sasa_min": charge_sasa_min,
        "charge_site_sasa_mean": mean_or_nan(charge_sasa),
        "charge_site_sasa_max": charge_sasa_max,
        "hydrophobic_heavy_atom_rg": coordinate_rg(conf, hydrophobic_indices),
    }


def ensemble_rmsd_summary(mol: Chem.Mol, conformer_ids: Sequence[int]) -> tuple[float, float]:
    values = []
    for i in range(len(conformer_ids)):
        for j in range(i + 1, len(conformer_ids)):
            values.append(
                AllChem.GetConformerRMS(
                    mol, conformer_ids[i], conformer_ids[j], prealigned=False
                )
            )
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), max(values)


def max_coordinate_difference(mol_a: Chem.Mol, mol_b: Chem.Mol, conf_id: int = 0) -> float:
    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms():
        return float("inf")
    a = mol_a.GetConformer(conf_id)
    b = mol_b.GetConformer(conf_id)
    maximum = 0.0
    for index in range(mol_a.GetNumAtoms()):
        pa = a.GetAtomPosition(index)
        pb = b.GetAtomPosition(index)
        maximum = max(
            maximum,
            abs(pa.x - pb.x),
            abs(pa.y - pb.y),
            abs(pa.z - pb.z),
        )
    return maximum


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.8f}"


def write_tsv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summary_row(metric: str, value: object) -> dict[str, object]:
    return {"metric": metric, "value": value}


def run_probe(
    manifest: Path,
    expected_manifest_fingerprint: str,
    out_dir: Path,
    probe_size: int,
    matched_charge_pairs: int,
    minimum_per_ptm_class: int,
    num_conformers: int,
    microstates: int,
    max_minimize_iters: int,
    reproducibility_checks: int,
    mass_tolerance_da: float,
) -> dict[str, str]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _, manifest_rows, observed_fingerprint = load_manifest(manifest)
    if observed_fingerprint != expected_manifest_fingerprint:
        raise ValueError(
            "frozen Stage B manifest fingerprint mismatch: "
            f"observed={observed_fingerprint} expected={expected_manifest_fingerprint}"
        )

    selected = select_probe_rows(
        manifest_rows,
        probe_size=probe_size,
        matched_charge_pairs=matched_charge_pairs,
        minimum_per_ptm_class=minimum_per_ptm_class,
    )

    selection_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    conformer_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    reproducibility_rows: list[dict[str, object]] = []
    per_identity_descriptor_medians: dict[str, dict[str, float]] = {}

    build_success = 0
    microstate_success = 0
    conformers_generated = 0
    descriptor_complete = 0
    forcefield_converged_conformers = 0
    ensemble_spread_microstates = 0
    explicit_atom_total = 0
    fallback_charge_site_identities = 0
    forcefield_counts: Counter[str] = Counter()
    embed_mode_counts: Counter[str] = Counter()
    ptm_counts: Counter[str] = Counter()
    charge_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    total_started = time.perf_counter()

    descriptor_names = [
        "radius_of_gyration",
        "asphericity",
        "eccentricity",
        "inertial_shape_factor",
        "npr1",
        "npr2",
        "pmi1",
        "pmi2",
        "pmi3",
        "spherocity_index",
        "pbf",
        "molecular_volume",
        "total_sasa",
        "end_to_end_distance",
        "charge_pair_distance_min",
        "charge_pair_distance_mean",
        "charge_pair_distance_max",
        "charge_acidic_distance_min",
        "charge_acidic_distance_mean",
        "charge_acidic_distance_max",
        "charge_site_sasa_min",
        "charge_site_sasa_mean",
        "charge_site_sasa_max",
        "hydrophobic_heavy_atom_rg",
    ]

    reproducibility_remaining = reproducibility_checks

    for probe_order, entry in enumerate(selected, start=1):
        row = entry.row
        identity_key = row["identity_key"]
        charge = parse_int(row["charge"], "charge")
        selection_rows.append(
            {
                "probe_order": probe_order,
                "stage_b_order": row["stage_b_order"],
                "identity_key": identity_key,
                "peptidoform": row["peptidoform"],
                "sequence": row["sequence"],
                "charge": charge,
                "length_bin": row["length_bin"],
                "ptm_class": row["ptm_class"],
                "selection_reasons": ";".join(sorted(entry.reasons)),
            }
        )
        ptm_counts[row["ptm_class"]] += 1
        charge_counts[str(charge)] += 1
        length_counts[row["length_bin"]] += 1
        identity_started = time.perf_counter()
        identity_conformers = []
        identity_microstates = 0
        identity_forcefields: Counter[str] = Counter()
        identity_embed_modes: Counter[str] = Counter()
        fallback_used = False
        try:
            built = build_neutral_molecule(row, mass_tolerance_da=mass_tolerance_da)
            build_success += 1
            candidates = candidate_charge_sites(built.mol, built.blocked_basic_atoms)
            if any(site.kind == "carbonyl_o_fallback" for site in candidates[:charge]):
                fallback_used = True
                fallback_charge_site_identities += 1
            acidic = acidic_atom_indices(built.mol, built.phosphate_acidic_atoms)
            nterm_idx, cterm_idx = terminal_atom_indices(
                built.mol, parse_int(row["sequence_length"], "sequence_length")
            )
            hydrophobic = hydrophobic_atom_indices(built.mol)

            microstate_signatures: set[tuple[int, ...]] = set()
            for microstate_index in range(max(1, microstates)):
                sites = microstate_sites(candidates, charge, microstate_index)
                signature = tuple(site.atom_idx for site in sites)
                if signature in microstate_signatures:
                    continue
                microstate_signatures.add(signature)
                charged = apply_positive_charge(built.mol, sites, charge)
                try:
                    (
                        mol_h,
                        conformer_ids,
                        forcefield,
                        optimization,
                        embed_mode,
                        elapsed,
                    ) = embed_and_minimize(
                        charged,
                        row["peptidoform"],
                        microstate_index,
                        num_conformers,
                        max_minimize_iters,
                    )
                except Exception as exc:
                    failure_rows.append(
                        {
                            "identity_key": identity_key,
                            "stage": "conformer_generation",
                            "microstate_index": microstate_index,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    continue

                microstate_success += 1
                identity_microstates += 1
                forcefield_counts[forcefield] += 1
                embed_mode_counts[embed_mode] += 1
                identity_forcefields[forcefield] += 1
                identity_embed_modes[embed_mode] += 1
                charge_site_indices = [site.atom_idx for site in sites]
                rmsd_mean, rmsd_max = ensemble_rmsd_summary(mol_h, conformer_ids)
                if math.isfinite(rmsd_max) and rmsd_max >= 0.5:
                    ensemble_spread_microstates += 1

                for local_rank, conf_id in enumerate(conformer_ids, start=1):
                    descriptors = conformer_descriptors(
                        mol_h,
                        conf_id,
                        charge_site_indices,
                        acidic,
                        nterm_idx,
                        cterm_idx,
                        hydrophobic,
                    )
                    opt_not_converged, energy = optimization[local_rank - 1]
                    finite_required = [
                        descriptors["radius_of_gyration"],
                        descriptors["asphericity"],
                        descriptors["molecular_volume"],
                        descriptors["total_sasa"],
                        descriptors["end_to_end_distance"],
                        descriptors["charge_site_sasa_mean"],
                    ]
                    complete = all(math.isfinite(value) for value in finite_required)
                    descriptor_complete += int(complete)
                    forcefield_converged_conformers += int(opt_not_converged == 0)
                    conformers_generated += 1
                    explicit_atom_total += mol_h.GetNumAtoms()
                    identity_conformers.append(descriptors)
                    conformer_rows.append(
                        {
                            "identity_key": identity_key,
                            "peptidoform": row["peptidoform"],
                            "charge": charge,
                            "ptm_class": row["ptm_class"],
                            "length_bin": row["length_bin"],
                            "microstate_index": microstate_index,
                            "charge_sites": ";".join(site.label for site in sites),
                            "carbonyl_fallback_sites": sum(
                                site.kind == "carbonyl_o_fallback" for site in sites
                            ),
                            "conformer_rank": local_rank,
                            "explicit_atom_count": mol_h.GetNumAtoms(),
                            "forcefield": forcefield,
                            "forcefield_not_converged": opt_not_converged,
                            "forcefield_energy": format_float(energy),
                            "embed_mode": embed_mode,
                            "microstate_runtime_seconds": f"{elapsed:.6f}",
                            "ensemble_pairwise_rmsd_mean": format_float(rmsd_mean),
                            "ensemble_pairwise_rmsd_max": format_float(rmsd_max),
                            "descriptor_complete": "YES" if complete else "NO",
                            **{name: format_float(descriptors[name]) for name in descriptor_names},
                        }
                    )

                if reproducibility_remaining > 0:
                    try:
                        repeat_mol, repeat_ids, _, _, _, _ = embed_and_minimize(
                            charged,
                            row["peptidoform"],
                            microstate_index,
                            1,
                            max_minimize_iters,
                        )
                        reference_mol, reference_ids, _, _, _, _ = embed_and_minimize(
                            charged,
                            row["peptidoform"],
                            microstate_index,
                            1,
                            max_minimize_iters,
                        )
                        max_delta = max_coordinate_difference(reference_mol, repeat_mol, 0)
                        reproducibility_rows.append(
                            {
                                "identity_key": identity_key,
                                "microstate_index": microstate_index,
                                "max_coordinate_abs_difference_angstrom": f"{max_delta:.12g}",
                                "reproducible_within_1e-7_angstrom": "YES"
                                if max_delta <= 1e-7
                                else "NO",
                            }
                        )
                    except Exception as exc:
                        reproducibility_rows.append(
                            {
                                "identity_key": identity_key,
                                "microstate_index": microstate_index,
                                "max_coordinate_abs_difference_angstrom": "",
                                "reproducible_within_1e-7_angstrom": "NO",
                                "error": str(exc),
                            }
                        )
                    reproducibility_remaining -= 1

            if identity_conformers:
                medians = {}
                for name in descriptor_names:
                    finite = [d[name] for d in identity_conformers if math.isfinite(d[name])]
                    if finite:
                        medians[name] = statistics.median(finite)
                per_identity_descriptor_medians[identity_key] = medians

            identity_rows.append(
                {
                    "identity_key": identity_key,
                    "peptidoform": row["peptidoform"],
                    "sequence": row["sequence"],
                    "charge": charge,
                    "ptm_class": row["ptm_class"],
                    "length_bin": row["length_bin"],
                    "neutral_build_success": "YES",
                    "neutral_exact_mass_da": f"{built.neutral_exact_mass:.8f}",
                    "stage_a_theoretical_mass_da": row["theoretical_neutral_mass_da"],
                    "neutral_mass_error_da": f"{built.mass_error_da:+.8f}",
                    "canonical_isomeric_smiles": built.canonical_smiles,
                    "charge_candidate_count": len(candidates),
                    "carbonyl_fallback_used": "YES" if fallback_used else "NO",
                    "microstates_succeeded": identity_microstates,
                    "conformers_generated": len(identity_conformers),
                    "forcefields": ";".join(
                        f"{name}:{count}" for name, count in sorted(identity_forcefields.items())
                    ),
                    "embed_modes": ";".join(
                        f"{name}:{count}" for name, count in sorted(identity_embed_modes.items())
                    ),
                    "runtime_seconds": f"{time.perf_counter() - identity_started:.6f}",
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "identity_key": identity_key,
                    "stage": "neutral_molecule_build",
                    "microstate_index": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            identity_rows.append(
                {
                    "identity_key": identity_key,
                    "peptidoform": row["peptidoform"],
                    "sequence": row["sequence"],
                    "charge": charge,
                    "ptm_class": row["ptm_class"],
                    "length_bin": row["length_bin"],
                    "neutral_build_success": "NO",
                    "runtime_seconds": f"{time.perf_counter() - identity_started:.6f}",
                }
            )

    # Matched observed-charge diagnostic. This is descriptive only; no measured
    # mobility or CCS appears anywhere in the calculation.
    pair_variation_rows: list[dict[str, object]] = []
    selected_by_peptidoform: dict[str, list[SelectedProbeRow]] = defaultdict(list)
    for entry in selected:
        if "matched_observed_charge_pair" in entry.reasons:
            selected_by_peptidoform[entry.row["peptidoform"]].append(entry)
    for peptidoform, group in sorted(selected_by_peptidoform.items()):
        group = sorted(group, key=lambda entry: parse_int(entry.row["charge"], "charge"))
        if len(group) < 2:
            continue
        left, right = group[0], group[-1]
        left_desc = per_identity_descriptor_medians.get(left.row["identity_key"], {})
        right_desc = per_identity_descriptor_medians.get(right.row["identity_key"], {})
        delta_rg = (
            right_desc.get("radius_of_gyration", float("nan"))
            - left_desc.get("radius_of_gyration", float("nan"))
        )
        delta_volume = (
            right_desc.get("molecular_volume", float("nan"))
            - left_desc.get("molecular_volume", float("nan"))
        )
        delta_asphericity = (
            right_desc.get("asphericity", float("nan"))
            - left_desc.get("asphericity", float("nan"))
        )
        delta_charge_sasa = (
            right_desc.get("charge_site_sasa_mean", float("nan"))
            - left_desc.get("charge_site_sasa_mean", float("nan"))
        )
        nontrivial_shape_change = any(
            [
                math.isfinite(delta_rg) and abs(delta_rg) >= 0.10,
                math.isfinite(delta_asphericity) and abs(delta_asphericity) >= 0.01,
                math.isfinite(delta_charge_sasa) and abs(delta_charge_sasa) >= 1.0,
            ]
        )
        pair_variation_rows.append(
            {
                "peptidoform": peptidoform,
                "identity_key_a": left.row["identity_key"],
                "charge_a": left.row["charge"],
                "identity_key_b": right.row["identity_key"],
                "charge_b": right.row["charge"],
                "delta_radius_of_gyration": format_float(delta_rg),
                "delta_molecular_volume": format_float(delta_volume),
                "delta_asphericity": format_float(delta_asphericity),
                "delta_charge_site_sasa_mean": format_float(delta_charge_sasa),
                "nontrivial_charge_conditioned_shape_change": "YES"
                if nontrivial_shape_change
                else "NO",
            }
        )

    elapsed_total = time.perf_counter() - total_started
    expected_microstates_max = probe_size * max(1, microstates)
    expected_conformers_max = expected_microstates_max * num_conformers
    build_fraction = build_success / probe_size if probe_size else 0.0
    conformer_identity_successes = sum(
        parse_int(str(row.get("conformers_generated", "0") or "0"), "conformers_generated") > 0
        for row in identity_rows
    )
    conformer_identity_fraction = conformer_identity_successes / probe_size if probe_size else 0.0
    descriptor_fraction = (
        descriptor_complete / conformers_generated if conformers_generated else 0.0
    )
    reproducibility_pass = sum(
        row.get("reproducible_within_1e-7_angstrom") == "YES"
        for row in reproducibility_rows
    )
    reproduction_fraction = (
        reproducibility_pass / len(reproducibility_rows) if reproducibility_rows else 0.0
    )
    forcefield_convergence_fraction = (
        forcefield_converged_conformers / conformers_generated if conformers_generated else 0.0
    )
    ensemble_spread_fraction = (
        ensemble_spread_microstates / microstate_success if microstate_success else 0.0
    )
    charge_variation_passes = sum(
        row.get("nontrivial_charge_conditioned_shape_change") == "YES"
        for row in pair_variation_rows
    )
    charge_variation_fraction = (
        charge_variation_passes / len(pair_variation_rows) if pair_variation_rows else 0.0
    )
    mean_explicit_atoms = explicit_atom_total / conformers_generated if conformers_generated else 0.0
    projected_raw_coordinate_gb = (
        mean_explicit_atoms
        * 3.0
        * 8.0
        * 8000.0
        * max(1, microstates)
        * num_conformers
        / 1_000_000_000.0
    )

    # Predeclared mechanical feasibility gates for deciding whether to scale from
    # the tiny probe to the frozen 8k manifest. They are not CCS-performance gates.
    gate_build = build_fraction >= 0.95
    gate_conformer = conformer_identity_fraction >= 0.90
    gate_descriptors = descriptor_fraction >= 0.95
    gate_repro = reproduction_fraction >= 0.95 if reproducibility_rows else False
    gate_ensemble_spread = ensemble_spread_fraction >= 0.50
    gate_charge_variation = (
        len(pair_variation_rows) >= 2 and charge_variation_fraction >= 0.50
    )
    gate_runtime = elapsed_total / probe_size <= 120.0
    overall_gate = (
        gate_build
        and gate_conformer
        and gate_descriptors
        and gate_repro
        and gate_ensemble_spread
        and gate_charge_variation
        and gate_runtime
    )

    summary = {
        "audit_version": AUDIT_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "partition_scope": "TRAIN_frozen_stage_b_manifest_only",
        "input_manifest": str(manifest),
        "manifest_fingerprint": observed_fingerprint,
        "measured_ccs_used": "NO",
        "mobility_labels_used": "NO",
        "dev_labels_used": "NO",
        "holdout_used": "NO",
        "probe_size": str(probe_size),
        "matched_charge_pairs_requested": str(matched_charge_pairs),
        "probe_identities_selected": str(len(selected)),
        "neutral_build_successes": str(build_success),
        "neutral_build_success_fraction": f"{build_fraction:.8f}",
        "identities_with_conformers": str(conformer_identity_successes),
        "identity_conformer_success_fraction": f"{conformer_identity_fraction:.8f}",
        "microstates_succeeded": str(microstate_success),
        "conformers_generated": str(conformers_generated),
        "maximum_requested_conformers": str(expected_conformers_max),
        "descriptor_complete_conformers": str(descriptor_complete),
        "descriptor_complete_fraction": f"{descriptor_fraction:.8f}",
        "forcefield_converged_conformers": str(forcefield_converged_conformers),
        "forcefield_convergence_fraction": f"{forcefield_convergence_fraction:.8f}",
        "microstates_with_pairwise_rmsd_max_ge_0_5A": str(ensemble_spread_microstates),
        "ensemble_spread_fraction": f"{ensemble_spread_fraction:.8f}",
        "identities_using_carbonyl_charge_fallback": str(fallback_charge_site_identities),
        "reproducibility_checks": str(len(reproducibility_rows)),
        "reproducibility_pass_fraction": f"{reproduction_fraction:.8f}",
        "matched_observed_charge_pairs_analyzed": str(len(pair_variation_rows)),
        "matched_charge_pairs_with_nontrivial_shape_change": str(charge_variation_passes),
        "matched_charge_shape_change_fraction": f"{charge_variation_fraction:.8f}",
        "mean_explicit_atom_count": f"{mean_explicit_atoms:.6f}",
        "projected_8000_raw_coordinate_storage_gb": f"{projected_raw_coordinate_gb:.6f}",
        "total_runtime_seconds": f"{elapsed_total:.6f}",
        "mean_runtime_seconds_per_identity": f"{elapsed_total / probe_size:.6f}",
        "projected_8000_identity_runtime_hours_serial": f"{elapsed_total / probe_size * 8000 / 3600.0:.6f}",
        "forcefield_counts": ";".join(f"{k}:{v}" for k, v in sorted(forcefield_counts.items())),
        "embed_mode_counts": ";".join(f"{k}:{v}" for k, v in sorted(embed_mode_counts.items())),
        "gate_neutral_build_ge_95pct": "PASS" if gate_build else "FAIL",
        "gate_identity_conformer_success_ge_90pct": "PASS" if gate_conformer else "FAIL",
        "gate_descriptor_complete_ge_95pct": "PASS" if gate_descriptors else "FAIL",
        "gate_reproducibility_ge_95pct": "PASS" if gate_repro else "FAIL",
        "gate_ensemble_spread_ge_50pct": "PASS" if gate_ensemble_spread else "FAIL",
        "gate_matched_charge_shape_change_ge_50pct": "PASS" if gate_charge_variation else "FAIL",
        "gate_mean_runtime_le_120s_per_identity": "PASS" if gate_runtime else "FAIL",
        "tiny_probe_mechanical_gate": "PASS" if overall_gate else "FAIL",
    }

    write_tsv(
        out_dir / "probe_selection.tsv",
        [
            "probe_order",
            "stage_b_order",
            "identity_key",
            "peptidoform",
            "sequence",
            "charge",
            "length_bin",
            "ptm_class",
            "selection_reasons",
        ],
        selection_rows,
    )
    identity_fields = [
        "identity_key",
        "peptidoform",
        "sequence",
        "charge",
        "ptm_class",
        "length_bin",
        "neutral_build_success",
        "neutral_exact_mass_da",
        "stage_a_theoretical_mass_da",
        "neutral_mass_error_da",
        "canonical_isomeric_smiles",
        "charge_candidate_count",
        "carbonyl_fallback_used",
        "microstates_succeeded",
        "conformers_generated",
        "forcefields",
        "embed_modes",
        "runtime_seconds",
    ]
    write_tsv(out_dir / "probe_identity_summary.tsv", identity_fields, identity_rows)
    conformer_fields = [
        "identity_key",
        "peptidoform",
        "charge",
        "ptm_class",
        "length_bin",
        "microstate_index",
        "charge_sites",
        "carbonyl_fallback_sites",
        "conformer_rank",
        "explicit_atom_count",
        "forcefield",
        "forcefield_not_converged",
        "forcefield_energy",
        "embed_mode",
        "microstate_runtime_seconds",
        "ensemble_pairwise_rmsd_mean",
        "ensemble_pairwise_rmsd_max",
        "descriptor_complete",
        *descriptor_names,
    ]
    write_tsv(out_dir / "probe_conformers.tsv", conformer_fields, conformer_rows)
    write_tsv(
        out_dir / "probe_failures.tsv",
        ["identity_key", "stage", "microstate_index", "error_type", "error"],
        failure_rows,
    )
    write_tsv(
        out_dir / "probe_reproducibility.tsv",
        [
            "identity_key",
            "microstate_index",
            "max_coordinate_abs_difference_angstrom",
            "reproducible_within_1e-7_angstrom",
            "error",
        ],
        reproducibility_rows,
    )
    write_tsv(
        out_dir / "probe_charge_pair_variation.tsv",
        [
            "peptidoform",
            "identity_key_a",
            "charge_a",
            "identity_key_b",
            "charge_b",
            "delta_radius_of_gyration",
            "delta_molecular_volume",
            "delta_asphericity",
            "delta_charge_site_sasa_mean",
            "nontrivial_charge_conditioned_shape_change",
        ],
        pair_variation_rows,
    )
    write_tsv(
        out_dir / "probe_summary.tsv",
        ["metric", "value"],
        [summary_row(key, value) for key, value in summary.items()],
    )
    write_tsv(
        out_dir / "probe_by_ptm_class.tsv",
        ["ptm_class", "selected"],
        [{"ptm_class": key, "selected": value} for key, value in sorted(ptm_counts.items())],
    )
    write_tsv(
        out_dir / "probe_by_charge.tsv",
        ["charge", "selected"],
        [
            {"charge": key, "selected": value}
            for key, value in sorted(charge_counts.items(), key=lambda item: int(item[0]))
        ],
    )
    write_tsv(
        out_dir / "probe_by_length_bin.tsv",
        ["length_bin", "selected"],
        [{"length_bin": key, "selected": value} for key, value in sorted(length_counts.items())],
    )

    with (out_dir / "probe_report.md").open("w", encoding="utf-8") as handle:
        handle.write("# ReDeeM structure-signal feasibility v1 — Stage B tiny engine probe\n\n")
        handle.write(f"- Manifest fingerprint: `{observed_fingerprint}`\n")
        handle.write(f"- RDKit: `{rdBase.rdkitVersion}`\n")
        handle.write(f"- Probe identities: `{probe_size}`\n")
        handle.write(f"- Neutral build success: `{build_success}/{probe_size}` ({build_fraction:.2%})\n")
        handle.write(
            f"- Identities with conformers: `{conformer_identity_successes}/{probe_size}` ({conformer_identity_fraction:.2%})\n"
        )
        handle.write(
            f"- Descriptor-complete conformers: `{descriptor_complete}/{conformers_generated}` ({descriptor_fraction:.2%})\n"
        )
        handle.write(
            f"- Reproducibility checks: `{reproducibility_pass}/{len(reproducibility_rows)}` ({reproduction_fraction:.2%})\n"
        )
        handle.write(
            f"- Microstates with ensemble RMSD max >= 0.5 A: `{ensemble_spread_microstates}/{microstate_success}` ({ensemble_spread_fraction:.2%})\n"
        )
        handle.write(
            f"- Matched observed-charge pairs with nontrivial shape change: `{charge_variation_passes}/{len(pair_variation_rows)}` ({charge_variation_fraction:.2%})\n"
        )
        handle.write(f"- Mean runtime / identity: `{elapsed_total / probe_size:.3f} s`\n")
        handle.write(
            f"- Serial 8k runtime projection: `{elapsed_total / probe_size * 8000 / 3600.0:.3f} h`\n"
        )
        handle.write(f"- Tiny mechanical gate: **{summary['tiny_probe_mechanical_gate']}**\n\n")
        handle.write("## Scientific boundary\n\n")
        handle.write(
            "Charge-site microstates are deterministic heuristics used to test whether charge-conditioned structural generation is operationally viable. They are not inferred experimental protomers. No measured CCS or mobility label is read or used.\n"
        )

    print(f"audit_version\t{AUDIT_VERSION}")
    print(f"manifest_fingerprint\t{observed_fingerprint}")
    print(f"rdkit_version\t{rdBase.rdkitVersion}")
    print(f"probe_identities\t{probe_size}")
    print(f"neutral_build_successes\t{build_success}")
    print(f"identities_with_conformers\t{conformer_identity_successes}")
    print(f"conformers_generated\t{conformers_generated}")
    print(f"descriptor_complete_fraction\t{descriptor_fraction:.8f}")
    print(f"reproducibility_pass_fraction\t{reproduction_fraction:.8f}")
    print(f"mean_runtime_seconds_per_identity\t{elapsed_total / probe_size:.6f}")
    print(f"tiny_probe_mechanical_gate\t{summary['tiny_probe_mechanical_gate']}")
    print("measured_ccs_used\tNO")
    print("dev_labels_used\tNO")
    print("holdout_used\tNO")
    return summary


def synthetic_manifest(path: Path) -> str:
    fields = [
        "stage_b_order",
        "stratum_key",
        "stratum_population",
        "stratum_quota",
        "stratum_rank",
        "ptm_class",
        "selection_hash_sha256",
        "identity_key",
        "peptidoform",
        "sequence",
        "charge",
        "sequence_length",
        "length_bin",
        "modified",
        "modification_count",
        "modification_labels",
        "record_count",
        "source_count",
        "precursor_mz_count",
        "precursor_mz_mean",
        "precursor_mz_min",
        "precursor_mz_max",
        "theoretical_neutral_mass_da",
        "chemistry_status",
        "current_exact_ptm_local_topology_ready",
        "existing_attachment_template_candidate",
        "all_modifications_composition_known",
        "failure_reasons",
    ]
    examples = [
        ("PEPTIDE", "PEPTIDE", "unmodified"),
        ("ACDMK", "ACDMK|R1:UniMod:4|R3:UniMod:35", "common_ptm_combination"),
        ("ANQK", "ANQK|R1:UniMod:7|R2:UniMod:7", "deamidation"),
        ("ASTYK", "ASTYK|R1:UniMod:21", "phospho"),
        ("AKAAA", "AKAAA|R1:UniMod:1", "acetyl"),
        ("AMAAA", "AMAAA|R1:UniMod:35", "oxidation"),
        ("ACAAA", "ACAAA|R1:UniMod:4", "carbamidomethyl"),
    ]
    rows: list[dict[str, str]] = []
    order = 1
    for charge in (1, 2, 3, 5):
        for sequence, peptidoform, ptm_class in examples:
            neutral = Chem.MolFromSequence(sequence)
            fake = {
                "peptidoform": peptidoform,
                "sequence": sequence,
                "theoretical_neutral_mass_da": "0",
                "identity_key": f"{peptidoform}|z{charge}",
            }
            # Build once using individual known deltas to establish the expected
            # Stage-A-like neutral mass for the synthetic fixture.
            parts = peptidoform.split("|")
            mass = rdMolDescriptors.CalcExactMolWt(neutral)
            deltas = {1: 42.010564684, 4: 57.021463716, 7: 0.984015588, 21: 79.966330522, 35: 15.994914620}
            for token in parts[1:]:
                _, identity = token.split(":", 1)
                uid = int(identity.split(":", 1)[1])
                mass += deltas[uid]
            identity_key = f"{peptidoform}|z{charge}"
            selection_hash = hashlib.sha256(identity_key.encode()).hexdigest()
            length = len(sequence)
            row = {field: "" for field in fields}
            row.update(
                {
                    "stage_b_order": str(order),
                    "stratum_key": f"z{charge}|01_1-7|{ptm_class}",
                    "stratum_population": "10",
                    "stratum_quota": "1",
                    "stratum_rank": "1",
                    "ptm_class": ptm_class,
                    "selection_hash_sha256": selection_hash,
                    "identity_key": identity_key,
                    "peptidoform": peptidoform,
                    "sequence": sequence,
                    "charge": str(charge),
                    "sequence_length": str(length),
                    "length_bin": "01_1-7",
                    "modified": "NO" if ptm_class == "unmodified" else "YES",
                    "modification_count": str(len(parts) - 1),
                    "record_count": "1",
                    "source_count": "1",
                    "precursor_mz_count": "1",
                    "theoretical_neutral_mass_da": f"{mass:.8f}",
                    "chemistry_status": "unmodified" if ptm_class == "unmodified" else "current_exact_ptm_local_topology",
                    "current_exact_ptm_local_topology_ready": "YES",
                    "existing_attachment_template_candidate": "YES",
                    "all_modifications_composition_known": "YES",
                }
            )
            rows.append(row)
            order += 1
    write_tsv(path, fields, rows)
    return manifest_fingerprint(rows)


def run_self_test() -> None:
    # Exact PTM mass semantics.
    base_rows = [
        ("ACAAA|R1:UniMod:4", "ACAAA", 57.021463716),
        ("AMAAA|R1:UniMod:35", "AMAAA", 15.994914620),
        ("ANAAA|R1:UniMod:7", "ANAAA", 0.984015588),
        ("ASAAA|R1:UniMod:21", "ASAAA", 79.966330522),
        ("AKAAA|R1:UniMod:1", "AKAAA", 42.010564684),
        ("AAAAA|N-term:UniMod:1", "AAAAA", 42.010564684),
    ]
    for peptidoform, sequence, delta in base_rows:
        base = Chem.MolFromSequence(sequence)
        expected = rdMolDescriptors.CalcExactMolWt(base) + delta
        row = {
            "identity_key": f"{peptidoform}|z2",
            "peptidoform": peptidoform,
            "sequence": sequence,
            "theoretical_neutral_mass_da": f"{expected:.8f}",
        }
        built = build_neutral_molecule(row, 0.002)
        if abs(built.mass_error_da) > 1e-6:
            raise RuntimeError(f"self-test PTM mass mismatch for {peptidoform}")

    # Charge application must produce the requested net charge and deterministic
    # ETKDG coordinates with a fixed seed.
    row = {
        "identity_key": "PEPTIDERK|z3",
        "peptidoform": "PEPTIDERK",
        "sequence": "PEPTIDERK",
        "theoretical_neutral_mass_da": f"{rdMolDescriptors.CalcExactMolWt(Chem.MolFromSequence('PEPTIDERK')):.8f}",
    }
    built = build_neutral_molecule(row, 0.002)
    candidates = candidate_charge_sites(built.mol, set())
    sites = microstate_sites(candidates, 3, 0)
    charged = apply_positive_charge(built.mol, sites, 3)
    a, ids_a, _, _, _, _ = embed_and_minimize(charged, row["peptidoform"], 0, 1, 25)
    b, ids_b, _, _, _, _ = embed_and_minimize(charged, row["peptidoform"], 0, 1, 25)
    if not ids_a or not ids_b:
        raise RuntimeError("self-test ETKDG returned no conformer")
    if max_coordinate_difference(a, b, 0) > 1e-7:
        raise RuntimeError("self-test fixed-seed ETKDG/minimization is not reproducible")

    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.tsv"
        fingerprint = synthetic_manifest(manifest)
        _, rows, observed = load_manifest(manifest)
        if fingerprint != observed:
            raise RuntimeError("self-test manifest fingerprint mismatch")
        selected_a = select_probe_rows(rows, 21, 2, 2)
        selected_b = select_probe_rows(list(reversed(rows)), 21, 2, 2)
        keys_a = sorted(entry.row["identity_key"] for entry in selected_a)
        keys_b = sorted(entry.row["identity_key"] for entry in selected_b)
        if keys_a != keys_b:
            raise RuntimeError("self-test probe selection changed with input order")

    print("structure_signal_stage_b_probe_v1_self_test=PASS")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.manifest is None or args.out_dir is None or not args.expected_manifest_fingerprint:
        raise SystemExit(
            "--manifest, --expected-manifest-fingerprint and --out-dir are required unless --self-test is used"
        )
    run_probe(
        manifest=args.manifest,
        expected_manifest_fingerprint=args.expected_manifest_fingerprint,
        out_dir=args.out_dir,
        probe_size=args.probe_size,
        matched_charge_pairs=args.matched_charge_pairs,
        minimum_per_ptm_class=args.minimum_per_ptm_class,
        num_conformers=args.num_conformers,
        microstates=args.microstates,
        max_minimize_iters=args.max_minimize_iters,
        reproducibility_checks=args.reproducibility_checks,
        mass_tolerance_da=args.mass_tolerance_da,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
