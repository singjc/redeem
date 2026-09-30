#!/usr/bin/env python3
"""Full frozen Stage B ETKDG-only structure descriptor generation.

This consumes the immutable 8,000-identity TRAIN-only Stage B manifest produced by
``structure_signal_stage_b_manifest_v1``.  It deliberately does not read measured
CCS/mobility labels and deliberately performs no force-field minimization.

The executable has two operational subcommands:

* ``shard``: deterministically processes one round-robin shard of the frozen
  manifest and emits per-identity/per-conformer descriptor tables.
* ``finalize``: verifies all frozen identities were processed exactly once,
  merges the shard outputs, builds one row of structure features per identity,
  and emits final operational/scientific summaries.

The chemistry, mass gate, charge-site policy, fixed-seed ETKDG settings and 3D
structural descriptors are imported from the already validated tiny-probe helper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import structure_signal_stage_b_probe_v1 as probe

AUDIT_VERSION = "structure_signal_feasibility_v1_stage_b_full_etkdg_v1"
DEFAULT_EXPECTED_IDENTITIES = 8000
DEFAULT_SHARD_COUNT = 80
DEFAULT_NUM_CONFORMERS = 3
DEFAULT_MICROSTATES = 2
DEFAULT_MASS_TOLERANCE_DA = 0.002

DESCRIPTOR_NAMES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    shard = subparsers.add_parser("shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--expected-manifest-fingerprint", required=True)
    shard.add_argument("--out-dir", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    shard.add_argument(
        "--expected-identities", type=int, default=DEFAULT_EXPECTED_IDENTITIES
    )
    shard.add_argument("--num-conformers", type=int, default=DEFAULT_NUM_CONFORMERS)
    shard.add_argument("--microstates", type=int, default=DEFAULT_MICROSTATES)
    shard.add_argument(
        "--mass-tolerance-da", type=float, default=DEFAULT_MASS_TOLERANCE_DA
    )

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--expected-manifest-fingerprint", required=True)
    finalize.add_argument("--shards-root", type=Path, required=True)
    finalize.add_argument("--out-dir", type=Path, required=True)
    finalize.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    finalize.add_argument(
        "--expected-identities", type=int, default=DEFAULT_EXPECTED_IDENTITIES
    )
    return parser.parse_args()


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.8f}"


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_tsv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summary_value(path: Path, metric: str) -> str:
    _, rows = read_tsv(path)
    for row in rows:
        if row.get("metric") == metric:
            return row.get("value", "")
    raise ValueError(f"missing metric {metric!r} in {path}")


def shard_rows(
    rows: Sequence[dict[str, str]], shard_index: int, shard_count: int
) -> list[dict[str, str]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard_index must be in [0,{shard_count}); observed {shard_index}"
        )
    selected = []
    for row in rows:
        order = probe.parse_int(row["stage_b_order"], "stage_b_order")
        if (order - 1) % shard_count == shard_index:
            selected.append(row)
    return selected


def embed_etkdg_only(
    charged_mol: probe.Chem.Mol,
    peptidoform: str,
    microstate_index: int,
    num_conformers: int,
) -> tuple[probe.Chem.Mol, list[int], str, float]:
    if num_conformers <= 0:
        raise ValueError("num_conformers must be positive")
    started = time.perf_counter()
    mol_h = probe.Chem.AddHs(charged_mol)
    seed = probe.seed_for(peptidoform, microstate_index)

    params = probe.AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.pruneRmsThresh = -1.0
    embed_mode = "etkdgv3"
    conformer_ids = list(
        probe.AllChem.EmbedMultipleConfs(
            mol_h, numConfs=num_conformers, params=params
        )
    )
    if not conformer_ids:
        params = probe.AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1
        params.pruneRmsThresh = -1.0
        params.useRandomCoords = True
        embed_mode = "etkdgv3_random_coords_fallback"
        conformer_ids = list(
            probe.AllChem.EmbedMultipleConfs(
                mol_h, numConfs=num_conformers, params=params
            )
        )
    if not conformer_ids:
        raise RuntimeError("ETKDGv3 produced zero conformers")
    return mol_h, conformer_ids, embed_mode, time.perf_counter() - started


def conformer_fields() -> list[str]:
    return [
        "stage_b_order",
        "identity_key",
        "peptidoform",
        "sequence",
        "charge",
        "ptm_class",
        "length_bin",
        "microstate_index",
        "charge_sites",
        "carbonyl_fallback_sites",
        "conformer_rank",
        "explicit_atom_count",
        "generation_method",
        "embed_mode",
        "microstate_runtime_seconds",
        "ensemble_pairwise_rmsd_mean",
        "ensemble_pairwise_rmsd_max",
        "descriptor_complete",
        *DESCRIPTOR_NAMES,
    ]


def run_shard(
    manifest: Path,
    expected_manifest_fingerprint: str,
    out_dir: Path,
    shard_index: int,
    shard_count: int,
    expected_identities: int,
    num_conformers: int,
    microstates: int,
    mass_tolerance_da: float,
) -> dict[str, object]:
    if out_dir.exists():
        raise ValueError(f"shard output must be fresh: {out_dir}")

    _, rows, fingerprint = probe.load_manifest(manifest)
    if fingerprint != expected_manifest_fingerprint:
        raise ValueError(
            "frozen Stage B manifest fingerprint mismatch: "
            f"observed={fingerprint} expected={expected_manifest_fingerprint}"
        )
    if len(rows) != expected_identities:
        raise ValueError(
            f"frozen Stage B identity count mismatch: observed={len(rows)} "
            f"expected={expected_identities}"
        )

    selected = shard_rows(rows, shard_index, shard_count)
    out_dir.mkdir(parents=True)

    identity_rows: list[dict[str, object]] = []
    conformer_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    build_successes = 0
    identities_with_conformers = 0
    microstates_succeeded = 0
    conformers_generated = 0
    descriptor_complete = 0
    carbonyl_fallback_identities = 0
    embed_modes: Counter[str] = Counter()
    total_started = time.perf_counter()

    for row in selected:
        identity_started = time.perf_counter()
        identity_key = row["identity_key"]
        charge = probe.parse_int(row["charge"], "charge")
        identity_microstates = 0
        identity_conformers = 0
        identity_embed_modes: Counter[str] = Counter()
        fallback_used = False
        try:
            if not probe.normalize_yes(row["existing_attachment_template_candidate"]):
                raise ValueError("full Stage B refuses non-template-candidate chemistry")
            if row["ptm_class"] not in probe.SUPPORTED_PTM_CLASSES:
                raise ValueError(
                    f"full Stage B does not implement PTM class {row['ptm_class']!r}"
                )

            built = probe.build_neutral_molecule(
                row, mass_tolerance_da=mass_tolerance_da
            )
            build_successes += 1
            candidates = probe.candidate_charge_sites(
                built.mol, built.blocked_basic_atoms
            )
            if any(site.kind == "carbonyl_o_fallback" for site in candidates[:charge]):
                fallback_used = True
                carbonyl_fallback_identities += 1
            acidic = probe.acidic_atom_indices(
                built.mol, built.phosphate_acidic_atoms
            )
            nterm_idx, cterm_idx = probe.terminal_atom_indices(
                built.mol,
                probe.parse_int(row["sequence_length"], "sequence_length"),
            )
            hydrophobic = probe.hydrophobic_atom_indices(built.mol)

            signatures: set[tuple[int, ...]] = set()
            for microstate_index in range(max(1, microstates)):
                sites = probe.microstate_sites(candidates, charge, microstate_index)
                signature = tuple(site.atom_idx for site in sites)
                if signature in signatures:
                    continue
                signatures.add(signature)
                charged = probe.apply_positive_charge(built.mol, sites, charge)
                try:
                    mol_h, conformer_ids, embed_mode, elapsed = embed_etkdg_only(
                        charged,
                        row["peptidoform"],
                        microstate_index,
                        num_conformers,
                    )
                except Exception as exc:
                    failure_rows.append(
                        {
                            "stage_b_order": row["stage_b_order"],
                            "identity_key": identity_key,
                            "stage": "conformer_generation",
                            "microstate_index": microstate_index,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    continue

                identity_microstates += 1
                microstates_succeeded += 1
                identity_embed_modes[embed_mode] += 1
                embed_modes[embed_mode] += 1
                rmsd_mean, rmsd_max = probe.ensemble_rmsd_summary(
                    mol_h, conformer_ids
                )
                charge_site_indices = [site.atom_idx for site in sites]

                for local_rank, conf_id in enumerate(conformer_ids, start=1):
                    descriptors = probe.conformer_descriptors(
                        mol_h,
                        conf_id,
                        charge_site_indices,
                        acidic,
                        nterm_idx,
                        cterm_idx,
                        hydrophobic,
                    )
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
                    conformers_generated += 1
                    identity_conformers += 1
                    conformer_rows.append(
                        {
                            "stage_b_order": row["stage_b_order"],
                            "identity_key": identity_key,
                            "peptidoform": row["peptidoform"],
                            "sequence": row["sequence"],
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
                            "generation_method": "fixed_seed_etkdgv3_no_forcefield",
                            "embed_mode": embed_mode,
                            "microstate_runtime_seconds": f"{elapsed:.6f}",
                            "ensemble_pairwise_rmsd_mean": fmt(rmsd_mean),
                            "ensemble_pairwise_rmsd_max": fmt(rmsd_max),
                            "descriptor_complete": "YES" if complete else "NO",
                            **{
                                name: fmt(descriptors[name])
                                for name in DESCRIPTOR_NAMES
                            },
                        }
                    )

            if identity_conformers > 0:
                identities_with_conformers += 1

            identity_rows.append(
                {
                    "stage_b_order": row["stage_b_order"],
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
                    "conformers_generated": identity_conformers,
                    "embed_modes": ";".join(
                        f"{name}:{count}"
                        for name, count in sorted(identity_embed_modes.items())
                    ),
                    "runtime_seconds": f"{time.perf_counter() - identity_started:.6f}",
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "stage_b_order": row["stage_b_order"],
                    "identity_key": identity_key,
                    "stage": "neutral_molecule_build",
                    "microstate_index": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            identity_rows.append(
                {
                    "stage_b_order": row["stage_b_order"],
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

    elapsed = time.perf_counter() - total_started
    identity_fields = [
        "stage_b_order",
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
        "embed_modes",
        "runtime_seconds",
    ]
    failure_fields = [
        "stage_b_order",
        "identity_key",
        "stage",
        "microstate_index",
        "error_type",
        "error",
    ]
    write_tsv(out_dir / "shard_identity_summary.tsv", identity_fields, identity_rows)
    write_tsv(out_dir / "shard_conformers.tsv", conformer_fields(), conformer_rows)
    write_tsv(out_dir / "shard_failures.tsv", failure_fields, failure_rows)

    expected_shard_orders = sorted(
        probe.parse_int(row["stage_b_order"], "stage_b_order") for row in selected
    )
    shard_hasher = hashlib.sha256()
    for row in sorted(selected, key=lambda item: int(item["stage_b_order"])):
        shard_hasher.update(row["stage_b_order"].encode("ascii"))
        shard_hasher.update(b"\t")
        shard_hasher.update(row["identity_key"].encode("utf-8"))
        shard_hasher.update(b"\n")

    summary = [
        {"metric": "audit_version", "value": AUDIT_VERSION},
        {"metric": "partition_scope", "value": "TRAIN_frozen_stage_b_manifest_only"},
        {"metric": "manifest_fingerprint", "value": fingerprint},
        {"metric": "generation_method", "value": "fixed_seed_etkdgv3_no_forcefield"},
        {"metric": "forcefield_minimization", "value": "NO"},
        {"metric": "shard_index", "value": shard_index},
        {"metric": "shard_count", "value": shard_count},
        {"metric": "expected_total_identities", "value": expected_identities},
        {"metric": "shard_identities", "value": len(selected)},
        {
            "metric": "shard_order_min",
            "value": min(expected_shard_orders) if expected_shard_orders else "",
        },
        {
            "metric": "shard_order_max",
            "value": max(expected_shard_orders) if expected_shard_orders else "",
        },
        {"metric": "shard_identity_fingerprint", "value": "sha256:" + shard_hasher.hexdigest()},
        {"metric": "neutral_build_successes", "value": build_successes},
        {"metric": "identities_with_conformers", "value": identities_with_conformers},
        {"metric": "microstates_succeeded", "value": microstates_succeeded},
        {"metric": "conformers_generated", "value": conformers_generated},
        {"metric": "descriptor_complete_conformers", "value": descriptor_complete},
        {
            "metric": "carbonyl_fallback_identities",
            "value": carbonyl_fallback_identities,
        },
        {
            "metric": "embed_mode_counts",
            "value": ";".join(f"{k}:{v}" for k, v in sorted(embed_modes.items())),
        },
        {"metric": "runtime_seconds", "value": f"{elapsed:.6f}"},
        {"metric": "measured_ccs_used", "value": "NO"},
        {"metric": "mobility_labels_used", "value": "NO"},
        {"metric": "dev_labels_used", "value": "NO"},
        {"metric": "holdout_used", "value": "NO"},
    ]
    write_tsv(out_dir / "shard_summary.tsv", ["metric", "value"], summary)

    print(f"audit_version\t{AUDIT_VERSION}")
    print(f"manifest_fingerprint\t{fingerprint}")
    print(f"shard_index\t{shard_index}")
    print(f"shard_count\t{shard_count}")
    print(f"shard_identities\t{len(selected)}")
    print(f"neutral_build_successes\t{build_successes}")
    print(f"identities_with_conformers\t{identities_with_conformers}")
    print(f"conformers_generated\t{conformers_generated}")
    print(f"runtime_seconds\t{elapsed:.6f}")
    print("forcefield_minimization\tNO")
    print("measured_ccs_used\tNO")
    print("dev_labels_used\tNO")
    print("holdout_used\tNO")
    return {row["metric"]: row["value"] for row in summary}


def finite_values(rows: Sequence[Mapping[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            values.append(parsed)
    return values


def aggregate_features(
    manifest_rows: Sequence[dict[str, str]],
    identity_rows: Sequence[dict[str, str]],
    conformer_rows: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    identity_by_key = {row["identity_key"]: row for row in identity_rows}
    conf_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in conformer_rows:
        conf_by_key[row["identity_key"]].append(row)

    features: list[dict[str, object]] = []
    for manifest_row in manifest_rows:
        key = manifest_row["identity_key"]
        identity = identity_by_key[key]
        confs = conf_by_key.get(key, [])
        microstate_modes: dict[str, str] = {}
        for conf in confs:
            microstate_modes.setdefault(conf["microstate_index"], conf["embed_mode"])
        fallback_microstates = sum(
            mode == "etkdgv3_random_coords_fallback"
            for mode in microstate_modes.values()
        )
        feature: dict[str, object] = {
            "stage_b_order": manifest_row["stage_b_order"],
            "identity_key": key,
            "peptidoform": manifest_row["peptidoform"],
            "sequence": manifest_row["sequence"],
            "charge": manifest_row["charge"],
            "ptm_class": manifest_row["ptm_class"],
            "length_bin": manifest_row["length_bin"],
            "neutral_build_success": identity.get("neutral_build_success", "NO"),
            "carbonyl_fallback_used": identity.get("carbonyl_fallback_used", ""),
            "microstates_succeeded": identity.get("microstates_succeeded", "0"),
            "conformers_generated": identity.get("conformers_generated", "0"),
            "embed_random_coords_fallback_fraction": (
                f"{fallback_microstates / len(microstate_modes):.8f}"
                if microstate_modes
                else ""
            ),
            "identity_runtime_seconds": identity.get("runtime_seconds", ""),
        }
        for descriptor in DESCRIPTOR_NAMES:
            values = finite_values(confs, descriptor)
            if values:
                feature[f"{descriptor}_median"] = fmt(statistics.median(values))
                feature[f"{descriptor}_mean"] = fmt(statistics.fmean(values))
                feature[f"{descriptor}_std"] = fmt(
                    statistics.pstdev(values) if len(values) > 1 else 0.0
                )
                feature[f"{descriptor}_min"] = fmt(min(values))
                feature[f"{descriptor}_max"] = fmt(max(values))
            else:
                for suffix in ("median", "mean", "std", "min", "max"):
                    feature[f"{descriptor}_{suffix}"] = ""
        rmsd_values = finite_values(confs, "ensemble_pairwise_rmsd_max")
        feature["ensemble_pairwise_rmsd_max_median"] = (
            fmt(statistics.median(rmsd_values)) if rmsd_values else ""
        )
        features.append(feature)
    return features


def feature_fingerprint(fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update("\t".join(str(row.get(field, "")) for field in fields).encode("utf-8"))
        hasher.update(b"\n")
    return "sha256:" + hasher.hexdigest()


def categorical_summary(
    manifest_rows: Sequence[dict[str, str]],
    identity_rows: Sequence[dict[str, str]],
    field: str,
) -> list[dict[str, object]]:
    by_identity = {row["identity_key"]: row for row in identity_rows}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in manifest_rows:
        group = row[field]
        identity = by_identity[row["identity_key"]]
        counts[group]["identities"] += 1
        counts[group]["neutral_build_success"] += int(
            identity.get("neutral_build_success") == "YES"
        )
        counts[group]["conformer_success"] += int(
            int(identity.get("conformers_generated") or 0) > 0
        )
    output = []
    for group in sorted(counts, key=lambda value: (len(value), value)):
        item = counts[group]
        total = item["identities"]
        output.append(
            {
                field: group,
                "identities": total,
                "neutral_build_success": item["neutral_build_success"],
                "conformer_success": item["conformer_success"],
                "conformer_success_fraction": f"{item['conformer_success'] / total:.8f}",
            }
        )
    return output


def charge_pair_variation(
    feature_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in feature_rows:
        if row.get("radius_of_gyration_median", "") != "":
            groups[str(row["peptidoform"])].append(row)
    output = []
    for peptidoform, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda item: int(str(item["charge"])))
        for left, right in zip(ordered, ordered[1:]):
            delta_rg = float(str(right["radius_of_gyration_median"])) - float(
                str(left["radius_of_gyration_median"])
            )
            delta_volume = float(str(right["molecular_volume_median"])) - float(
                str(left["molecular_volume_median"])
            )
            delta_asphericity = float(str(right["asphericity_median"])) - float(
                str(left["asphericity_median"])
            )
            delta_sasa = float(str(right["charge_site_sasa_mean_median"])) - float(
                str(left["charge_site_sasa_mean_median"])
            )
            nontrivial = (
                abs(delta_rg) >= 0.10
                or abs(delta_asphericity) >= 0.01
                or abs(delta_sasa) >= 1.0
            )
            output.append(
                {
                    "peptidoform": peptidoform,
                    "identity_key_a": left["identity_key"],
                    "charge_a": left["charge"],
                    "identity_key_b": right["identity_key"],
                    "charge_b": right["charge"],
                    "delta_radius_of_gyration": fmt(delta_rg),
                    "delta_molecular_volume": fmt(delta_volume),
                    "delta_asphericity": fmt(delta_asphericity),
                    "delta_charge_site_sasa_mean": fmt(delta_sasa),
                    "nontrivial_charge_conditioned_shape_change": "YES"
                    if nontrivial
                    else "NO",
                }
            )
    return output


def run_finalize(
    manifest: Path,
    expected_manifest_fingerprint: str,
    shards_root: Path,
    out_dir: Path,
    shard_count: int,
    expected_identities: int,
) -> dict[str, object]:
    if out_dir.exists():
        raise ValueError(f"final output must be fresh: {out_dir}")
    _, manifest_rows, fingerprint = probe.load_manifest(manifest)
    if fingerprint != expected_manifest_fingerprint:
        raise ValueError(
            "frozen Stage B manifest fingerprint mismatch: "
            f"observed={fingerprint} expected={expected_manifest_fingerprint}"
        )
    if len(manifest_rows) != expected_identities:
        raise ValueError(
            f"manifest identity count mismatch: observed={len(manifest_rows)} expected={expected_identities}"
        )

    identity_rows: list[dict[str, str]] = []
    conformer_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []
    total_shard_runtime = 0.0
    for shard_index in range(shard_count):
        shard_dir = shards_root / f"shard_{shard_index:03d}_of_{shard_count:03d}"
        for filename in (
            "shard_summary.tsv",
            "shard_identity_summary.tsv",
            "shard_conformers.tsv",
            "shard_failures.tsv",
        ):
            if not (shard_dir / filename).is_file():
                raise ValueError(f"missing shard output: {shard_dir / filename}")
        if summary_value(shard_dir / "shard_summary.tsv", "manifest_fingerprint") != fingerprint:
            raise ValueError(f"shard {shard_index} manifest fingerprint mismatch")
        if int(summary_value(shard_dir / "shard_summary.tsv", "shard_index")) != shard_index:
            raise ValueError(f"shard {shard_index} index mismatch")
        if int(summary_value(shard_dir / "shard_summary.tsv", "shard_count")) != shard_count:
            raise ValueError(f"shard {shard_index} count mismatch")
        total_shard_runtime += float(
            summary_value(shard_dir / "shard_summary.tsv", "runtime_seconds")
        )
        _, shard_identity = read_tsv(shard_dir / "shard_identity_summary.tsv")
        _, shard_conformers = read_tsv(shard_dir / "shard_conformers.tsv")
        _, shard_failures = read_tsv(shard_dir / "shard_failures.tsv")
        identity_rows.extend(shard_identity)
        conformer_rows.extend(shard_conformers)
        failure_rows.extend(shard_failures)

    manifest_keys = [row["identity_key"] for row in manifest_rows]
    observed_keys = [row["identity_key"] for row in identity_rows]
    counts = Counter(observed_keys)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(set(manifest_keys) - set(observed_keys))
    extras = sorted(set(observed_keys) - set(manifest_keys))
    if duplicates or missing or extras or len(identity_rows) != expected_identities:
        raise ValueError(
            "full Stage B identity coverage failure: "
            f"rows={len(identity_rows)} duplicates={len(duplicates)} "
            f"missing={len(missing)} extras={len(extras)}"
        )

    identity_by_key = {row["identity_key"]: row for row in identity_rows}
    ordered_identity_rows = [identity_by_key[key] for key in manifest_keys]
    features = aggregate_features(manifest_rows, ordered_identity_rows, conformer_rows)
    feature_fields = list(features[0].keys())
    structure_fingerprint = feature_fingerprint(feature_fields, features)

    build_success = sum(row["neutral_build_success"] == "YES" for row in ordered_identity_rows)
    conformer_success = sum(int(row.get("conformers_generated") or 0) > 0 for row in ordered_identity_rows)
    descriptor_complete = sum(row.get("descriptor_complete") == "YES" for row in conformer_rows)
    conformers_generated = len(conformer_rows)
    build_fraction = build_success / expected_identities
    conformer_fraction = conformer_success / expected_identities
    descriptor_fraction = descriptor_complete / conformers_generated if conformers_generated else 0.0
    mean_runtime = total_shard_runtime / expected_identities

    microstate_modes: dict[tuple[str, str], str] = {}
    for row in conformer_rows:
        microstate_modes.setdefault(
            (row["identity_key"], row["microstate_index"]), row["embed_mode"]
        )
    fallback_microstates = sum(
        mode == "etkdgv3_random_coords_fallback" for mode in microstate_modes.values()
    )
    fallback_fraction = fallback_microstates / len(microstate_modes) if microstate_modes else 0.0
    carbonyl_fallback = sum(
        row.get("carbonyl_fallback_used") == "YES" for row in ordered_identity_rows
    )

    pair_rows = charge_pair_variation(features)
    pair_nontrivial = sum(
        row["nontrivial_charge_conditioned_shape_change"] == "YES"
        for row in pair_rows
    )

    gate = (
        build_fraction >= 0.95
        and conformer_fraction >= 0.90
        and descriptor_fraction >= 0.95
        and mean_runtime <= 120.0
    )

    out_dir.mkdir(parents=True)
    write_tsv(out_dir / "stage_b_structure_features.tsv", feature_fields, features)
    write_tsv(
        out_dir / "stage_b_identity_summary.tsv",
        list(ordered_identity_rows[0].keys()),
        ordered_identity_rows,
    )
    write_tsv(
        out_dir / "stage_b_conformers.tsv",
        conformer_fields(),
        sorted(
            conformer_rows,
            key=lambda row: (
                int(row["stage_b_order"]),
                int(row["microstate_index"]),
                int(row["conformer_rank"]),
            ),
        ),
    )
    failure_fields = [
        "stage_b_order",
        "identity_key",
        "stage",
        "microstate_index",
        "error_type",
        "error",
    ]
    write_tsv(
        out_dir / "stage_b_failures.tsv",
        failure_fields,
        sorted(failure_rows, key=lambda row: (int(row["stage_b_order"]), row["stage"])),
    )
    pair_fields = [
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
    ]
    write_tsv(out_dir / "stage_b_charge_pair_variation.tsv", pair_fields, pair_rows)

    for field, filename in (
        ("charge", "stage_b_by_charge.tsv"),
        ("length_bin", "stage_b_by_length_bin.tsv"),
        ("ptm_class", "stage_b_by_ptm_class.tsv"),
    ):
        rows = categorical_summary(manifest_rows, ordered_identity_rows, field)
        write_tsv(
            out_dir / filename,
            [field, "identities", "neutral_build_success", "conformer_success", "conformer_success_fraction"],
            rows,
        )

    summary_rows = [
        {"metric": "audit_version", "value": AUDIT_VERSION},
        {"metric": "partition_scope", "value": "TRAIN_frozen_stage_b_manifest_only"},
        {"metric": "manifest_fingerprint", "value": fingerprint},
        {"metric": "structure_feature_fingerprint", "value": structure_fingerprint},
        {"metric": "rdkit_version", "value": probe.rdBase.rdkitVersion},
        {"metric": "generation_method", "value": "fixed_seed_etkdgv3_no_forcefield"},
        {"metric": "forcefield_minimization", "value": "NO"},
        {"metric": "shard_count", "value": shard_count},
        {"metric": "expected_identities", "value": expected_identities},
        {"metric": "identities_verified_exactly_once", "value": expected_identities},
        {"metric": "neutral_build_successes", "value": build_success},
        {"metric": "neutral_build_success_fraction", "value": f"{build_fraction:.8f}"},
        {"metric": "identities_with_conformers", "value": conformer_success},
        {"metric": "identity_conformer_success_fraction", "value": f"{conformer_fraction:.8f}"},
        {"metric": "microstates_succeeded", "value": len(microstate_modes)},
        {"metric": "conformers_generated", "value": conformers_generated},
        {"metric": "descriptor_complete_conformers", "value": descriptor_complete},
        {"metric": "descriptor_complete_fraction", "value": f"{descriptor_fraction:.8f}"},
        {"metric": "random_coords_fallback_microstates", "value": fallback_microstates},
        {"metric": "random_coords_fallback_fraction", "value": f"{fallback_fraction:.8f}"},
        {"metric": "carbonyl_charge_fallback_identities", "value": carbonyl_fallback},
        {"metric": "matched_adjacent_charge_pairs", "value": len(pair_rows)},
        {"metric": "matched_charge_pairs_nontrivial", "value": pair_nontrivial},
        {
            "metric": "matched_charge_nontrivial_fraction",
            "value": f"{pair_nontrivial / len(pair_rows):.8f}" if pair_rows else "",
        },
        {"metric": "aggregate_shard_cpu_seconds", "value": f"{total_shard_runtime:.6f}"},
        {"metric": "aggregate_shard_cpu_hours", "value": f"{total_shard_runtime / 3600.0:.6f}"},
        {"metric": "mean_runtime_seconds_per_identity", "value": f"{mean_runtime:.6f}"},
        {"metric": "gate_neutral_build_ge_95pct", "value": "PASS" if build_fraction >= 0.95 else "FAIL"},
        {"metric": "gate_identity_conformer_success_ge_90pct", "value": "PASS" if conformer_fraction >= 0.90 else "FAIL"},
        {"metric": "gate_descriptor_complete_ge_95pct", "value": "PASS" if descriptor_fraction >= 0.95 else "FAIL"},
        {"metric": "gate_mean_runtime_le_120s_per_identity", "value": "PASS" if mean_runtime <= 120.0 else "FAIL"},
        {"metric": "full_stage_b_operational_gate", "value": "PASS" if gate else "FAIL"},
        {"metric": "measured_ccs_used", "value": "NO"},
        {"metric": "mobility_labels_used", "value": "NO"},
        {"metric": "dev_labels_used", "value": "NO"},
        {"metric": "holdout_used", "value": "NO"},
    ]
    write_tsv(out_dir / "stage_b_summary.tsv", ["metric", "value"], summary_rows)

    with (out_dir / "stage_b_report.md").open("w", encoding="utf-8") as handle:
        handle.write("# ReDeeM structure-signal feasibility v1 — full frozen Stage B ETKDG descriptors\n\n")
        handle.write(f"- Manifest fingerprint: `{fingerprint}`\n")
        handle.write(f"- Structure-feature fingerprint: `{structure_fingerprint}`\n")
        handle.write(f"- RDKit: `{probe.rdBase.rdkitVersion}`\n")
        handle.write(f"- Frozen identities verified exactly once: `{expected_identities}`\n")
        handle.write(f"- Neutral build success: `{build_success}/{expected_identities}` ({build_fraction:.2%})\n")
        handle.write(f"- Identity conformer success: `{conformer_success}/{expected_identities}` ({conformer_fraction:.2%})\n")
        handle.write(f"- Descriptor-complete conformers: `{descriptor_complete}/{conformers_generated}` ({descriptor_fraction:.2%})\n")
        handle.write(f"- Random-coordinate fallback microstates: `{fallback_microstates}/{len(microstate_modes)}` ({fallback_fraction:.2%})\n")
        handle.write(f"- Carbonyl-charge fallback identities: `{carbonyl_fallback}`\n")
        handle.write(f"- Aggregate shard CPU hours: `{total_shard_runtime / 3600.0:.3f}`\n")
        handle.write(f"- Mean runtime / identity: `{mean_runtime:.3f} s`\n")
        handle.write(f"- Full Stage B operational gate: **{'PASS' if gate else 'FAIL'}**\n\n")
        handle.write("## Frozen method\n\n")
        handle.write("Structures are generated with fixed-seed RDKit ETKDGv3, with the validated random-coordinate ETKDG fallback when ordinary embedding yields no conformer. No MMFF/UFF force-field minimization is performed. Charge-site microstates remain deterministic feasibility heuristics rather than inferred experimental protomers.\n\n")
        handle.write("No measured CCS or mobility label is read or used.\n")

    print(f"audit_version\t{AUDIT_VERSION}")
    print(f"manifest_fingerprint\t{fingerprint}")
    print(f"structure_feature_fingerprint\t{structure_fingerprint}")
    print(f"identities_verified_exactly_once\t{expected_identities}")
    print(f"neutral_build_success_fraction\t{build_fraction:.8f}")
    print(f"identity_conformer_success_fraction\t{conformer_fraction:.8f}")
    print(f"descriptor_complete_fraction\t{descriptor_fraction:.8f}")
    print(f"mean_runtime_seconds_per_identity\t{mean_runtime:.6f}")
    print(f"full_stage_b_operational_gate\t{'PASS' if gate else 'FAIL'}")
    print("measured_ccs_used\tNO")
    print("dev_labels_used\tNO")
    print("holdout_used\tNO")
    return {row["metric"]: row["value"] for row in summary_rows}


def tiny_manifest(path: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "synthetic.tsv"
        probe.synthetic_manifest(source)
        fields, rows = read_tsv(source)
        chosen = [row for row in rows if row["charge"] == "2"]
        for order, row in enumerate(chosen, start=1):
            row["stage_b_order"] = str(order)
        write_tsv(path, fields, chosen)
        return probe.manifest_fingerprint(chosen)


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.tsv"
        fingerprint = tiny_manifest(manifest)
        _, rows, observed = probe.load_manifest(manifest)
        if observed != fingerprint or len(rows) != 7:
            raise RuntimeError("full Stage B self-test manifest mismatch")

        assigned = []
        for shard_index in range(2):
            subset = shard_rows(rows, shard_index, 2)
            assigned.extend(row["identity_key"] for row in subset)
        if sorted(assigned) != sorted(row["identity_key"] for row in rows):
            raise RuntimeError("full Stage B shard assignment is not complete/disjoint")

        shards_root = root / "shards"
        shards_root.mkdir()
        for shard_index in range(2):
            run_shard(
                manifest=manifest,
                expected_manifest_fingerprint=fingerprint,
                out_dir=shards_root / f"shard_{shard_index:03d}_of_002",
                shard_index=shard_index,
                shard_count=2,
                expected_identities=7,
                num_conformers=1,
                microstates=1,
                mass_tolerance_da=0.002,
            )
        final = root / "final"
        summary = run_finalize(
            manifest=manifest,
            expected_manifest_fingerprint=fingerprint,
            shards_root=shards_root,
            out_dir=final,
            shard_count=2,
            expected_identities=7,
        )
        if summary["identities_verified_exactly_once"] != 7:
            raise RuntimeError("full Stage B self-test finalize count mismatch")
        if not (final / "stage_b_structure_features.tsv").is_file():
            raise RuntimeError("full Stage B self-test did not write feature table")
    print("structure_signal_stage_b_full_v1_self_test=PASS")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.command == "shard":
        run_shard(
            manifest=args.manifest,
            expected_manifest_fingerprint=args.expected_manifest_fingerprint,
            out_dir=args.out_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            expected_identities=args.expected_identities,
            num_conformers=args.num_conformers,
            microstates=args.microstates,
            mass_tolerance_da=args.mass_tolerance_da,
        )
        return 0
    if args.command == "finalize":
        run_finalize(
            manifest=args.manifest,
            expected_manifest_fingerprint=args.expected_manifest_fingerprint,
            shards_root=args.shards_root,
            out_dir=args.out_dir,
            shard_count=args.shard_count,
            expected_identities=args.expected_identities,
        )
        return 0
    raise SystemExit("use --self-test or one of: shard, finalize")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
