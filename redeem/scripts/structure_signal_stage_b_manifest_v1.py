#!/usr/bin/env python3
"""Build the deterministic TRAIN-only Stage B conformer pilot manifest.

This script consumes ``identity_inventory.tsv`` produced by
``foundation_audit_structure_signal_feasibility_v1`` Stage A. It never reads
measured CCS or mobility labels. Inputs that expose label-like columns are
rejected so Stage B sample selection cannot accidentally depend on them.

Selection is deterministic and input-order independent:

* eligible universe: positive-charge identities with
  ``existing_attachment_template_candidate == YES``;
* stratification: precursor charge x Stage-A length bin x PTM class;
* every non-empty stratum receives a small minimum quota where capacity allows;
* remaining quota is distributed proportional to sqrt(stratum population),
  which preserves broad population weighting while deliberately protecting
  rare but scientifically important strata;
* identities within each stratum are ranked by SHA-256 over a fixed versioned
  salt and ``identity_key``. No RNG state is used.

The default target is 8,000 identities, within the predeclared 5k-10k Stage B
pilot range. The output is a manifest only. It does not build molecules,
generate conformers, calculate structure descriptors, or train a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

AUDIT_VERSION = "structure_signal_feasibility_v1_stage_b_manifest_v1"
DEFAULT_TARGET_SIZE = 8_000
DEFAULT_MINIMUM_PER_STRATUM = 4
SELECTION_SALT = "redeem_structure_signal_stage_b_manifest_v1_sha256"

REQUIRED_COLUMNS = {
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

COMMON_PTM_FAMILIES = {
    "UniMod:4": "carbamidomethyl",
    "UniMod:35": "oxidation",
    "UniMod:7": "deamidation",
    "UniMod:21": "phospho",
    "UniMod:1": "acetyl",
}


@dataclass(frozen=True, order=True)
class Stratum:
    charge: int
    length_bin: str
    ptm_class: str

    def key(self) -> str:
        return f"z{self.charge}|{self.length_bin}|{self.ptm_class}"


@dataclass
class FirstPass:
    fieldnames: list[str]
    total_rows: int
    eligible_rows: int
    stratum_counts: dict[Stratum, int]
    exclusion_counts: Counter[str]
    chemistry_status_counts: Counter[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-inventory", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument(
        "--minimum-per-stratum",
        type=int,
        default=DEFAULT_MINIMUM_PER_STRATUM,
        help="Minimum selected identities per non-empty stratum before weighted fill.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def normalize_yes(value: str) -> bool:
    return value.strip().upper() == "YES"


def parse_positive_int(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not an integer: {value!r}") from exc
    return parsed


def modification_ptm_labels(modification_labels: str) -> list[str]:
    labels: list[str] = []
    for token in modification_labels.split(";"):
        token = token.strip()
        if not token:
            continue
        labels.append(token.split("@", 1)[0].strip())
    return labels


def classify_ptm(row: Mapping[str, str]) -> str:
    if not normalize_yes(row.get("modified", "")):
        return "unmodified"

    labels = modification_ptm_labels(row.get("modification_labels", ""))
    if not labels:
        # Stage A should not emit modified=YES without labels. Keep this explicit
        # rather than silently treating it as unmodified.
        return "other_supported"

    families = [COMMON_PTM_FAMILIES.get(label) for label in labels]
    known_families = sorted({family for family in families if family is not None})
    unknown_count = sum(family is None for family in families)

    if unknown_count == 0 and len(known_families) == 1:
        return known_families[0]
    if unknown_count == 0 and len(known_families) >= 2:
        return "common_ptm_combination"
    if known_families:
        return "supported_mixed_other"
    return "other_supported"


def validate_header(fieldnames: Iterable[str] | None) -> list[str]:
    if fieldnames is None:
        raise ValueError("identity inventory has no TSV header")
    fields = list(fieldnames)
    missing = sorted(REQUIRED_COLUMNS - set(fields))
    if missing:
        raise ValueError(
            "identity inventory is missing required columns: " + ", ".join(missing)
        )
    forbidden = sorted(FORBIDDEN_LABEL_COLUMNS & {name.strip().lower() for name in fields})
    if forbidden:
        raise ValueError(
            "refusing identity inventory containing CCS/mobility label columns: "
            + ", ".join(forbidden)
        )
    return fields


def exclusion_reason(row: Mapping[str, str]) -> str | None:
    charge = parse_positive_int(row["charge"], "charge")
    if charge <= 0:
        return "nonpositive_charge"
    if not normalize_yes(row["existing_attachment_template_candidate"]):
        status = row.get("chemistry_status", "").strip() or "unknown"
        return f"chemistry_not_template_candidate:{status}"
    return None


def row_stratum(row: Mapping[str, str]) -> Stratum:
    charge = parse_positive_int(row["charge"], "charge")
    return Stratum(
        charge=charge,
        length_bin=row["length_bin"].strip(),
        ptm_class=classify_ptm(row),
    )


def first_pass(path: Path) -> FirstPass:
    total_rows = 0
    eligible_rows = 0
    stratum_counts: Counter[Stratum] = Counter()
    exclusion_counts: Counter[str] = Counter()
    chemistry_status_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = validate_header(reader.fieldnames)
        for row in reader:
            total_rows += 1
            chemistry_status_counts[row.get("chemistry_status", "").strip() or "unknown"] += 1
            reason = exclusion_reason(row)
            if reason is not None:
                exclusion_counts[reason] += 1
                continue
            eligible_rows += 1
            stratum_counts[row_stratum(row)] += 1

    return FirstPass(
        fieldnames=fieldnames,
        total_rows=total_rows,
        eligible_rows=eligible_rows,
        stratum_counts=dict(stratum_counts),
        exclusion_counts=exclusion_counts,
        chemistry_status_counts=chemistry_status_counts,
    )


def allocate_quotas(
    counts: Mapping[Stratum, int], target_size: int, minimum_per_stratum: int
) -> dict[Stratum, int]:
    if target_size <= 0:
        raise ValueError("target size must be positive")
    if minimum_per_stratum < 0:
        raise ValueError("minimum per stratum must be non-negative")
    total = sum(counts.values())
    if total == 0:
        return {stratum: 0 for stratum in counts}
    target = min(target_size, total)

    quotas = {
        stratum: min(count, minimum_per_stratum)
        for stratum, count in sorted(counts.items())
    }
    base_total = sum(quotas.values())

    # If the requested target is too small to satisfy every configured minimum,
    # allocate in stable rounds. Every non-empty stratum gets one seat before a
    # second seat is assigned when the target permits it.
    if base_total > target:
        quotas = {stratum: 0 for stratum in counts}
        ordered = sorted(
            counts,
            key=lambda stratum: (
                -math.sqrt(counts[stratum]),
                stratum.charge,
                stratum.length_bin,
                stratum.ptm_class,
            ),
        )
        remaining_small = target
        while remaining_small > 0:
            progressed = False
            for stratum in ordered:
                if remaining_small == 0:
                    break
                cap = min(counts[stratum], minimum_per_stratum)
                if quotas[stratum] >= cap:
                    continue
                quotas[stratum] += 1
                remaining_small -= 1
                progressed = True
            if not progressed:
                break
        if sum(quotas.values()) != target:
            raise RuntimeError(
                f"small-target quota allocation mismatch: selected={sum(quotas.values())} target={target}"
            )
        return quotas

    remaining = target - base_total
    capacities = {
        stratum: counts[stratum] - quotas[stratum]
        for stratum in counts
    }

    while remaining > 0:
        active = [stratum for stratum in counts if capacities[stratum] > 0]
        if not active:
            break
        weight_total = sum(math.sqrt(counts[stratum]) for stratum in active)
        shares = {
            stratum: remaining * math.sqrt(counts[stratum]) / weight_total
            for stratum in active
        }

        added = 0
        for stratum in sorted(active):
            increment = min(capacities[stratum], int(math.floor(shares[stratum])))
            if increment <= 0:
                continue
            quotas[stratum] += increment
            capacities[stratum] -= increment
            remaining -= increment
            added += increment
            if remaining == 0:
                break

        if remaining == 0:
            break
        if added > 0:
            continue

        # All proportional shares are <1. Allocate the remaining seats by
        # largest fractional share, with the Stratum ordering as a stable tie
        # breaker.
        ranked = sorted(
            active,
            key=lambda stratum: (-shares[stratum], stratum),
        )
        for stratum in ranked:
            if remaining == 0:
                break
            if capacities[stratum] <= 0:
                continue
            quotas[stratum] += 1
            capacities[stratum] -= 1
            remaining -= 1

    if sum(quotas.values()) != target:
        raise RuntimeError(
            f"quota allocation mismatch: selected={sum(quotas.values())} target={target}"
        )
    return quotas


def selection_hash(identity_key: str) -> tuple[int, str]:
    digest = hashlib.sha256(
        f"{SELECTION_SALT}\0{identity_key}".encode("utf-8")
    ).hexdigest()
    return int(digest, 16), digest


def second_pass_select(
    path: Path,
    quotas: Mapping[Stratum, int],
) -> dict[Stratum, list[tuple[int, str, dict[str, str]]]]:
    # A sorted bounded list is used rather than retaining all 1.1M identities.
    # Typical per-stratum quotas are small, and the total retained state is only
    # the requested pilot size.
    selected: dict[Stratum, list[tuple[int, str, dict[str, str]]]] = defaultdict(list)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        validate_header(reader.fieldnames)
        for row in reader:
            if exclusion_reason(row) is not None:
                continue
            stratum = row_stratum(row)
            quota = quotas.get(stratum, 0)
            if quota <= 0:
                continue
            identity_key = row["identity_key"]
            hash_int, digest = selection_hash(identity_key)
            bucket = selected[stratum]
            candidate = (hash_int, digest, dict(row))
            if len(bucket) < quota:
                bucket.append(candidate)
                if len(bucket) == quota:
                    bucket.sort(key=lambda item: (item[0], item[2]["identity_key"]))
                continue
            # The list is sorted ascending once full; replace its largest item
            # only when the new deterministic rank is smaller.
            worst = bucket[-1]
            candidate_key = (hash_int, identity_key)
            worst_key = (worst[0], worst[2]["identity_key"])
            if candidate_key < worst_key:
                bucket[-1] = candidate
                bucket.sort(key=lambda item: (item[0], item[2]["identity_key"]))

    for stratum, quota in quotas.items():
        observed = len(selected.get(stratum, []))
        if observed != quota:
            raise RuntimeError(
                f"stratum selection mismatch for {stratum.key()}: observed={observed} quota={quota}"
            )
    return dict(selected)


def manifest_fingerprint(rows: Iterable[tuple[Stratum, int, str, Mapping[str, str]]]) -> str:
    hasher = hashlib.sha256()
    for stratum, _, digest, row in rows:
        hasher.update(stratum.key().encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(digest.encode("ascii"))
        hasher.update(b"\t")
        hasher.update(row["identity_key"].encode("utf-8"))
        hasher.update(b"\n")
    return "sha256:" + hasher.hexdigest()


def flatten_selection(
    selected: Mapping[Stratum, list[tuple[int, str, dict[str, str]]]],
) -> list[tuple[Stratum, int, str, dict[str, str]]]:
    rows: list[tuple[Stratum, int, str, dict[str, str]]] = []
    for stratum in sorted(selected):
        bucket = sorted(
            selected[stratum], key=lambda item: (item[0], item[2]["identity_key"])
        )
        for rank, (_, digest, row) in enumerate(bucket, start=1):
            rows.append((stratum, rank, digest, row))
    return rows


def write_tsv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dimension_rows(
    counts: Mapping[Stratum, int], quotas: Mapping[Stratum, int], attribute: str
) -> list[dict[str, object]]:
    population: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    for stratum, count in counts.items():
        key = str(getattr(stratum, attribute))
        population[key] += count
        selected[key] += quotas.get(stratum, 0)
    def sort_key(item: str) -> tuple[int, object]:
        if attribute == "charge":
            return (0, int(item))
        return (1, item)
    rows = []
    for key in sorted(population, key=sort_key):
        rows.append(
            {
                attribute: key,
                "eligible_population": population[key],
                "selected": selected[key],
                "selection_fraction": f"{selected[key] / population[key]:.8f}",
            }
        )
    return rows


def run_manifest(
    identity_inventory: Path,
    out_dir: Path,
    target_size: int,
    minimum_per_stratum: int,
) -> dict[str, str]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    first = first_pass(identity_inventory)
    if first.eligible_rows == 0:
        raise ValueError("no Stage B eligible identities were found")

    quotas = allocate_quotas(
        first.stratum_counts,
        target_size=target_size,
        minimum_per_stratum=minimum_per_stratum,
    )
    selected = second_pass_select(identity_inventory, quotas)
    flat = flatten_selection(selected)
    selected_count = len(flat)
    expected_count = min(target_size, first.eligible_rows)
    if selected_count != expected_count:
        raise RuntimeError(
            f"selected identity count mismatch: observed={selected_count} expected={expected_count}"
        )

    fingerprint = manifest_fingerprint(flat)

    manifest_fields = [
        "stage_b_order",
        "stratum_key",
        "stratum_population",
        "stratum_quota",
        "stratum_rank",
        "ptm_class",
        "selection_hash_sha256",
    ] + first.fieldnames
    manifest_rows: list[dict[str, object]] = []
    for global_rank, (stratum, stratum_rank, digest, row) in enumerate(flat, start=1):
        manifest_rows.append(
            {
                "stage_b_order": global_rank,
                "stratum_key": stratum.key(),
                "stratum_population": first.stratum_counts[stratum],
                "stratum_quota": quotas[stratum],
                "stratum_rank": stratum_rank,
                "ptm_class": stratum.ptm_class,
                "selection_hash_sha256": digest,
                **row,
            }
        )
    write_tsv(out_dir / "stage_b_manifest.tsv", manifest_fields, manifest_rows)

    stratum_rows = []
    for stratum in sorted(first.stratum_counts):
        population = first.stratum_counts[stratum]
        quota = quotas[stratum]
        stratum_rows.append(
            {
                "stratum_key": stratum.key(),
                "charge": stratum.charge,
                "length_bin": stratum.length_bin,
                "ptm_class": stratum.ptm_class,
                "eligible_population": population,
                "selected": quota,
                "selection_fraction": f"{quota / population:.8f}",
            }
        )
    write_tsv(
        out_dir / "stage_b_stratum_summary.tsv",
        [
            "stratum_key",
            "charge",
            "length_bin",
            "ptm_class",
            "eligible_population",
            "selected",
            "selection_fraction",
        ],
        stratum_rows,
    )

    for attribute, filename in [
        ("charge", "stage_b_by_charge.tsv"),
        ("length_bin", "stage_b_by_length_bin.tsv"),
        ("ptm_class", "stage_b_by_ptm_class.tsv"),
    ]:
        rows = dimension_rows(first.stratum_counts, quotas, attribute)
        write_tsv(
            out_dir / filename,
            [attribute, "eligible_population", "selected", "selection_fraction"],
            rows,
        )

    exclusion_rows = [
        {"exclusion_reason": reason, "identity_count": count}
        for reason, count in sorted(first.exclusion_counts.items())
    ]
    write_tsv(
        out_dir / "stage_b_exclusion_summary.tsv",
        ["exclusion_reason", "identity_count"],
        exclusion_rows,
    )

    chemistry_rows = [
        {"chemistry_status": status, "identity_count": count}
        for status, count in sorted(first.chemistry_status_counts.items())
    ]
    write_tsv(
        out_dir / "stage_b_input_chemistry_status.tsv",
        ["chemistry_status", "identity_count"],
        chemistry_rows,
    )

    summary = {
        "audit_version": AUDIT_VERSION,
        "input_identity_inventory": str(identity_inventory),
        "partition_scope": "TRAIN_identity_inventory_only",
        "measured_ccs_used_for_selection": "NO",
        "mobility_labels_used_for_selection": "NO",
        "dev_labels_used": "NO",
        "holdout_used": "NO",
        "selection_salt": SELECTION_SALT,
        "stratification": "charge_x_length_bin_x_ptm_class",
        "quota_policy": "minimum_then_sqrt_population_weighted_fill",
        "minimum_per_nonempty_stratum": str(minimum_per_stratum),
        "requested_target_size": str(target_size),
        "input_identity_rows": str(first.total_rows),
        "eligible_existing_template_candidate_identities": str(first.eligible_rows),
        "excluded_identities": str(first.total_rows - first.eligible_rows),
        "nonempty_strata": str(len(first.stratum_counts)),
        "selected_identities": str(selected_count),
        "manifest_fingerprint": fingerprint,
    }
    write_tsv(
        out_dir / "stage_b_summary.tsv",
        ["metric", "value"],
        ({"metric": key, "value": value} for key, value in summary.items()),
    )

    report = [
        "# ReDeeM structure-signal feasibility v1 — Stage B manifest",
        "",
        f"- Audit version: `{AUDIT_VERSION}`",
        "- Input scope: **TRAIN Stage A identity inventory only**",
        "- Measured CCS used for selection: **NO**",
        "- Mobility labels used for selection: **NO**",
        "- DEV labels used: **NO**",
        "- HOLDOUT used: **NO**",
        f"- Eligible chemistry universe: `{first.eligible_rows}` identities",
        f"- Selected pilot identities: `{selected_count}`",
        f"- Non-empty strata: `{len(first.stratum_counts)}`",
        f"- Manifest fingerprint: `{fingerprint}`",
        "",
        "## Selection contract",
        "",
        "Eligible identities must have positive precursor charge and Stage A's",
        "`existing_attachment_template_candidate=YES`. Composition-only and",
        "ambiguous/mass-only identities remain excluded rather than guessed.",
        "",
        "The manifest is stratified by exact precursor charge, Stage A sequence-length",
        "bin, and PTM class. Each non-empty stratum receives a minimum quota where",
        "capacity permits; remaining seats are allocated proportional to the square",
        "root of stratum population. Within strata, identities are selected by the",
        "smallest versioned SHA-256 rank over `identity_key`.",
        "",
        "This file only defines the conformer pilot cohort. It does not choose a",
        "conformer engine, build molecular structures, calculate descriptors, or train",
        "v0.70.",
        "",
    ]
    (out_dir / "stage_b_report.md").write_text("\n".join(report), encoding="utf-8")

    return summary


def synthetic_rows() -> tuple[list[str], list[dict[str, str]]]:
    fieldnames = sorted(REQUIRED_COLUMNS)
    rows: list[dict[str, str]] = []

    def row(
        idx: int,
        charge: int,
        length_bin: str,
        modified: bool,
        label: str,
        eligible: bool = True,
    ) -> dict[str, str]:
        sequence = "PEPTIDE" + ("A" * (idx % 5))
        return {
            "identity_key": f"id-{idx:04d}",
            "peptidoform": f"PEPTIDE{idx}",
            "sequence": sequence,
            "charge": str(charge),
            "sequence_length": str(len(sequence)),
            "length_bin": length_bin,
            "modified": "YES" if modified else "NO",
            "modification_count": "1" if modified else "0",
            "modification_labels": label,
            "record_count": "1",
            "source_count": "1",
            "precursor_mz_count": "1",
            "precursor_mz_mean": "500.0",
            "precursor_mz_min": "500.0",
            "precursor_mz_max": "500.0",
            "theoretical_neutral_mass_da": "998.0",
            "chemistry_status": (
                "current_exact_ptm_local_topology" if eligible else "ambiguous_or_unsupported"
            ),
            "current_exact_ptm_local_topology_ready": "YES" if eligible else "NO",
            "existing_attachment_template_candidate": "YES" if eligible else "NO",
            "all_modifications_composition_known": "YES" if eligible else "NO",
            "failure_reasons": "" if eligible else "mass_only_modification_unmatched",
        }

    idx = 0
    for charge in (2, 3, 4):
        for length_bin in ("02_8-12", "03_13-18"):
            for label, modified in [
                ("", False),
                ("UniMod:4@Residue:C:current_exact_ptm_local_topology", True),
                ("UniMod:35@Residue:M:current_exact_ptm_local_topology", True),
                ("UniMod:21@Residue:S:existing_attachment_template_candidate", True),
            ]:
                for _ in range(9):
                    rows.append(row(idx, charge, length_bin, modified, label))
                    idx += 1
    # A rare but valid stratum must survive minimum quota allocation.
    rows.append(
        row(
            idx,
            6,
            "06_36+",
            True,
            "UniMod:1@N-term:A:current_exact_ptm_local_topology",
        )
    )
    idx += 1
    # Explicitly excluded chemistry.
    rows.append(
        row(
            idx,
            2,
            "02_8-12",
            True,
            "Mass:+37.0066@Residue:M:ambiguous_or_unsupported",
            eligible=False,
        )
    )
    return fieldnames, rows


def write_synthetic_inventory(path: Path) -> None:
    fieldnames, rows = synthetic_rows()
    write_tsv(path, fieldnames, rows)


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="redeem-stage-b-selftest-") as tmp:
        root = Path(tmp)
        inventory = root / "identity_inventory.tsv"
        write_synthetic_inventory(inventory)

        out_a = root / "out-a"
        out_b = root / "out-b"
        summary_a = run_manifest(inventory, out_a, target_size=40, minimum_per_stratum=1)
        summary_b = run_manifest(inventory, out_b, target_size=40, minimum_per_stratum=1)
        if summary_a["manifest_fingerprint"] != summary_b["manifest_fingerprint"]:
            raise RuntimeError("self-test: manifest is not deterministic")
        if summary_a["selected_identities"] != "40":
            raise RuntimeError("self-test: target size was not respected")

        manifest = list(
            csv.DictReader(
                (out_a / "stage_b_manifest.tsv").open("r", encoding="utf-8"),
                delimiter="\t",
            )
        )
        if any(row["identity_key"].endswith("0217") for row in manifest):
            raise RuntimeError("self-test: unsupported chemistry leaked into manifest")
        if not any(row["charge"] == "6" for row in manifest):
            raise RuntimeError("self-test: rare charge stratum was not retained")

        # Prove that label-bearing inputs are rejected rather than silently ignored.
        forbidden = root / "forbidden.tsv"
        fieldnames, rows = synthetic_rows()
        fieldnames = [*fieldnames, "target_ccs"]
        with forbidden.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows[:2]:
                writer.writerow({**row, "target_ccs": "123.4"})
        try:
            first_pass(forbidden)
        except ValueError as exc:
            if "CCS/mobility" not in str(exc):
                raise
        else:
            raise RuntimeError("self-test: label-bearing input was not rejected")

    print("structure_signal_stage_b_manifest_v1_self_test=PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.identity_inventory is None or args.out_dir is None:
        raise SystemExit("--identity-inventory and --out-dir are required unless --self-test is used")
    summary = run_manifest(
        identity_inventory=args.identity_inventory,
        out_dir=args.out_dir,
        target_size=args.target_size,
        minimum_per_stratum=args.minimum_per_stratum,
    )
    for key in [
        "audit_version",
        "partition_scope",
        "input_identity_rows",
        "eligible_existing_template_candidate_identities",
        "excluded_identities",
        "nonempty_strata",
        "selected_identities",
        "manifest_fingerprint",
        "measured_ccs_used_for_selection",
        "dev_labels_used",
        "holdout_used",
    ]:
        print(f"{key}\t{summary[key]}")


if __name__ == "__main__":
    main()
