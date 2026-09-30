#!/usr/bin/env python3
"""TRAIN-only utility gate for the frozen ReDeeM Stage B structure descriptors.

This analysis asks one question: do frozen, label-independent structure descriptors
explain CCS residual variation beyond a fixed non-3D identity baseline after the
accepted v0.38 model has already made its prediction?

Scientific firewall:
* consumes only v0.38 predictions exported from the TRAIN partition;
* consumes the immutable frozen Stage B manifest and structure-feature dataset;
* groups all charge states of one peptidoform into the same cross-validation fold;
* uses one fixed model family and one fixed feature definition;
* performs no hyperparameter/descriptor search;
* never reads DEV, TRAIN-HOLDOUT, historical VALIDATION, or historical TEST labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

AUDIT_VERSION = "structure_signal_stage_b_utility_v1"
FOLD_SALT = "redeem_structure_signal_stage_b_utility_v1_groupfold"
BOOTSTRAP_SEED = 20_260_930
FOLDS = 5
BOOTSTRAP_REPLICATES = 5000
MIN_LABELED_IDENTITIES = 1000
MIN_PEPTIDOFORMS = 500
MIN_ABSOLUTE_MAE_GAIN = 0.15
MIN_RELATIVE_MAE_GAIN = 0.02

AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
PTM_CLASSES = (
    "unmodified",
    "carbamidomethyl",
    "oxidation",
    "deamidation",
    "phospho",
    "acetyl",
    "common_ptm_combination",
    "supported_mixed_other",
    "other_supported",
)
DESCRIPTOR_STEMS = (
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
)
STAT_SUFFIXES = ("median", "mean", "std", "min", "max")
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


@dataclass(frozen=True)
class IdentityLabel:
    identity_key: str
    peptidoform: str
    sequence: str
    charge: int
    target_ccs_median: float
    predicted_ccs_median: float
    residual_ccs_median: float
    raw_record_count: int
    source_count: int
    raw_record_mae: float


@dataclass(frozen=True)
class JoinedRow:
    label: IdentityLabel
    manifest: Mapping[str, str]
    structure: Mapping[str, str]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--structure-features", type=Path)
    parser.add_argument("--stage-b-summary", type=Path)
    parser.add_argument("--v038-train-predictions", type=Path)
    parser.add_argument("--expected-manifest-fingerprint")
    parser.add_argument("--expected-structure-fingerprint")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def read_summary(path: Path) -> dict[str, str]:
    fields, rows = read_tsv(path)
    if fields != ["metric", "value"]:
        raise ValueError(f"summary must contain exactly metric/value columns: {path}")
    summary: dict[str, str] = {}
    for row in rows:
        key = row["metric"]
        if key in summary:
            raise ValueError(f"duplicate summary metric {key!r}")
        summary[key] = row["value"]
    return summary


def manifest_fingerprint(rows: Iterable[Mapping[str, str]]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        for field in ("stratum_key", "selection_hash_sha256", "identity_key"):
            if not row.get(field):
                raise ValueError(f"manifest row lacks required {field}")
        hasher.update(row["stratum_key"].encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(row["selection_hash_sha256"].encode("ascii"))
        hasher.update(b"\t")
        hasher.update(row["identity_key"].encode("utf-8"))
        hasher.update(b"\n")
    return "sha256:" + hasher.hexdigest()


def ensure_unique(rows: Sequence[Mapping[str, str]], field: str, label: str) -> dict[str, Mapping[str, str]]:
    index: dict[str, Mapping[str, str]] = {}
    for row in rows:
        key = row.get(field, "")
        if not key:
            raise ValueError(f"{label} contains empty {field}")
        if key in index:
            raise ValueError(f"{label} contains duplicate {field} {key!r}")
        index[key] = row
    return index


def aggregate_v038_predictions(path: Path) -> tuple[dict[str, IdentityLabel], int, float]:
    fields, rows = read_tsv(path)
    required = {
        "source_id",
        "identity_key",
        "sequence",
        "peptidoform",
        "charge",
        "target_ccs",
        "predicted_ccs",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"v0.38 TRAIN prediction export lacks columns: {missing}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    raw_abs_error = 0.0
    raw_count = 0
    for row in rows:
        target = float(row["target_ccs"])
        predicted = float(row["predicted_ccs"])
        if not (math.isfinite(target) and math.isfinite(predicted)):
            raise ValueError("v0.38 TRAIN export contains non-finite CCS value")
        grouped[row["identity_key"]].append(row)
        raw_abs_error += abs(target - predicted)
        raw_count += 1
    if not grouped:
        raise ValueError("v0.38 TRAIN prediction export is empty")

    labels: dict[str, IdentityLabel] = {}
    for identity_key, group in grouped.items():
        sequences = {row["sequence"] for row in group}
        peptidoforms = {row["peptidoform"] for row in group}
        charges = {int(row["charge"]) for row in group}
        if len(sequences) != 1 or len(peptidoforms) != 1 or len(charges) != 1:
            raise ValueError(f"inconsistent v0.38 rows for identity {identity_key!r}")
        targets = [float(row["target_ccs"]) for row in group]
        predictions = [float(row["predicted_ccs"]) for row in group]
        residuals = [target - prediction for target, prediction in zip(targets, predictions)]
        labels[identity_key] = IdentityLabel(
            identity_key=identity_key,
            peptidoform=next(iter(peptidoforms)),
            sequence=next(iter(sequences)),
            charge=next(iter(charges)),
            target_ccs_median=float(statistics.median(targets)),
            predicted_ccs_median=float(statistics.median(predictions)),
            residual_ccs_median=float(statistics.median(residuals)),
            raw_record_count=len(group),
            source_count=len({row["source_id"] for row in group}),
            raw_record_mae=float(statistics.mean(abs(value) for value in residuals)),
        )
    return labels, raw_count, raw_abs_error / raw_count


def parse_float(row: Mapping[str, str], field: str) -> float:
    value = row.get(field, "").strip()
    if not value:
        return float("nan")
    return float(value)


def baseline_feature_names() -> list[str]:
    return [
        "charge",
        "sequence_length",
        "theoretical_neutral_mass_da",
        "precursor_mz_mean",
        "modification_count",
        *[f"aa_fraction_{aa}" for aa in AA_ORDER],
        *[f"ptm_class_{name}" for name in PTM_CLASSES],
    ]


def baseline_features(row: JoinedRow) -> list[float]:
    sequence = row.label.sequence
    length = max(len(sequence), 1)
    counts = Counter(sequence)
    ptm_class = row.manifest.get("ptm_class", "")
    return [
        float(row.label.charge),
        parse_float(row.manifest, "sequence_length"),
        parse_float(row.manifest, "theoretical_neutral_mass_da"),
        parse_float(row.manifest, "precursor_mz_mean"),
        parse_float(row.manifest, "modification_count"),
        *[counts.get(aa, 0) / length for aa in AA_ORDER],
        *[1.0 if ptm_class == name else 0.0 for name in PTM_CLASSES],
    ]


def choose_structure_columns(fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> list[str]:
    forbidden = FORBIDDEN_LABEL_COLUMNS.intersection(fieldnames)
    if forbidden:
        raise ValueError(f"structure features unexpectedly contain label columns: {sorted(forbidden)}")
    expected = [f"{stem}_{suffix}" for stem in DESCRIPTOR_STEMS for suffix in STAT_SUFFIXES]
    chosen = [name for name in expected if name in fieldnames]
    for extra in ("median_ensemble_max_rmsd",):
        if extra in fieldnames:
            chosen.append(extra)
    finite_columns: list[str] = []
    for name in chosen:
        if any(
            value != "" and math.isfinite(float(value))
            for row in rows
            if (value := row.get(name, "").strip())
        ):
            finite_columns.append(name)
    if len(finite_columns) < 20:
        raise ValueError(
            f"too few recognized finite structure descriptor columns: {len(finite_columns)}"
        )
    return finite_columns


def structure_features(row: JoinedRow, columns: Sequence[str]) -> list[float]:
    return [parse_float(row.structure, name) for name in columns]


def group_folds(groups: Sequence[str], folds: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("fold count must be >=2")
    unique = sorted(set(groups))
    if len(unique) < folds:
        raise ValueError("fewer peptidoform groups than folds")
    ranked = sorted(
        unique,
        key=lambda value: (
            hashlib.sha256(f"{FOLD_SALT}\t{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )
    return {group: index % folds for index, group in enumerate(ranked)}


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=20_260_930,
        early_stopping=False,
    )


def mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(target - prediction)))


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(target - prediction))))


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction, method="linear"))


def grouped_bootstrap_gain(
    groups: Sequence[str],
    baseline_abs_error: np.ndarray,
    augmented_abs_error: np.ndarray,
    replicates: int,
) -> tuple[float, float, float]:
    if replicates < 100:
        raise ValueError("bootstrap replicates must be >=100")
    group_array = np.asarray(groups, dtype=object)
    names = sorted(set(groups))
    baseline_sums = []
    augmented_sums = []
    counts = []
    for group in names:
        indices = np.flatnonzero(group_array == group)
        baseline_sums.append(float(baseline_abs_error[indices].sum()))
        augmented_sums.append(float(augmented_abs_error[indices].sum()))
        counts.append(int(indices.size))
    baseline_sums_array = np.asarray(baseline_sums, dtype=np.float64)
    augmented_sums_array = np.asarray(augmented_sums, dtype=np.float64)
    counts_array = np.asarray(counts, dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gains: list[float] = []
    group_count = len(names)
    for _ in range(replicates):
        sampled = rng.integers(0, group_count, size=group_count)
        count = int(counts_array[sampled].sum())
        gain = (
            float(baseline_sums_array[sampled].sum())
            - float(augmented_sums_array[sampled].sum())
        ) / max(count, 1)
        gains.append(gain)
    return (
        float(statistics.mean(gains)),
        percentile(gains, 0.025),
        percentile(gains, 0.975),
    )


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, summary: Mapping[str, object]) -> None:
    write_tsv(path, ["metric", "value"], ({"metric": key, "value": value} for key, value in summary.items()))


def run_utility(
    manifest_path: Path,
    structure_path: Path,
    stage_b_summary_path: Path,
    prediction_path: Path,
    expected_manifest_fingerprint: str,
    expected_structure_fingerprint: str,
    out_dir: Path,
    folds: int = FOLDS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_fields, manifest_rows = read_tsv(manifest_path)
    if len(manifest_rows) != 8000:
        raise ValueError(f"frozen Stage B manifest must contain 8000 identities, observed {len(manifest_rows)}")
    manifest_index = ensure_unique(manifest_rows, "identity_key", "manifest")
    observed_manifest_fingerprint = manifest_fingerprint(manifest_rows)
    if observed_manifest_fingerprint != expected_manifest_fingerprint:
        raise ValueError(
            "Stage B manifest fingerprint mismatch: "
            f"observed={observed_manifest_fingerprint} expected={expected_manifest_fingerprint}"
        )

    stage_b_summary = read_summary(stage_b_summary_path)
    if stage_b_summary.get("manifest_fingerprint") != expected_manifest_fingerprint:
        raise ValueError("Stage B final summary manifest fingerprint mismatch")
    if stage_b_summary.get("structure_feature_fingerprint") != expected_structure_fingerprint:
        raise ValueError("Stage B structure-feature fingerprint mismatch")
    if stage_b_summary.get("full_stage_b_operational_gate") != "PASS":
        raise ValueError("Stage B final structure dataset is not operationally certified")

    structure_fields, structure_rows = read_tsv(structure_path)
    structure_index = ensure_unique(structure_rows, "identity_key", "structure feature table")
    if set(structure_index) != set(manifest_index):
        raise ValueError("structure feature identity set differs from frozen manifest")
    structure_columns = choose_structure_columns(structure_fields, structure_rows)

    labels, raw_prediction_records, raw_v038_mae = aggregate_v038_predictions(prediction_path)
    unknown_labels = sorted(set(labels) - set(manifest_index))
    if unknown_labels:
        raise ValueError(f"v0.38 TRAIN export contains identities outside frozen manifest: {unknown_labels[:3]}")

    joined = [
        JoinedRow(label=labels[key], manifest=manifest_index[key], structure=structure_index[key])
        for key in sorted(labels)
    ]
    peptidoforms = [row.label.peptidoform for row in joined]
    unique_peptidoforms = len(set(peptidoforms))
    label_sufficiency = len(joined) >= MIN_LABELED_IDENTITIES and unique_peptidoforms >= MIN_PEPTIDOFORMS
    if not label_sufficiency:
        raise ValueError(
            "insufficient frozen TRAIN CCS overlap for the predeclared utility gate: "
            f"identities={len(joined)} peptidoforms={unique_peptidoforms}"
        )

    base_names = baseline_feature_names()
    base_x = np.asarray([baseline_features(row) for row in joined], dtype=np.float64)
    struct_x = np.asarray([structure_features(row, structure_columns) for row in joined], dtype=np.float64)
    augmented_x = np.concatenate([base_x, struct_x], axis=1)
    target = np.asarray([row.label.residual_ccs_median for row in joined], dtype=np.float64)
    if not np.all(np.isfinite(target)):
        raise ValueError("identity-level residual target is non-finite")

    assignments = group_folds(peptidoforms, folds)
    fold_index = np.asarray([assignments[group] for group in peptidoforms], dtype=np.int64)
    base_pred = np.full(len(joined), np.nan, dtype=np.float64)
    augmented_pred = np.full(len(joined), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []

    for fold in range(folds):
        test = fold_index == fold
        train = ~test
        if int(test.sum()) == 0 or int(train.sum()) == 0:
            raise ValueError(f"empty train/test set for fold {fold}")
        base_model = new_model()
        augmented_model = new_model()
        base_model.fit(base_x[train], target[train])
        augmented_model.fit(augmented_x[train], target[train])
        base_pred[test] = base_model.predict(base_x[test])
        augmented_pred[test] = augmented_model.predict(augmented_x[test])
        fold_rows.append(
            {
                "fold": fold,
                "train_identities": int(train.sum()),
                "test_identities": int(test.sum()),
                "test_peptidoforms": len({peptidoforms[i] for i in np.flatnonzero(test)}),
                "v038_residual_zero_mae": mae(target[test], np.zeros(int(test.sum()))),
                "current_information_mae": mae(target[test], base_pred[test]),
                "structure_augmented_mae": mae(target[test], augmented_pred[test]),
            }
        )

    if not (np.all(np.isfinite(base_pred)) and np.all(np.isfinite(augmented_pred))):
        raise ValueError("OOF predictions are incomplete")

    zero_pred = np.zeros(len(joined), dtype=np.float64)
    zero_mae = mae(target, zero_pred)
    base_mae = mae(target, base_pred)
    augmented_mae = mae(target, augmented_pred)
    base_rmse = rmse(target, base_pred)
    augmented_rmse = rmse(target, augmented_pred)
    absolute_gain = base_mae - augmented_mae
    relative_gain = absolute_gain / base_mae if base_mae > 0 else float("nan")
    vs_v038_gain = zero_mae - augmented_mae
    baseline_abs = np.abs(target - base_pred)
    augmented_abs = np.abs(target - augmented_pred)
    bootstrap_mean, ci_low, ci_high = grouped_bootstrap_gain(
        peptidoforms,
        baseline_abs,
        augmented_abs,
        bootstrap_replicates,
    )

    gate_abs = absolute_gain >= MIN_ABSOLUTE_MAE_GAIN
    gate_rel = relative_gain >= MIN_RELATIVE_MAE_GAIN
    gate_ci = ci_low > 0.0
    gate_beats_v038 = augmented_mae < zero_mae
    utility_pass = gate_abs and gate_rel and gate_ci and gate_beats_v038

    # Sensitivity subsets are descriptive only and cannot alter the gate.
    fallback_indices = np.asarray(
        [row.structure.get("carbonyl_charge_fallback", "NO") == "YES" for row in joined],
        dtype=bool,
    )
    no_carbonyl = ~fallback_indices
    sensitivity_no_carbonyl_gain = (
        mae(target[no_carbonyl], base_pred[no_carbonyl])
        - mae(target[no_carbonyl], augmented_pred[no_carbonyl])
        if int(no_carbonyl.sum()) > 0
        else float("nan")
    )

    prediction_rows = []
    for index, row in enumerate(joined):
        prediction_rows.append(
            {
                "identity_key": row.label.identity_key,
                "peptidoform": row.label.peptidoform,
                "sequence": row.label.sequence,
                "charge": row.label.charge,
                "fold": int(fold_index[index]),
                "raw_record_count": row.label.raw_record_count,
                "source_count": row.label.source_count,
                "target_ccs_median": f"{row.label.target_ccs_median:.8f}",
                "v038_predicted_ccs_median": f"{row.label.predicted_ccs_median:.8f}",
                "v038_residual_ccs_median": f"{target[index]:.8f}",
                "current_information_residual_prediction": f"{base_pred[index]:.8f}",
                "structure_augmented_residual_prediction": f"{augmented_pred[index]:.8f}",
                "current_information_abs_error": f"{baseline_abs[index]:.8f}",
                "structure_augmented_abs_error": f"{augmented_abs[index]:.8f}",
                "carbonyl_charge_fallback": row.structure.get("carbonyl_charge_fallback", ""),
                "random_coords_fallback_fraction": row.structure.get("random_coords_fallback_fraction", ""),
            }
        )

    write_tsv(
        out_dir / "utility_oof_predictions.tsv",
        list(prediction_rows[0]),
        prediction_rows,
    )
    write_tsv(out_dir / "utility_fold_metrics.tsv", list(fold_rows[0]), fold_rows)
    write_tsv(
        out_dir / "utility_structure_features.tsv",
        ["feature_index", "feature_name"],
        ({"feature_index": i, "feature_name": name} for i, name in enumerate(structure_columns, start=1)),
    )
    write_tsv(
        out_dir / "utility_baseline_features.tsv",
        ["feature_index", "feature_name"],
        ({"feature_index": i, "feature_name": name} for i, name in enumerate(base_names, start=1)),
    )

    summary: dict[str, object] = {
        "audit_version": AUDIT_VERSION,
        "partition_scope": "TRAIN_frozen_stage_b_ccs_overlap_only",
        "manifest_fingerprint": expected_manifest_fingerprint,
        "structure_feature_fingerprint": expected_structure_fingerprint,
        "v038_prediction_records": raw_prediction_records,
        "v038_prediction_record_mae": f"{raw_v038_mae:.8f}",
        "labeled_identities": len(joined),
        "labeled_peptidoforms": unique_peptidoforms,
        "folds": folds,
        "bootstrap_replicates": bootstrap_replicates,
        "baseline_feature_count": base_x.shape[1],
        "structure_feature_count": struct_x.shape[1],
        "model_family": "HistGradientBoostingRegressor_fixed_v1",
        "target": "median_TRAIN_raw_ccs_residual_vs_v038",
        "v038_zero_correction_oof_mae": f"{zero_mae:.8f}",
        "current_information_oof_mae": f"{base_mae:.8f}",
        "structure_augmented_oof_mae": f"{augmented_mae:.8f}",
        "current_information_oof_rmse": f"{base_rmse:.8f}",
        "structure_augmented_oof_rmse": f"{augmented_rmse:.8f}",
        "structure_absolute_mae_gain": f"{absolute_gain:.8f}",
        "structure_relative_mae_gain": f"{relative_gain:.8f}",
        "structure_gain_vs_uncorrected_v038": f"{vs_v038_gain:.8f}",
        "group_bootstrap_gain_mean": f"{bootstrap_mean:.8f}",
        "group_bootstrap_gain_ci95_low": f"{ci_low:.8f}",
        "group_bootstrap_gain_ci95_high": f"{ci_high:.8f}",
        "no_carbonyl_fallback_identities": int(no_carbonyl.sum()),
        "no_carbonyl_fallback_structure_absolute_gain": f"{sensitivity_no_carbonyl_gain:.8f}",
        "gate_min_labeled_identities_ge_1000": "PASS" if len(joined) >= MIN_LABELED_IDENTITIES else "FAIL",
        "gate_min_peptidoforms_ge_500": "PASS" if unique_peptidoforms >= MIN_PEPTIDOFORMS else "FAIL",
        "gate_absolute_mae_gain_ge_0_15": "PASS" if gate_abs else "FAIL",
        "gate_relative_mae_gain_ge_2pct": "PASS" if gate_rel else "FAIL",
        "gate_group_bootstrap_ci95_low_gt_0": "PASS" if gate_ci else "FAIL",
        "gate_augmented_beats_uncorrected_v038": "PASS" if gate_beats_v038 else "FAIL",
        "structure_utility_gate": "PASS" if utility_pass else "FAIL",
        "dev_labels_used": "NO",
        "train_holdout_consumed": "NO",
        "historical_validation_consumed": "NO",
        "historical_test_consumed": "NO",
    }
    write_summary(out_dir / "utility_summary.tsv", summary)

    report = f"""# ReDeeM frozen Stage B structure utility gate\n\n- Frozen manifest: `{expected_manifest_fingerprint}`\n- Frozen structure features: `{expected_structure_fingerprint}`\n- TRAIN CCS-labeled identities: `{len(joined)}` across `{unique_peptidoforms}` peptidoforms\n- Raw v0.38 selected TRAIN record MAE: `{raw_v038_mae:.4f}`\n- Identity median residual MAE without correction: `{zero_mae:.4f}`\n- Current-information residual OOF MAE: `{base_mae:.4f}`\n- Current-information + frozen structure OOF MAE: `{augmented_mae:.4f}`\n- Absolute structure gain: `{absolute_gain:.4f}` CCS units\n- Relative structure gain: `{relative_gain:.2%}`\n- Peptidoform-group bootstrap 95% CI: `[{ci_low:.4f}, {ci_high:.4f}]`\n- Structure utility gate: **{'PASS' if utility_pass else 'FAIL'}**\n\n## Predeclared gate\n\nPASS requires all of:\n\n1. at least {MIN_LABELED_IDENTITIES} labeled identities and {MIN_PEPTIDOFORMS} peptidoform groups;\n2. >= {MIN_ABSOLUTE_MAE_GAIN:.2f} CCS-unit OOF MAE improvement over the fixed non-3D residual baseline;\n3. >= {MIN_RELATIVE_MAE_GAIN:.0%} relative OOF MAE improvement;\n4. peptidoform-group bootstrap 95% CI lower bound > 0;\n5. the structure-augmented correction improves on uncorrected v0.38 identity residual MAE.\n\nNo DEV, TRAIN-HOLDOUT, historical VALIDATION, or historical TEST labels are used.\n"""
    (out_dir / "utility_report.md").write_text(report, encoding="utf-8")

    for key, value in summary.items():
        print(f"{key}\t{value}")
    return summary


def synthetic_inputs(root: Path, n: int = 1400) -> tuple[Path, Path, Path, Path, str, str]:
    manifest = root / "manifest.tsv"
    structure = root / "structure.tsv"
    summary = root / "stage_b_summary.tsv"
    predictions = root / "predictions.tsv"

    rng = np.random.default_rng(1234)
    manifest_rows = []
    structure_rows = []
    prediction_rows = []
    descriptor_fields = [f"{stem}_{suffix}" for stem in DESCRIPTOR_STEMS[:5] for suffix in STAT_SUFFIXES]
    for i in range(8000):
        peptidoform = f"PEPTIDE{i // 2:04d}"
        charge = 2 + (i % 2)
        identity = f"{peptidoform}|z{charge}"
        selection_hash = hashlib.sha256(f"identity-{i}".encode()).hexdigest()
        manifest_rows.append(
            {
                "stage_b_order": i + 1,
                "stratum_key": f"z{charge}|03_13-18|unmodified",
                "selection_hash_sha256": selection_hash,
                "identity_key": identity,
                "peptidoform": peptidoform,
                "sequence": "ACDEFGHIKLMNPQR",
                "charge": charge,
                "sequence_length": 15,
                "theoretical_neutral_mass_da": 1600 + i * 0.01,
                "precursor_mz_mean": (1600 + i * 0.01) / charge,
                "modification_count": 0,
                "ptm_class": "unmodified",
            }
        )
        digest_value = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)
        structure_signal = (digest_value / float(0xFFFFFFFFFFFFFFFF)) * 2.0 - 1.0
        srow: dict[str, object] = {
            "identity_key": identity,
            "carbonyl_charge_fallback": "NO",
            "random_coords_fallback_fraction": 0.5,
        }
        for j, name in enumerate(descriptor_fields):
            srow[name] = structure_signal + j * 0.01 + rng.normal(0, 0.02)
        structure_rows.append(srow)
        if i < n:
            residual = 0.8 * structure_signal + rng.normal(0, 0.15)
            target = 300 + i * 0.001
            predicted = target - residual
            prediction_rows.append(
                {
                    "record_index": i,
                    "source_id": "synthetic",
                    "identity_key": identity,
                    "sequence": "ACDEFGHIKLMNPQR",
                    "peptidoform": peptidoform,
                    "charge": charge,
                    "precursor_mz": 500,
                    "target_ccs": target,
                    "predicted_ccs": predicted,
                }
            )

    write_tsv(manifest, list(manifest_rows[0]), manifest_rows)
    fingerprint = manifest_fingerprint(manifest_rows)
    write_tsv(structure, list(structure_rows[0]), structure_rows)
    structure_fingerprint = "sha256:" + "a" * 64
    write_summary(
        summary,
        {
            "manifest_fingerprint": fingerprint,
            "structure_feature_fingerprint": structure_fingerprint,
            "full_stage_b_operational_gate": "PASS",
        },
    )
    write_tsv(predictions, list(prediction_rows[0]), prediction_rows)
    return manifest, structure, summary, predictions, fingerprint, structure_fingerprint


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="redeem-structure-utility-") as temp:
        root = Path(temp)
        manifest, structure, summary, predictions, fingerprint, structure_fingerprint = synthetic_inputs(root)
        out = root / "out"
        result = run_utility(
            manifest,
            structure,
            summary,
            predictions,
            fingerprint,
            structure_fingerprint,
            out,
            folds=5,
            bootstrap_replicates=200,
        )
        if result["structure_utility_gate"] != "PASS":
            raise RuntimeError("synthetic structure utility self-test did not pass")
        if int(result["labeled_identities"]) < MIN_LABELED_IDENTITIES:
            raise RuntimeError("synthetic self-test lost labeled identities")
        print("structure_signal_stage_b_utility_v1_self_test=PASS")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        self_test()
        return 0
    required = {
        "--manifest": args.manifest,
        "--structure-features": args.structure_features,
        "--stage-b-summary": args.stage_b_summary,
        "--v038-train-predictions": args.v038_train_predictions,
        "--expected-manifest-fingerprint": args.expected_manifest_fingerprint,
        "--expected-structure-fingerprint": args.expected_structure_fingerprint,
        "--out": args.out,
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(missing))
    run_utility(
        args.manifest,
        args.structure_features,
        args.stage_b_summary,
        args.v038_train_predictions,
        args.expected_manifest_fingerprint,
        args.expected_structure_fingerprint,
        args.out,
        folds=args.folds,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
