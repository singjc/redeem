#!/usr/bin/env python3
"""Deterministic CCS learnability/residual audit for matched DEV predictions.

This script does not train a model. It compares two per-record CCS prediction
exports (intended for v0.38 and v0.60) on the same protected DEV records and
quantifies whether their remaining errors are shared, complementary, source-
structured, or coupled to cross-source measurement disagreement.

Expected TSV columns (aliases are accepted):
  record_index, source_id, source_record_index, sequence, peptidoform,
  charge, precursor_mz, target_ccs, predicted_ccs

The prediction files may contain extra columns. Rows with non-finite CCS values
are skipped. No HOLDOUT/VALIDATION/TEST access is performed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

VERSION = "redeem-ccs-learnability-audit-v1"

ALIASES = {
    "record_index": ["record_index", "index", "corpus_record_index"],
    "source_id": ["source_id", "source", "dataset", "dataset_id"],
    "source_record_index": ["source_record_index", "source_index", "row_index"],
    "sequence": ["sequence", "peptide", "naked_sequence"],
    "peptidoform": ["peptidoform", "modified_sequence", "modified_peptide"],
    "charge": ["charge", "precursor_charge", "z"],
    "precursor_mz": ["precursor_mz", "mz", "precursor_m/z"],
    "target_ccs": ["target_ccs", "ccs_target", "observed_ccs", "ccs"],
    "predicted_ccs": ["predicted_ccs", "ccs_prediction", "prediction", "pred_ccs"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v038-predictions", type=Path)
    p.add_argument("--v060-predictions", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--source-family-map",
        type=Path,
        default=None,
        help="optional TSV with columns source_id and family_id",
    )
    p.add_argument(
        "--max-target-delta",
        type=float,
        default=1e-4,
        help="maximum allowed target CCS mismatch between aligned model exports",
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def finite_float(value: object) -> float | None:
    try:
        x = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_int(value: object) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def resolve_columns(fieldnames: Sequence[str]) -> Dict[str, str]:
    normalized = {name.strip().lower(): name for name in fieldnames}
    resolved: Dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias.lower() in normalized:
                resolved[canonical] = normalized[alias.lower()]
                break
    required = ["record_index", "source_id", "target_ccs", "predicted_ccs"]
    missing = [x for x in required if x not in resolved]
    if missing:
        raise ValueError(
            f"missing required columns {missing}; available columns={list(fieldnames)}"
        )
    return resolved


def load_predictions(path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: TSV header is missing")
        cols = resolve_columns(reader.fieldnames)
        for raw in reader:
            record_index = parse_int(raw.get(cols["record_index"]))
            target = finite_float(raw.get(cols["target_ccs"]))
            pred = finite_float(raw.get(cols["predicted_ccs"]))
            if record_index is None or target is None or pred is None:
                continue
            if record_index in rows:
                raise ValueError(f"{path}: duplicate record_index={record_index}")
            def get(name: str, default: str = "") -> str:
                col = cols.get(name)
                return str(raw.get(col, default)).strip() if col else default
            rows[record_index] = {
                "record_index": record_index,
                "source_id": get("source_id"),
                "source_record_index": parse_int(get("source_record_index")),
                "sequence": get("sequence"),
                "peptidoform": get("peptidoform") or get("sequence"),
                "charge": parse_int(get("charge")),
                "precursor_mz": finite_float(get("precursor_mz")),
                "target_ccs": target,
                "predicted_ccs": pred,
            }
    if not rows:
        raise ValueError(f"{path}: no finite prediction rows loaded")
    return rows


def load_family_map(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    out: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"source_id", "family_id"}.issubset(reader.fieldnames):
            raise ValueError("source-family map needs source_id and family_id columns")
        for row in reader:
            source = row["source_id"].strip()
            family = row["family_id"].strip()
            if source and family:
                out[source] = family
    return out


def default_family(source_id: str) -> str:
    lower = source_id.lower()
    if lower.startswith("pxd034128_") or lower == "pxd034128":
        return "pxd034128_family"
    if lower.startswith("pxd058337_") or lower == "pxd058337":
        return "pxd058337_family"
    return source_id


def source_family(source_id: str, mapping: Mapping[str, str]) -> str:
    return mapping.get(source_id, default_family(source_id))


def median(xs: Sequence[float]) -> float:
    return statistics.median(xs)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def mae(errors: Sequence[float]) -> float:
    return mean([abs(x) for x in errors])


def rmse(errors: Sequence[float]) -> float:
    return math.sqrt(mean([x * x for x in errors]))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denom if denom > 0 else math.nan


def ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(ranks(xs), ranks(ys)) if len(xs) >= 2 else math.nan


def safe_ratio(num: float, den: float) -> float:
    return num / den if den and math.isfinite(den) else math.nan


def ptm_count(peptidoform: str) -> int:
    # ReDeeM peptidoforms commonly retain bracketed modification annotations.
    return peptidoform.count("[")


def ptm_class(peptidoform: str) -> str:
    n = ptm_count(peptidoform)
    return "unmodified" if n == 0 else "one_ptm" if n == 1 else "multi_ptm"


def sequence_len(sequence: str) -> int:
    return sum(1 for c in sequence if c.isalpha() and c.isupper())


def length_bin(n: int) -> str:
    if n <= 7:
        return "01_le7"
    if n <= 11:
        return "02_8_11"
    if n <= 15:
        return "03_12_15"
    if n <= 20:
        return "04_16_20"
    return "05_ge21"


def chemistry_counts(sequence: str) -> Tuple[int, int, int]:
    seq = sequence.upper()
    acidic = sum(seq.count(x) for x in "DE")
    basic = sum(seq.count(x) for x in "KRH")
    aromatic = sum(seq.count(x) for x in "FWY")
    return acidic, basic, aromatic


def quantile_edges(values: Sequence[float]) -> Tuple[float, float, float]:
    vals = sorted(x for x in values if math.isfinite(x))
    if not vals:
        return (math.nan, math.nan, math.nan)
    def q(p: float) -> float:
        pos = p * (len(vals) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return q(0.25), q(0.5), q(0.75)


def mz_bin(x: float | None, edges: Tuple[float, float, float]) -> str:
    if x is None or not math.isfinite(x) or not all(math.isfinite(e) for e in edges):
        return "missing"
    q1, q2, q3 = edges
    if x <= q1:
        return "q1"
    if x <= q2:
        return "q2"
    if x <= q3:
        return "q3"
    return "q4"


def identity_key(row: Mapping[str, object]) -> str:
    pep = str(row.get("peptidoform") or row.get("sequence") or "")
    charge = row.get("charge")
    return f"{pep}|z={charge if charge is not None else 'NA'}"


def align(a: Mapping[int, dict], b: Mapping[int, dict], max_target_delta: float) -> List[dict]:
    shared = sorted(set(a).intersection(b))
    if not shared:
        raise ValueError("prediction exports have no shared record_index values")
    out: List[dict] = []
    mismatches = 0
    for idx in shared:
        ra, rb = a[idx], b[idx]
        if abs(ra["target_ccs"] - rb["target_ccs"]) > max_target_delta:
            mismatches += 1
            continue
        source_a = ra["source_id"]
        source_b = rb["source_id"]
        if source_a and source_b and source_a != source_b:
            raise ValueError(f"record {idx}: source_id mismatch {source_a!r} != {source_b!r}")
        row = dict(ra)
        row["pred_v038"] = ra["predicted_ccs"]
        row["pred_v060"] = rb["predicted_ccs"]
        row["err_v038"] = ra["predicted_ccs"] - ra["target_ccs"]
        row["err_v060"] = rb["predicted_ccs"] - rb["target_ccs"]
        out.append(row)
    if mismatches:
        raise ValueError(f"target CCS mismatch in {mismatches} shared rows")
    return out


def add_derived(rows: List[dict], family_map: Mapping[str, str]) -> None:
    mz_edges = quantile_edges([r["precursor_mz"] for r in rows if r["precursor_mz"] is not None])
    for r in rows:
        r["identity"] = identity_key(r)
        r["source_family"] = source_family(r["source_id"], family_map)
        n = sequence_len(r["sequence"])
        acidic, basic, aromatic = chemistry_counts(r["sequence"])
        r["sequence_length"] = n
        r["length_bin"] = length_bin(n)
        r["ptm_count"] = ptm_count(r["peptidoform"])
        r["ptm_class"] = ptm_class(r["peptidoform"])
        r["acidic_count"] = acidic
        r["basic_count"] = basic
        r["aromatic_count"] = aromatic
        r["mz_bin"] = mz_bin(r["precursor_mz"], mz_edges)


def identity_repeatability(rows: Sequence[dict]) -> Tuple[List[dict], Dict[str, dict]]:
    by_id: MutableMapping[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_id[r["identity"]].append(r)

    per_row: List[dict] = []
    summaries: Dict[str, dict] = {}
    for ident, members in by_id.items():
        family_targets: MutableMapping[str, List[float]] = defaultdict(list)
        for r in members:
            family_targets[r["source_family"]].append(r["target_ccs"])
        family_means = {f: mean(v) for f, v in family_targets.items()}
        targets = list(family_means.values())
        if len(targets) >= 2:
            center = median(targets)
            pair_abs = []
            fams = sorted(family_means)
            for i, f1 in enumerate(fams):
                for f2 in fams[i + 1 :]:
                    pair_abs.append(abs(family_means[f1] - family_means[f2]))
            summaries[ident] = {
                "identity": ident,
                "families": len(family_means),
                "records": len(members),
                "family_median_ccs": center,
                "family_pairwise_mae": mean(pair_abs),
                "family_mad": median([abs(x - center) for x in targets]),
            }
            for r in members:
                others = [v for f, v in family_means.items() if f != r["source_family"]]
                if not others:
                    continue
                oracle = median(others)
                per_row.append(
                    {
                        "record_index": r["record_index"],
                        "identity": ident,
                        "source_family": r["source_family"],
                        "target_ccs": r["target_ccs"],
                        "leave_one_family_out_ccs": oracle,
                        "repeatability_error": oracle - r["target_ccs"],
                        "family_pairwise_mae": summaries[ident]["family_pairwise_mae"],
                        "family_mad": summaries[ident]["family_mad"],
                    }
                )
    return per_row, summaries


def source_bias_adjusted_mae(rows: Sequence[dict], error_key: str) -> Tuple[float, float]:
    by_source: MutableMapping[str, List[float]] = defaultdict(list)
    for r in rows:
        by_source[r["source_family"]].append(r[error_key])
    medians = {s: median(v) for s, v in by_source.items()}
    base = mae([r[error_key] for r in rows])
    adjusted = mae([r[error_key] - medians[r["source_family"]] for r in rows])
    return adjusted, 1.0 - safe_ratio(adjusted, base)


def summarize_group(rows: Sequence[dict], group_key: str) -> List[dict]:
    groups: MutableMapping[str, List[dict]] = defaultdict(list)
    for r in rows:
        value = r.get(group_key)
        groups[str(value if value is not None else "missing")].append(r)
    out = []
    for group in sorted(groups):
        members = groups[group]
        e38 = [x["err_v038"] for x in members]
        e60 = [x["err_v060"] for x in members]
        out.append({
            group_key: group,
            "records": len(members),
            "v038_mae": mae(e38),
            "v060_mae": mae(e60),
            "v038_bias": mean(e38),
            "v060_bias": mean(e60),
            "residual_pearson": pearson(e38, e60),
        })
    return out


def classify(summary: Mapping[str, float]) -> Tuple[str, List[str]]:
    # Diagnostic thresholds are deliberately predeclared and conservative. They
    # are not statistical hypothesis tests and must not be used to tune on DEV.
    corr = summary["residual_pearson"]
    sign = summary["residual_sign_agreement"]
    oracle_gain = summary["oracle_best_relative_gain_vs_best_model"]
    floor_ratio = summary.get("repeatability_mae_over_best_model_mae", math.nan)
    src_gain = max(
        summary["v038_source_bias_diagnostic_relative_gain"],
        summary["v060_source_bias_diagnostic_relative_gain"],
    )
    spread_corr = summary.get("abs_error_vs_identity_pairwise_disagreement_pearson_best", math.nan)

    common = math.isfinite(corr) and corr >= 0.75 and sign >= 0.70
    complementarity = math.isfinite(oracle_gain) and oracle_gain >= 0.10
    low_complementarity = math.isfinite(oracle_gain) and oracle_gain < 0.05
    label_domain = (
        (math.isfinite(floor_ratio) and floor_ratio >= 0.50)
        or src_gain >= 0.10
        or (math.isfinite(spread_corr) and spread_corr >= 0.25)
    )

    reasons = []
    if common:
        reasons.append("strong_shared_residual_component")
    if complementarity:
        reasons.append("strong_model_complementarity")
    if low_complementarity:
        reasons.append("weak_model_complementarity")
    if label_domain:
        reasons.append("substantial_measurement_or_domain_component")

    if common and low_complementarity and label_domain:
        return "NEW_INFORMATION_OR_STOP", reasons
    if complementarity and not common:
        return "REPRESENTATION_FUSION_REMAINS_PLAUSIBLE", reasons
    if common and low_complementarity:
        return "NEW_INDUCTIVE_BIAS_REQUIRED", reasons
    return "MIXED_REQUIRE_STRUCTURE_SIGNAL_BEFORE_NEW_MODEL", reasons


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(x: object) -> str:
    if isinstance(x, float):
        return "NA" if not math.isfinite(x) else f"{x:.8f}"
    return str(x)


def run(v038_path: Path, v060_path: Path, out: Path, family_map_path: Path | None, max_target_delta: float) -> dict:
    v038 = load_predictions(v038_path)
    v060 = load_predictions(v060_path)
    rows = align(v038, v060, max_target_delta)
    family_map = load_family_map(family_map_path)
    add_derived(rows, family_map)

    e38 = [r["err_v038"] for r in rows]
    e60 = [r["err_v060"] for r in rows]
    a38 = [abs(x) for x in e38]
    a60 = [abs(x) for x in e60]
    v038_mae = mae(e38)
    v060_mae = mae(e60)
    best_mae = min(v038_mae, v060_mae)
    oracle_best_mae = mean([min(x, y) for x, y in zip(a38, a60)])
    ensemble_errors = [0.5 * (r["pred_v038"] + r["pred_v060"]) - r["target_ccs"] for r in rows]
    sign_agreement = mean([
        1.0 if ((x == 0 and y == 0) or (x > 0 and y > 0) or (x < 0 and y < 0)) else 0.0
        for x, y in zip(e38, e60)
    ])

    rep_rows, rep_summary = identity_repeatability(rows)
    repeatability_mae = mae([r["repeatability_error"] for r in rep_rows]) if rep_rows else math.nan

    # Attach per-identity disagreement for residual association analysis.
    spread_pairs_best_x: List[float] = []
    spread_pairs_best_y: List[float] = []
    best_error_key = "err_v038" if v038_mae <= v060_mae else "err_v060"
    for r in rows:
        info = rep_summary.get(r["identity"])
        if info:
            spread_pairs_best_x.append(info["family_pairwise_mae"])
            spread_pairs_best_y.append(abs(r[best_error_key]))
            r["identity_family_pairwise_mae"] = info["family_pairwise_mae"]
            r["identity_family_mad"] = info["family_mad"]
        else:
            r["identity_family_pairwise_mae"] = math.nan
            r["identity_family_mad"] = math.nan

    v038_adjusted, v038_source_gain = source_bias_adjusted_mae(rows, "err_v038")
    v060_adjusted, v060_source_gain = source_bias_adjusted_mae(rows, "err_v060")

    summary = {
        "audit_version": VERSION,
        "aligned_records": len(rows),
        "v038_records_loaded": len(v038),
        "v060_records_loaded": len(v060),
        "v038_mae": v038_mae,
        "v060_mae": v060_mae,
        "best_model": "v038" if v038_mae <= v060_mae else "v060",
        "best_model_mae": best_mae,
        "residual_pearson": pearson(e38, e60),
        "residual_spearman": spearman(e38, e60),
        "absolute_error_pearson": pearson(a38, a60),
        "absolute_error_spearman": spearman(a38, a60),
        "residual_sign_agreement": sign_agreement,
        "oracle_best_mae": oracle_best_mae,
        "oracle_best_relative_gain_vs_best_model": 1.0 - safe_ratio(oracle_best_mae, best_mae),
        "equal_weight_ensemble_mae": mae(ensemble_errors),
        "equal_weight_ensemble_relative_gain_vs_best_model": 1.0 - safe_ratio(mae(ensemble_errors), best_mae),
        "multifamily_identity_count": len(rep_summary),
        "repeatability_rows": len(rep_rows),
        "leave_one_source_family_out_repeatability_mae": repeatability_mae,
        "repeatability_mae_over_best_model_mae": safe_ratio(repeatability_mae, best_mae),
        "v038_source_bias_diagnostic_mae": v038_adjusted,
        "v038_source_bias_diagnostic_relative_gain": v038_source_gain,
        "v060_source_bias_diagnostic_mae": v060_adjusted,
        "v060_source_bias_diagnostic_relative_gain": v060_source_gain,
        "abs_error_vs_identity_pairwise_disagreement_pearson_best": pearson(spread_pairs_best_x, spread_pairs_best_y),
        "abs_error_vs_identity_pairwise_disagreement_spearman_best": spearman(spread_pairs_best_x, spread_pairs_best_y),
    }
    decision, reasons = classify(summary)
    summary["decision"] = decision
    summary["decision_reasons"] = ",".join(reasons)

    out.mkdir(parents=True, exist_ok=True)
    aligned_columns = [
        "record_index", "source_id", "source_record_index", "source_family", "sequence",
        "peptidoform", "charge", "precursor_mz", "target_ccs", "pred_v038", "pred_v060",
        "err_v038", "err_v060", "sequence_length", "length_bin", "ptm_count", "ptm_class",
        "acidic_count", "basic_count", "aromatic_count", "mz_bin",
        "identity_family_pairwise_mae", "identity_family_mad",
    ]
    write_tsv(out / "aligned_predictions.tsv", rows, aligned_columns)
    if rep_rows:
        write_tsv(
            out / "repeatability_leave_one_family_out.tsv",
            rep_rows,
            ["record_index", "identity", "source_family", "target_ccs", "leave_one_family_out_ccs", "repeatability_error", "family_pairwise_mae", "family_mad"],
        )
        write_tsv(
            out / "repeatability_identity_summary.tsv",
            list(rep_summary.values()),
            ["identity", "families", "records", "family_median_ccs", "family_pairwise_mae", "family_mad"],
        )

    for key, name in [
        ("source_family", "residual_by_source_family.tsv"),
        ("charge", "residual_by_charge.tsv"),
        ("length_bin", "residual_by_length_bin.tsv"),
        ("ptm_class", "residual_by_ptm_class.tsv"),
        ("mz_bin", "residual_by_mz_quartile.tsv"),
    ]:
        grouped = summarize_group(rows, key)
        write_tsv(out / name, grouped, [key, "records", "v038_mae", "v060_mae", "v038_bias", "v060_bias", "residual_pearson"])

    with (out / "summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        for key, value in summary.items():
            handle.write(f"{key}\t{fmt(value)}\n")
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report = [
        "# ReDeeM CCS learnability audit v1",
        "",
        f"Aligned DEV records: **{len(rows)}**",
        "",
        "## Model residual agreement",
        "",
        f"- v0.38 MAE: `{v038_mae:.6f}`",
        f"- v0.60 MAE: `{v060_mae:.6f}`",
        f"- residual Pearson: `{summary['residual_pearson']:.4f}`",
        f"- residual Spearman: `{summary['residual_spearman']:.4f}`",
        f"- residual sign agreement: `{100*sign_agreement:.2f}%`",
        f"- per-record oracle-best MAE: `{oracle_best_mae:.6f}`",
        f"- oracle-best relative gain over the better single model: `{100*summary['oracle_best_relative_gain_vs_best_model']:.2f}%`",
        f"- equal-weight v0.38/v0.60 ensemble MAE: `{summary['equal_weight_ensemble_mae']:.6f}`",
        "",
        "## Measurement / source diagnostics",
        "",
        f"- multi-source-family identities: `{len(rep_summary)}`",
        f"- leave-one-source-family-out repeatability rows: `{len(rep_rows)}`",
        f"- leave-one-source-family-out repeatability MAE: `{fmt(repeatability_mae)}`",
        f"- repeatability MAE / best-model MAE: `{fmt(summary['repeatability_mae_over_best_model_mae'])}`",
        f"- v0.38 source-bias diagnostic relative gain: `{100*summary['v038_source_bias_diagnostic_relative_gain']:.2f}%`",
        f"- v0.60 source-bias diagnostic relative gain: `{100*summary['v060_source_bias_diagnostic_relative_gain']:.2f}%`",
        f"- best-model abs-error vs identity disagreement Pearson: `{fmt(summary['abs_error_vs_identity_pairwise_disagreement_pearson_best'])}`",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        "Reasons: " + (", ".join(reasons) if reasons else "none of the predeclared diagnostic criteria fired"),
        "",
        "The source-bias-adjusted and oracle-best numbers are diagnostics only. They use DEV labels and must not be reported as deployable model performance or used to tune a model on DEV.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        header = "record_index\tsource_id\tsource_record_index\tsequence\tpeptidoform\tcharge\tprecursor_mz\ttarget_ccs\tpredicted_ccs\n"
        rows38 = [
            "0\ta\t0\tPEPTIDEK\tPEPTIDEK\t2\t500\t100\t101\n",
            "1\tb\t0\tPEPTIDEK\tPEPTIDEK\t2\t500\t104\t105\n",
            "2\ta\t1\tACDEK\tACDEK[+15.99]\t3\t600\t120\t118\n",
            "3\tb\t1\tACDEK\tACDEK[+15.99]\t3\t600\t124\t122\n",
        ]
        rows60 = [
            "0\ta\t0\tPEPTIDEK\tPEPTIDEK\t2\t500\t100\t101.5\n",
            "1\tb\t0\tPEPTIDEK\tPEPTIDEK\t2\t500\t104\t105.5\n",
            "2\ta\t1\tACDEK\tACDEK[+15.99]\t3\t600\t120\t119\n",
            "3\tb\t1\tACDEK\tACDEK[+15.99]\t3\t600\t124\t123\n",
        ]
        p38 = root / "v38.tsv"; p60 = root / "v60.tsv"
        p38.write_text(header + "".join(rows38), encoding="utf-8")
        p60.write_text(header + "".join(rows60), encoding="utf-8")
        summary = run(p38, p60, root / "out", None, 1e-4)
        assert summary["aligned_records"] == 4
        assert summary["multifamily_identity_count"] == 2
        assert math.isfinite(summary["residual_pearson"])
        assert (root / "out" / "report.md").exists()
        assert (root / "out" / "residual_by_charge.tsv").exists()
    print("self_test=PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.v038_predictions is None or args.v060_predictions is None or args.out is None:
        raise SystemExit("--v038-predictions, --v060-predictions, and --out are required unless --self-test is used")
    summary = run(
        args.v038_predictions,
        args.v060_predictions,
        args.out,
        args.source_family_map,
        args.max_target_delta,
    )
    for key in [
        "aligned_records", "v038_mae", "v060_mae", "residual_pearson",
        "residual_sign_agreement", "oracle_best_mae",
        "oracle_best_relative_gain_vs_best_model",
        "leave_one_source_family_out_repeatability_mae",
        "repeatability_mae_over_best_model_mae",
        "decision", "decision_reasons",
    ]:
        print(f"{key}\t{fmt(summary[key])}")


if __name__ == "__main__":
    main()
