#!/usr/bin/env python3
"""Render the frozen v0.52 historical VALIDATION RT/MS2 comparison against AlphaPeptDeep 1.5.1."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def metric(summary: pd.DataFrame, prop: str, model: str, column: str) -> float:
    row = summary[(summary["property"] == prop) & (summary["model"] == model)]
    if len(row) != 1 or column not in row:
        return math.nan
    return float(pd.to_numeric(row.iloc[0][column], errors="coerce"))


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.8f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = pd.read_csv(args.comparison_dir / "alphapeptdeep_comparison_summary.tsv", sep="\t")
    precursors = pd.read_csv(args.comparison_dir / "alphapeptdeep_comparison.tsv", sep="\t", na_values=["NA"])
    metrics = pd.read_csv(args.comparison_dir / "alphapeptdeep_comparison_ms2_metrics.tsv", sep="\t", na_values=["NA"])

    partitions = set(precursors["partition"].dropna().astype(str).unique())
    if not partitions.issubset({"VALIDATION"}):
        raise SystemExit(f"v0.52 APD report accepts historical VALIDATION rows only; observed {sorted(partitions)}")
    if (precursors["selected_ccs"].astype(str) != "NO").any():
        raise SystemExit("v0.52 APD report is RT/MS2-only; selected_ccs must be NO for every row")

    rt_f_mae = metric(summary, "RT", "foundation", "mae")
    rt_a_mae = metric(summary, "RT", "alphapeptdeep", "mae")
    rt_f_r = metric(summary, "RT", "foundation", "pearson")
    rt_a_r = metric(summary, "RT", "alphapeptdeep", "pearson")
    ms2_f_cos = metric(summary, "MS2", "foundation", "mean_cosine")
    ms2_a_cos = metric(summary, "MS2", "alphapeptdeep", "mean_cosine")
    ms2_f_sa = metric(summary, "MS2", "foundation", "mean_spectral_angle")
    ms2_a_sa = metric(summary, "MS2", "alphapeptdeep", "mean_spectral_angle")
    ms2_f_p = metric(summary, "MS2", "foundation", "pearson")
    ms2_a_p = metric(summary, "MS2", "alphapeptdeep", "pearson")

    f_sa = pd.to_numeric(metrics["foundation_spectral_angle"], errors="coerce")
    a_sa = pd.to_numeric(metrics["alphapeptdeep_spectral_angle"], errors="coerce")
    delta = (f_sa - a_sa).dropna()
    redeem_better = float((delta > 0).mean()) if len(delta) else math.nan
    apd_better = float((delta < 0).mean()) if len(delta) else math.nan
    median_delta = float(delta.median()) if len(delta) else math.nan

    apd_version = "unknown"
    if "alphapeptdeep_version" in summary and summary["alphapeptdeep_version"].notna().any():
        apd_version = str(summary["alphapeptdeep_version"].dropna().iloc[0])

    headline = pd.DataFrame(
        [
            {"property": "RT", "metric": "mae", "redeem_v0520": rt_f_mae, "alphapeptdeep": rt_a_mae},
            {"property": "RT", "metric": "pearson", "redeem_v0520": rt_f_r, "alphapeptdeep": rt_a_r},
            {"property": "MS2", "metric": "mean_cosine", "redeem_v0520": ms2_f_cos, "alphapeptdeep": ms2_a_cos},
            {"property": "MS2", "metric": "mean_spectral_angle", "redeem_v0520": ms2_f_sa, "alphapeptdeep": ms2_a_sa},
            {"property": "MS2", "metric": "mean_pearson", "redeem_v0520": ms2_f_p, "alphapeptdeep": ms2_a_p},
        ]
    )
    headline.to_csv(args.output_dir / "v0520_apd_headline.tsv", sep="\t", index=False)

    report = args.output_dir / "v0520_apd_report.md"
    report.write_text(
        "# ReDeeM v0.52 vs AlphaPeptDeep historical VALIDATION\n\n"
        f"- AlphaPeptDeep version: `{apd_version}`\n"
        "- Evaluation partition: `historical VALIDATION`\n"
        "- RT calibration partition: `TRAIN` only\n"
        "- TRAIN-HOLDOUT consumed previously: `YES`\n"
        "- historical TEST consumed: `NO`\n"
        "- Comparison scope: `RT and MS2 only`; CCS remains frozen separately at v0.38.\n\n"
        "## RT\n\n"
        f"- ReDeeM v0.52 MAE: `{fmt(rt_f_mae)}`; Pearson: `{fmt(rt_f_r)}`\n"
        f"- AlphaPeptDeep MAE: `{fmt(rt_a_mae)}`; Pearson: `{fmt(rt_a_r)}`\n\n"
        "## MS2\n\n"
        f"- ReDeeM v0.52 mean cosine: `{fmt(ms2_f_cos)}`; mean spectral angle: `{fmt(ms2_f_sa)}`; mean Pearson: `{fmt(ms2_f_p)}`\n"
        f"- AlphaPeptDeep mean cosine: `{fmt(ms2_a_cos)}`; mean spectral angle: `{fmt(ms2_a_sa)}`; mean Pearson: `{fmt(ms2_a_p)}`\n"
        f"- ReDeeM higher per-spectrum spectral angle: `{fmt(redeem_better)}`\n"
        f"- AlphaPeptDeep higher per-spectrum spectral angle: `{fmt(apd_better)}`\n"
        f"- Median ReDeeM - AlphaPeptDeep spectral-angle delta: `{fmt(median_delta)}`\n\n"
        "This comparison is descriptive only. The v0.52 checkpoint was selected before historical VALIDATION was opened and must not be changed or tuned in response to this result.\n"
    )

    print(f"alphapeptdeep_version\t{apd_version}")
    print(f"rt_redeem_mae\t{fmt(rt_f_mae)}")
    print(f"rt_apd_mae\t{fmt(rt_a_mae)}")
    print(f"rt_redeem_pearson\t{fmt(rt_f_r)}")
    print(f"rt_apd_pearson\t{fmt(rt_a_r)}")
    print(f"ms2_redeem_cosine\t{fmt(ms2_f_cos)}")
    print(f"ms2_apd_cosine\t{fmt(ms2_a_cos)}")
    print(f"ms2_redeem_spectral_angle\t{fmt(ms2_f_sa)}")
    print(f"ms2_apd_spectral_angle\t{fmt(ms2_a_sa)}")
    print(f"ms2_redeem_pearson\t{fmt(ms2_f_p)}")
    print(f"ms2_apd_pearson\t{fmt(ms2_a_p)}")
    print(f"ms2_redeem_better_fraction\t{fmt(redeem_better)}")
    print(f"ms2_apd_better_fraction\t{fmt(apd_better)}")
    print(f"ms2_median_sa_delta_redeem_minus_apd\t{fmt(median_delta)}")
    print("train_holdout_consumed\tYES")
    print("historical_validation_consumed\tYES")
    print("historical_test_consumed\tNO")


if __name__ == "__main__":
    main()
