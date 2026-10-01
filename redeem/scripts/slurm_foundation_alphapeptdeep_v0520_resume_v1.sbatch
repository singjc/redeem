#!/usr/bin/env bash
#SBATCH --job-name=redeem-apd-v0520-resume
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=redeem-apd-v0520-resume-%j.out

module load singularityce/4.1.0

: "${APD_SIF:?Set APD_SIF to the frozen AlphaPeptDeep 1.5.1 research SIF}"
: "${OUT:?Set OUT to the existing v0.52 historical VALIDATION comparison directory}"
: "${REPORT_SCRIPT:?Set REPORT_SCRIPT to the frozen v0.52 APD report script}"

EXPECTED_RT_RECORDS="${EXPECTED_RT_RECORDS:-121193}"
EXPECTED_MS2_RECORDS="${EXPECTED_MS2_RECORDS:-102554}"
EXPECTED_UNION_RECORDS="${EXPECTED_UNION_RECORDS:-121193}"
EXPECTED_RT_CALIBRATION="${EXPECTED_RT_CALIBRATION:-256}"

EXPORT="$OUT/export"
EXPORT_LOG="$OUT/export.log"
PRECURSORS="$EXPORT/foundation_comparison_precursors.tsv"
FRAGMENTS="$EXPORT/foundation_comparison_fragments.tsv"
RT_CALIBRATION="$EXPORT/foundation_comparison_rt_train_calibration.tsv"
FROZEN_HASHES="$OUT/export_frozen_sha256.tsv"

[ -f "$APD_SIF" ] || { echo "AlphaPeptDeep SIF not found: $APD_SIF" >&2; exit 2; }
[ -d "$OUT" ] || { echo "existing comparison OUT not found: $OUT" >&2; exit 2; }
[ -d "$EXPORT" ] || { echo "completed export directory not found: $EXPORT" >&2; exit 2; }
[ -f "$EXPORT_LOG" ] || { echo "completed export log not found: $EXPORT_LOG" >&2; exit 2; }
[ -f "$PRECURSORS" ] || { echo "precursor export missing: $PRECURSORS" >&2; exit 2; }
[ -f "$FRAGMENTS" ] || { echo "fragment export missing: $FRAGMENTS" >&2; exit 2; }
[ -f "$RT_CALIBRATION" ] || { echo "RT calibration export missing: $RT_CALIBRATION" >&2; exit 2; }
[ -f "$REPORT_SCRIPT" ] || { echo "v0.52 report script not found: $REPORT_SCRIPT" >&2; exit 2; }
[ ! -e "$OUT/comparison" ] || { echo "comparison output must be fresh: $OUT/comparison" >&2; exit 2; }
[ ! -e "$OUT/report" ] || { echo "report output must be fresh: $OUT/report" >&2; exit 2; }

ENGINE="$(command -v singularity)"
[ -n "$ENGINE" ] || { echo "singularity not available after module load" >&2; exit 127; }

read_metric() {
  awk -F $'\t' -v key="$1" '$1 == key {print $2; exit}' "$EXPORT_LOG"
}

EXPORT_VERSION="$(read_metric comparison_export_version)"
PARTITION="$(read_metric partition)"
RT_RECORDS="$(read_metric validation_rt_records)"
MS2_RECORDS="$(read_metric validation_ms2_records)"
UNION_RECORDS="$(read_metric validation_union_records)"
RT_CALIBRATION_RECORDS="$(read_metric rt_train_calibration_records)"
VALIDATION_CONSUMED="$(read_metric historical_validation_consumed)"
TEST_CONSUMED="$(read_metric historical_test_consumed)"

[ "$EXPORT_VERSION" = "foundation_v0520_historical_validation_apd_export_v1" ] || {
  echo "unexpected export version: $EXPORT_VERSION" >&2
  exit 2
}
[ "$PARTITION" = "HISTORICAL_VALIDATION" ] || {
  echo "unexpected export partition: $PARTITION" >&2
  exit 2
}
[ "$RT_RECORDS" = "$EXPECTED_RT_RECORDS" ] || {
  echo "historical VALIDATION RT cohort mismatch: $RT_RECORDS != $EXPECTED_RT_RECORDS" >&2
  exit 2
}
[ "$MS2_RECORDS" = "$EXPECTED_MS2_RECORDS" ] || {
  echo "historical VALIDATION MS2 cohort mismatch: $MS2_RECORDS != $EXPECTED_MS2_RECORDS" >&2
  exit 2
}
[ "$UNION_RECORDS" = "$EXPECTED_UNION_RECORDS" ] || {
  echo "historical VALIDATION union cohort mismatch: $UNION_RECORDS != $EXPECTED_UNION_RECORDS" >&2
  exit 2
}
[ "$RT_CALIBRATION_RECORDS" = "$EXPECTED_RT_CALIBRATION" ] || {
  echo "TRAIN RT calibration cohort mismatch: $RT_CALIBRATION_RECORDS != $EXPECTED_RT_CALIBRATION" >&2
  exit 2
}
[ "$VALIDATION_CONSUMED" = "YES" ] || {
  echo "completed export must record historical_validation_consumed=YES" >&2
  exit 2
}
[ "$TEST_CONSUMED" = "NO" ] || {
  echo "historical TEST must remain unopened; observed: $TEST_CONSUMED" >&2
  exit 2
}

PRECURSOR_ROWS="$(awk 'END { print NR > 0 ? NR - 1 : 0 }' "$PRECURSORS")"
CALIBRATION_ROWS="$(awk 'END { print NR > 0 ? NR - 1 : 0 }' "$RT_CALIBRATION")"
[ "$PRECURSOR_ROWS" = "$EXPECTED_UNION_RECORDS" ] || {
  echo "precursor TSV row mismatch: $PRECURSOR_ROWS != $EXPECTED_UNION_RECORDS" >&2
  exit 2
}
[ "$CALIBRATION_ROWS" = "$EXPECTED_RT_CALIBRATION" ] || {
  echo "RT calibration TSV row mismatch: $CALIBRATION_ROWS != $EXPECTED_RT_CALIBRATION" >&2
  exit 2
}
[ -s "$FRAGMENTS" ] || {
  echo "fragment TSV is empty: $FRAGMENTS" >&2
  exit 2
}

bind_dirs="$({ dirname "$OUT"; dirname "$REPORT_SCRIPT"; } | sort -u | paste -sd, -)"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  bind_dirs="${bind_dirs},${REDEEM_SINGULARITY_BINDS}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-NA}"
echo "comparison_mode=RESUME_FROM_FROZEN_V0520_VALIDATION_EXPORT"
echo "comparison_partition=HISTORICAL_VALIDATION"
echo "comparison_scope=RT_MS2_ONLY"
echo "export_version=$EXPORT_VERSION"
echo "validation_rt_records=$RT_RECORDS"
echo "validation_ms2_records=$MS2_RECORDS"
echo "validation_union_records=$UNION_RECORDS"
echo "rt_train_calibration_records=$RT_CALIBRATION_RECORDS"
echo "train_holdout_consumed=YES"
echo "historical_validation_consumed=YES"
echo "historical_validation_consumed_for_selection=NO"
echo "historical_test_consumed=NO"
echo "apd_sif=$APD_SIF"

"$ENGINE" exec --bind "$bind_dirs" "$APD_SIF" python - <<'PY'
import importlib.metadata
v = importlib.metadata.version("peptdeep")
print("alphapeptdeep_version\t" + v)
assert v == "1.5.1", f"expected peptdeep 1.5.1, observed {v}"
PY
apd_gate_rc=$?
if [ "$apd_gate_rc" -ne 0 ]; then exit "$apd_gate_rc"; fi

{
  printf 'artifact\tsha256\n'
  for f in "$PRECURSORS" "$FRAGMENTS" "$RT_CALIBRATION"; do
    hash="$(sha256sum "$f" | awk '{print $1}')"
    printf '%s\t%s\n' "$(basename "$f")" "$hash"
  done
} > "$FROZEN_HASHES"

echo "frozen_export_hashes=$FROZEN_HASHES"
cat "$FROZEN_HASHES"
echo "historical_validation_export_gate=PASS"

srun "$ENGINE" exec --bind "$bind_dirs" "$APD_SIF" \
  /usr/bin/time -v -o "$OUT/comparison.time.txt" \
  redeem-foundation compare-alphapeptdeep-v0240 \
    --precursors "$PRECURSORS" \
    --fragments "$FRAGMENTS" \
    --rt-train-calibration "$RT_CALIBRATION" \
    --output-dir "$OUT/comparison" \
  2>&1 | tee "$OUT/comparison.log"
comparison_rc=${PIPESTATUS[0]}
if [ "$comparison_rc" -ne 0 ]; then exit "$comparison_rc"; fi

srun "$ENGINE" exec --bind "$bind_dirs" "$APD_SIF" \
  /usr/bin/time -v -o "$OUT/report.time.txt" \
  python "$REPORT_SCRIPT" \
    --comparison-dir "$OUT/comparison" \
    --output-dir "$OUT/report" \
  2>&1 | tee "$OUT/report.log"
report_rc=${PIPESTATUS[0]}
if [ "$report_rc" -ne 0 ]; then exit "$report_rc"; fi

cat "$OUT/comparison/alphapeptdeep_comparison_summary.tsv"
cat "$OUT/report/v0520_apd_headline.tsv"
cat "$OUT/report/v0520_apd_report.md"

echo "train_holdout_consumed=YES"
echo "historical_validation_consumed=YES"
echo "historical_validation_consumed_for_selection=NO"
echo "historical_test_consumed=NO"
echo "v0520_apd_comparison_complete=YES"
echo "v0520_apd_out=$OUT"
