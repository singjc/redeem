#!/usr/bin/env bash
#SBATCH --job-name=redeem-apd-v0520
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=redeem-apd-v0520-%j.out

module load singularityce/4.1.0

: "${V0520_SIF:?Set V0520_SIF to the SHA-qualified v0.52 APD exporter SIF}"
: "${APD_SIF:?Set APD_SIF to the existing AlphaPeptDeep 1.5.1 research SIF}"
: "${RUN_YAML:?Set RUN_YAML to the frozen prepared run YAML}"
: "${V052_BEST:?Set V052_BEST to the frozen selected v0.52 best checkpoint}"
: "${OUT:?Set OUT to a fresh historical VALIDATION comparison directory}"
: "${REPORT_SCRIPT:?Set REPORT_SCRIPT to render_foundation_alphapeptdeep_report_v0520.py}"

EXPECTED_RT_RECORDS="${EXPECTED_RT_RECORDS:-302034}"
EXPECTED_MS2_RECORDS="${EXPECTED_MS2_RECORDS:-255483}"
EXPECTED_UNION_RECORDS="${EXPECTED_UNION_RECORDS:-302034}"
EXPECTED_RT_CALIBRATION="${EXPECTED_RT_CALIBRATION:-256}"

[ -f "$V0520_SIF" ] || { echo "v0.52 exporter SIF not found: $V0520_SIF" >&2; exit 2; }
[ -f "$APD_SIF" ] || { echo "AlphaPeptDeep SIF not found: $APD_SIF" >&2; exit 2; }
[ -f "$RUN_YAML" ] || { echo "RUN_YAML not found: $RUN_YAML" >&2; exit 2; }
[ -f "$V052_BEST/model.safetensors" ] || { echo "v0.52 model missing: $V052_BEST" >&2; exit 2; }
[ -f "$V052_BEST/metadata.yaml" ] || { echo "v0.52 metadata missing: $V052_BEST" >&2; exit 2; }
[ -f "$REPORT_SCRIPT" ] || { echo "v0.52 report script not found: $REPORT_SCRIPT" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "comparison output must be fresh: $OUT" >&2; exit 2; }

ENGINE="$(command -v singularity)"
[ -n "$ENGINE" ] || { echo "singularity not available after module load" >&2; exit 127; }

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
if [ "$GPU_CC" != "80" ]; then
  echo "v0.52 historical VALIDATION exporter expects A100 sm80; observed $GPU_NAME sm_$GPU_CC" >&2
  exit 2
fi

mkdir -p "$OUT/export" "$OUT/comparison"
bind_dirs="$({ dirname "$RUN_YAML"; dirname "$V052_BEST"; dirname "$OUT"; dirname "$REPORT_SCRIPT"; } | sort -u | paste -sd, -)"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  bind_dirs="${bind_dirs},${REDEEM_SINGULARITY_BINDS}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export REDEEM_COMPARISON_DEVICE=cuda

echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "gpu_name=$GPU_NAME"
echo "cuda_compute_capability=$GPU_CC"
echo "comparison_partition=HISTORICAL_VALIDATION"
echo "comparison_scope=RT_MS2_ONLY"
echo "rt_calibration_partition=TRAIN"
echo "train_holdout_consumed=YES"
echo "historical_validation_consumed_before_run=NO"
echo "historical_test_consumed=NO"
echo "v0520_sif=$V0520_SIF"
echo "apd_sif=$APD_SIF"

"$ENGINE" exec --bind "$bind_dirs" "$APD_SIF" python - <<'PY'
import importlib.metadata
v = importlib.metadata.version("peptdeep")
print("alphapeptdeep_version\t" + v)
assert v == "1.5.1", f"expected peptdeep 1.5.1, observed {v}"
PY

srun "$ENGINE" exec --nv --bind "$bind_dirs" "$V0520_SIF" \
  /usr/bin/time -v -o "$OUT/export.time.txt" \
  foundation_export_alphapeptdeep_comparison_v0520 \
    "$RUN_YAML" "$V052_BEST" "$OUT/export" \
  2>&1 | tee "$OUT/export.log"
export_rc=${PIPESTATUS[0]}
if [ "$export_rc" -ne 0 ]; then exit "$export_rc"; fi

read_metric() {
  awk -F $'\t' -v key="$1" '$1 == key {print $2; exit}' "$OUT/export.log"
}
RT_RECORDS="$(read_metric validation_rt_records)"
MS2_RECORDS="$(read_metric validation_ms2_records)"
UNION_RECORDS="$(read_metric validation_union_records)"
RT_CALIBRATION="$(read_metric rt_train_calibration_records)"

[ "$RT_RECORDS" = "$EXPECTED_RT_RECORDS" ] || { echo "historical VALIDATION RT cohort mismatch: $RT_RECORDS != $EXPECTED_RT_RECORDS" >&2; exit 2; }
[ "$MS2_RECORDS" = "$EXPECTED_MS2_RECORDS" ] || { echo "historical VALIDATION MS2 cohort mismatch: $MS2_RECORDS != $EXPECTED_MS2_RECORDS" >&2; exit 2; }
[ "$UNION_RECORDS" = "$EXPECTED_UNION_RECORDS" ] || { echo "historical VALIDATION union cohort mismatch: $UNION_RECORDS != $EXPECTED_UNION_RECORDS" >&2; exit 2; }
[ "$RT_CALIBRATION" = "$EXPECTED_RT_CALIBRATION" ] || { echo "TRAIN RT calibration cohort mismatch: $RT_CALIBRATION != $EXPECTED_RT_CALIBRATION" >&2; exit 2; }

echo "historical_validation_cohort_count_gate=PASS"

srun "$ENGINE" exec --bind "$bind_dirs" "$APD_SIF" \
  /usr/bin/time -v -o "$OUT/comparison.time.txt" \
  redeem-foundation compare-alphapeptdeep-v0240 \
    --precursors "$OUT/export/foundation_comparison_precursors.tsv" \
    --fragments "$OUT/export/foundation_comparison_fragments.tsv" \
    --rt-train-calibration "$OUT/export/foundation_comparison_rt_train_calibration.tsv" \
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
