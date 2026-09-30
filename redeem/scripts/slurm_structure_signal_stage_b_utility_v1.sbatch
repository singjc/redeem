#!/usr/bin/env bash
#SBATCH --job-name=redeem-structure-utility-v1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=redeem-structure-utility-v1-%j.out

module load singularityce/4.1.0

for name in SIF RUN_V0260 PARENT_V0350 V038_MODEL_ROOT MANIFEST STRUCTURE_FEATURES STAGE_B_SUMMARY EXPECTED_MANIFEST_FINGERPRINT EXPECTED_STRUCTURE_FINGERPRINT OUT; do
  eval "value=\${$name:-}"
  if [ -z "$value" ]; then
    echo "Set $name for the Stage B structure utility gate" >&2
    exit 2
  fi
done

for file in "$SIF" "$RUN_V0260" "$MANIFEST" "$STRUCTURE_FEATURES" "$STAGE_B_SUMMARY"; do
  if [ ! -f "$file" ]; then
    echo "Missing required file: $file" >&2
    exit 2
  fi
done

for dir in "$PARENT_V0350" "$V038_MODEL_ROOT/best"; do
  if [ ! -d "$dir" ]; then
    echo "Missing required directory: $dir" >&2
    exit 2
  fi
done

for file in \
  "$PARENT_V0350/metadata.yaml" \
  "$V038_MODEL_ROOT/best/model.safetensors" \
  "$V038_MODEL_ROOT/best/metadata.yaml"; do
  if [ ! -f "$file" ]; then
    echo "Missing required checkpoint file: $file" >&2
    exit 2
  fi
done

if [ -e "$OUT" ]; then
  echo "Utility OUT must be fresh: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"

if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required" >&2
  exit 127
fi

ENGINE="$(command -v singularity)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"

if [ "$GPU_CC" != "80" ]; then
  echo "Utility image is built for A100 compute capability 8.0; observed $GPU_NAME sm_$GPU_CC" >&2
  exit 2
fi

PREDICTIONS="$OUT/v038_train_selected_predictions.tsv"
ANALYSIS_OUT="$OUT/analysis"

bind_dirs="$({
  dirname "$RUN_V0260"
  dirname "$PARENT_V0350"
  dirname "$V038_MODEL_ROOT"
  dirname "$MANIFEST"
  dirname "$STRUCTURE_FEATURES"
  dirname "$STAGE_B_SUMMARY"
  dirname "$OUT"
} | sort -u | paste -sd, -)"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  bind_dirs="${bind_dirs},${REDEEM_SINGULARITY_BINDS}"
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "gpu_name=$GPU_NAME"
echo "cuda_compute_capability=$GPU_CC"
echo "sif=$SIF"
echo "run_v0260=$RUN_V0260"
echo "parent_v0350=$PARENT_V0350"
echo "v038_model_root=$V038_MODEL_ROOT"
echo "manifest=$MANIFEST"
echo "structure_features=$STRUCTURE_FEATURES"
echo "stage_b_summary=$STAGE_B_SUMMARY"
echo "expected_manifest_fingerprint=$EXPECTED_MANIFEST_FINGERPRINT"
echo "expected_structure_fingerprint=$EXPECTED_STRUCTURE_FINGERPRINT"
echo "out=$OUT"
echo "partition_scope=TRAIN_frozen_stage_b_ccs_overlap_only"
echo "dev_labels_used=NO"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"

echo "===== container gates ====="
srun "$ENGINE" exec --nv --bind "$bind_dirs" "$SIF" \
  python -c 'import sklearn; print("scikit_learn_version=" + sklearn.__version__)'

srun "$ENGINE" exec --nv --bind "$bind_dirs" "$SIF" \
  structure_signal_stage_b_utility_v1.py --self-test

printf '%s\n' 'timestamp,index,gpu_util_percent,gpu_memory_util_percent,gpu_memory_used_mib,gpu_memory_total_mib' > "$OUT/gpu_usage.csv"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits -l 2 >> "$OUT/gpu_usage.csv" 2> "$OUT/gpu_usage.stderr" &
GPU_MONITOR_PID=$!

finish_monitor() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
  awk -F',' '
    NR==1 { next }
    {
      for (i=1;i<=NF;i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
      gpu=$3+0; used=$5+0; total=$6+0
      n++; gu+=gpu; mm+=used
      if (gpu>gmax) gmax=gpu
      if (used>mmmax) mmmax=used
      if (total>0) { pct=100.0*used/total; mp+=pct; if (pct>mpmax) mpmax=pct }
    }
    END {
      printf "samples\t%d\n", n
      if (n) {
        printf "mean_gpu_util_percent\t%.3f\nmax_gpu_util_percent\t%.3f\n", gu/n, gmax
        printf "mean_gpu_memory_used_mib\t%.3f\nmax_gpu_memory_used_mib\t%.3f\n", mm/n, mmmax
        printf "mean_gpu_vram_used_percent\t%.3f\nmax_gpu_vram_used_percent\t%.3f\n", mp/n, mpmax
      }
    }
  ' "$OUT/gpu_usage.csv" > "$OUT/gpu_usage_summary.tsv" || true
}
trap finish_monitor EXIT

echo "===== v0.38 selected TRAIN export ====="
export SINGULARITYENV_REDEEM_CCS_AUDIT_EXPORT_TSV="$PREDICTIONS"
export SINGULARITYENV_REDEEM_CCS_AUDIT_SELECTED_IDENTITIES_TSV="$MANIFEST"
srun "$ENGINE" exec --nv --bind "$bind_dirs" "$SIF" \
  foundation_train_multimodal_v0380 \
    "$RUN_V0260" \
    "$V038_MODEL_ROOT" \
    "$PARENT_V0350" \
    10 \
    96 \
    3 \
    0.002 \
    20261038 \
    3e-5 \
    audit-export-train-selected \
  2>&1 | tee "$OUT/v038_train_export.log"
v038_rc=${PIPESTATUS[0]}
if [ "$v038_rc" -ne 0 ]; then
  echo "v0.38 selected TRAIN export failed with rc=$v038_rc" >&2
  exit "$v038_rc"
fi

manifest_identities=$(awk -F'\t' '$1=="v0380_train_selected_manifest_identities" {print $2}' "$OUT/v038_train_export.log" | tail -n1)
exported_identities=$(awk -F'\t' '$1=="v0380_train_selected_export_identities" {print $2}' "$OUT/v038_train_export.log" | tail -n1)
exported_records=$(awk -F'\t' '$1=="v0380_train_selected_export_records" {print $2}' "$OUT/v038_train_export.log" | tail -n1)

echo "manifest_identities=$manifest_identities"
echo "ccs_labeled_identities=$exported_identities"
echo "ccs_labeled_records=$exported_records"
if [ "$manifest_identities" -ne 8000 ]; then
  echo "Expected exactly 8000 frozen Stage B identities" >&2
  exit 2
fi

echo "===== grouped TRAIN residual utility analysis ====="
srun "$ENGINE" exec --nv --bind "$bind_dirs" "$SIF" \
  structure_signal_stage_b_utility_v1.py \
    --manifest "$MANIFEST" \
    --structure-features "$STRUCTURE_FEATURES" \
    --stage-b-summary "$STAGE_B_SUMMARY" \
    --v038-train-predictions "$PREDICTIONS" \
    --expected-manifest-fingerprint "$EXPECTED_MANIFEST_FINGERPRINT" \
    --expected-structure-fingerprint "$EXPECTED_STRUCTURE_FINGERPRINT" \
    --out "$ANALYSIS_OUT" \
  2>&1 | tee "$OUT/utility.log"
utility_rc=${PIPESTATUS[0]}
if [ "$utility_rc" -ne 0 ]; then
  echo "Structure utility analysis failed with rc=$utility_rc" >&2
  exit "$utility_rc"
fi

utility_gate=$(awk -F'\t' '$1=="structure_utility_gate" {print $2}' "$ANALYSIS_OUT/utility_summary.tsv" | tail -n1)
echo "structure_utility_gate=$utility_gate"
echo "dev_labels_used=NO"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"
echo "utility_complete=YES"
echo "analysis_out=$ANALYSIS_OUT"
