#!/usr/bin/env bash
#SBATCH --job-name=redeem-ccs-audit-v1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=redeem-ccs-audit-v1-%j.out

module load singularityce/4.1.0

for name in SIF RUN_V0260 PARENT_V0350 V038_MODEL_ROOT V060_MODEL_ROOT AUDIT_SCRIPT OUT; do
  eval "value=\${$name:-}"
  if [ -z "$value" ]; then
    echo "Set $name for the CCS learnability audit" >&2
    exit 2
  fi
done

for file in "$SIF" "$RUN_V0260" "$AUDIT_SCRIPT"; do
  if [ ! -f "$file" ]; then
    echo "Missing required file: $file" >&2
    exit 2
  fi
done

for dir in "$PARENT_V0350" "$V038_MODEL_ROOT/best" "$V060_MODEL_ROOT/best"; do
  if [ ! -d "$dir" ]; then
    echo "Missing required directory: $dir" >&2
    exit 2
  fi
done

for file in \
  "$PARENT_V0350/metadata.yaml" \
  "$V038_MODEL_ROOT/best/model.safetensors" \
  "$V038_MODEL_ROOT/best/metadata.yaml" \
  "$V060_MODEL_ROOT/best/model.safetensors" \
  "$V060_MODEL_ROOT/best/metadata.yaml"; do
  if [ ! -f "$file" ]; then
    echo "Missing required checkpoint file: $file" >&2
    exit 2
  fi
done

if [ -e "$OUT" ]; then
  echo "Audit OUT must be fresh: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"

if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required" >&2
  exit 127
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for the deterministic audit analyzer" >&2
  exit 127
fi

ENGINE="$(command -v singularity)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"

if [ "$GPU_CC" != "80" ]; then
  echo "Audit image is built for A100 compute capability 8.0; observed $GPU_NAME sm_$GPU_CC" >&2
  exit 2
fi

V038_TSV="$OUT/v038_dev_predictions.tsv"
V060_TSV="$OUT/v060_dev_predictions.tsv"
ANALYSIS_OUT="$OUT/analysis"

bind_dirs="$({
  dirname "$RUN_V0260"
  dirname "$PARENT_V0350"
  dirname "$V038_MODEL_ROOT"
  dirname "$V060_MODEL_ROOT"
  dirname "$OUT"
} | sort -u | paste -sd, -)"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  bind_dirs="${bind_dirs},${REDEEM_SINGULARITY_BINDS}"
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
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
echo "v060_model_root=$V060_MODEL_ROOT"
echo "audit_script=$AUDIT_SCRIPT"
echo "out=$OUT"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"

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
      gpu=$3+0; memutil=$4+0; used=$5+0; total=$6+0
      n++; gu+=gpu; mu+=memutil; mm+=used
      if (gpu>gmax) gmax=gpu
      if (memutil>mumax) mumax=memutil
      if (used>mmmax) mmmax=used
      if (total>0) { pct=100.0*used/total; mp+=pct; if (pct>mpmax) mpmax=pct }
    }
    END {
      printf "samples\t%d\n", n
      if (n) {
        printf "mean_gpu_util_percent\t%.3f\nmax_gpu_util_percent\t%.3f\n", gu/n, gmax
        printf "mean_gpu_memory_controller_util_percent\t%.3f\nmax_gpu_memory_controller_util_percent\t%.3f\n", mu/n, mumax
        printf "mean_gpu_memory_used_mib\t%.3f\nmax_gpu_memory_used_mib\t%.3f\n", mm/n, mmmax
        printf "mean_gpu_vram_used_percent\t%.3f\nmax_gpu_vram_used_percent\t%.3f\n", mp/n, mpmax
      }
    }
  ' "$OUT/gpu_usage.csv" > "$OUT/gpu_usage_summary.tsv" || true
}
trap finish_monitor EXIT

echo "===== v0.38 full DEV export ====="
export SINGULARITYENV_REDEEM_CCS_AUDIT_EXPORT_TSV="$V038_TSV"
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
    audit-export \
  2>&1 | tee "$OUT/v038_export.log"
v038_rc=${PIPESTATUS[0]}
if [ "$v038_rc" -ne 0 ]; then
  echo "v0.38 export failed with rc=$v038_rc" >&2
  exit "$v038_rc"
fi
v038_mae=$(awk -F'\t' '$1=="v0380_audit_export_raw_ccs_mae" {print $2}' "$OUT/v038_export.log" | tail -n1)
python3 - "$v038_mae" <<'PY'
import math, sys
x = float(sys.argv[1])
expected = 8.80611929
if not math.isfinite(x) or abs(x - expected) > 1e-4:
    raise SystemExit(f"v0.38 audit export MAE mismatch: observed={x:.8f} expected={expected:.8f}")
print(f"v038_reference_reproduction=PASS observed={x:.8f}")
PY

echo "===== v0.60 full DEV export ====="
export SINGULARITYENV_REDEEM_CCS_AUDIT_EXPORT_TSV="$V060_TSV"
srun "$ENGINE" exec --nv --bind "$bind_dirs" "$SIF" \
  foundation_train_multimodal_v0600 \
    "$RUN_V0260" \
    "$V060_MODEL_ROOT" \
    "$PARENT_V0350" \
    6 \
    4 \
    8 \
    32 \
    3 \
    0.002 \
    20261060 \
    2e-5 \
    audit-export \
  2>&1 | tee "$OUT/v060_export.log"
v060_rc=${PIPESTATUS[0]}
if [ "$v060_rc" -ne 0 ]; then
  echo "v0.60 export failed with rc=$v060_rc" >&2
  exit "$v060_rc"
fi
v060_mae=$(awk -F'\t' '$1=="v0600_audit_export_raw_ccs_mae" {print $2}' "$OUT/v060_export.log" | tail -n1)
python3 - "$v060_mae" <<'PY'
import math, sys
x = float(sys.argv[1])
expected = 8.98711661
if not math.isfinite(x) or abs(x - expected) > 1e-4:
    raise SystemExit(f"v0.60 audit export MAE mismatch: observed={x:.8f} expected={expected:.8f}")
print(f"v060_reference_reproduction=PASS observed={x:.8f}")
PY

v038_rows=$(($(wc -l < "$V038_TSV") - 1))
v060_rows=$(($(wc -l < "$V060_TSV") - 1))
echo "v038_export_rows=$v038_rows"
echo "v060_export_rows=$v060_rows"
if [ "$v038_rows" -ne "$v060_rows" ]; then
  echo "Prediction exports have different row counts" >&2
  exit 2
fi
if [ "$v038_rows" -ne 50906 ]; then
  echo "Expected 50906 full raw DEV CCS rows, observed $v038_rows" >&2
  exit 2
fi

python3 "$AUDIT_SCRIPT" \
  --v038-predictions "$V038_TSV" \
  --v060-predictions "$V060_TSV" \
  --out "$ANALYSIS_OUT" \
  2>&1 | tee "$OUT/audit.log"
audit_rc=${PIPESTATUS[0]}
if [ "$audit_rc" -ne 0 ]; then
  echo "CCS learnability analysis failed with rc=$audit_rc" >&2
  exit "$audit_rc"
fi

echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"
echo "audit_complete=YES"
echo "analysis_out=$ANALYSIS_OUT"
