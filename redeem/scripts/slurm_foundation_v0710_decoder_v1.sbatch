#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=redeem-v0710-%j.out

module load singularityce/4.1.0

: "${SIF:?Set SIF to the v0.71 causal-decoder SIF}"
: "${RUN_V0260:?Set RUN_V0260 to the prepared run_v0260.yaml}"
: "${PARENT_V070:?Set PARENT_V070 to the frozen v0.70 best checkpoint directory}"
: "${OUT:?Set OUT to the v0.71 experiment root}"

MODE="${MODE:-train}"
MAX_EPOCHS="${MAX_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
PATIENCE="${PATIENCE:-2}"
MIN_DELTA="${MIN_DELTA:-0.002}"
SEED="${SEED:-20261071}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

case "$MODE" in
  smoke|train|resume) ;;
  *) echo "MODE must be smoke, train, or resume (got $MODE)" >&2; exit 2 ;;
esac

[ -f "$SIF" ] || { echo "Missing v0.71 SIF: $SIF" >&2; exit 2; }
[ -f "$RUN_V0260" ] || { echo "Missing prepared run YAML: $RUN_V0260" >&2; exit 2; }
[ -f "$PARENT_V070/model.safetensors" ] || {
  echo "Missing v0.70 model.safetensors: $PARENT_V070" >&2
  exit 2
}
[ -f "$PARENT_V070/metadata.yaml" ] || {
  echo "Missing v0.70 metadata.yaml: $PARENT_V070" >&2
  exit 2
}

MODEL_OUT="$OUT/model"
if [ "$MODE" = "resume" ]; then
  [ -f "$MODEL_OUT/latest/model.safetensors" ] || {
    echo "v0.71 resume requires latest/model.safetensors" >&2
    exit 2
  }
  [ -f "$MODEL_OUT/latest/optimizer.safetensors" ] || {
    echo "v0.71 resume requires latest/optimizer.safetensors" >&2
    exit 2
  }
  [ -f "$MODEL_OUT/best/model.safetensors" ] || {
    echo "v0.71 resume requires best/model.safetensors" >&2
    exit 2
  }
else
  if [ -e "$OUT" ]; then
    echo "v0.71 $MODE output must be fresh: $OUT" >&2
    exit 2
  fi
  mkdir -p "$OUT"
fi

ENGINE="$(command -v singularity)"
[ -n "$ENGINE" ] || { echo "singularity is unavailable after module load" >&2; exit 127; }

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"

echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "gpu_name=$GPU_NAME"
echo "cuda_compute_capability=$GPU_CC"
echo "mode=$MODE"
echo "sif=$SIF"
echo "run_v0260=$RUN_V0260"
echo "parent_v070=$PARENT_V070"
echo "experiment=$OUT"
echo "training_scope=TRAIN"
echo "selection_scope=DEV"
echo "rt_conditioning=NO"
echo "ccs_conditioning=NO"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"

if [ "$GPU_CC" != "80" ]; then
  echo "v0.71 image is built for A100 compute capability 8.0; observed $GPU_NAME sm_$GPU_CC" >&2
  exit 2
fi

bind_dirs="$({
  dirname "$RUN_V0260"
  dirname "$PARENT_V070"
  dirname "$OUT"
} | sort -u | paste -sd, -)"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  bind_dirs="${bind_dirs},${REDEEM_SINGULARITY_BINDS}"
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

RUN_TAG="$MODE"
if [ "$MODE" = "resume" ]; then
  RUN_TAG="resume_${SLURM_JOB_ID:-manual}"
fi

GPU_USAGE_CSV="$OUT/gpu_usage_${RUN_TAG}.csv"
GPU_USAGE_SUMMARY="$OUT/gpu_usage_${RUN_TAG}_summary.tsv"
printf '%s\n' 'timestamp,index,gpu_util_percent,gpu_memory_util_percent,gpu_memory_used_mib,gpu_memory_total_mib' > "$GPU_USAGE_CSV"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits -l 5 >> "$GPU_USAGE_CSV" 2> "$OUT/gpu_usage_${RUN_TAG}.stderr" &
GPU_MONITOR_PID=$!

cleanup() {
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
  ' "$GPU_USAGE_CSV" > "$GPU_USAGE_SUMMARY" || true
}
trap cleanup EXIT

LOG="$OUT/${RUN_TAG}.log"
TIME_FILE="$OUT/${RUN_TAG}.time.txt"

srun "$ENGINE" exec --nv --bind "$bind_dirs" "$SIF" \
  /usr/bin/time -v -o "$TIME_FILE" \
  foundation_train_spectrum_causal_v0710 \
    "$RUN_V0260" \
    "$MODEL_OUT" \
    "$PARENT_V070" \
    "$MODE" \
    "$MAX_EPOCHS" \
    "$BATCH_SIZE" \
    "$STEPS_PER_EPOCH" \
    "$PATIENCE" \
    "$MIN_DELTA" \
    "$SEED" \
    "$LEARNING_RATE" \
  2>&1 | tee "$LOG"

run_rc=${PIPESTATUS[0]}
exit "$run_rc"
