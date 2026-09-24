#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0500
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=redeem-v0500-%j.out

module load singularityce/4.1.0

if [ -z "${SIF:-}" ]; then
  echo "Set SIF to the v0.50 joint-training SIF" >&2
  exit 2
fi
if [ -z "${RUN_V0260:-}" ]; then
  echo "Set RUN_V0260 to the existing prepared run_v0260.yaml" >&2
  exit 2
fi
if [ -z "${PARENT:-}" ]; then
  echo "Set PARENT to the frozen v0.35 final checkpoint directory" >&2
  exit 2
fi
if [ -z "${OUT:-}" ]; then
  echo "Set OUT to the v0.50 experiment directory" >&2
  exit 2
fi

MODE="${MODE:-smoke}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PATIENCE="${PATIENCE:-3}"
MIN_DELTA="${MIN_DELTA:-0.002}"
SEED="${SEED:-20261050}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

case "$MODE" in
  smoke|train|resume) ;;
  *) echo "MODE must be smoke, train, or resume (got $MODE)" >&2; exit 2 ;;
esac

if [ ! -f "$SIF" ]; then
  echo "Missing v0.50 SIF: $SIF" >&2
  exit 2
fi
if [ ! -f "$RUN_V0260" ]; then
  echo "Missing prepared v0.26 run YAML: $RUN_V0260" >&2
  exit 2
fi
if [ ! -f "$PARENT/model.safetensors" ] || [ ! -f "$PARENT/metadata.yaml" ]; then
  echo "Frozen v0.35 parent checkpoint is incomplete: $PARENT" >&2
  exit 2
fi

MODEL_OUT="$OUT/model"
if [ "$MODE" = "resume" ]; then
  if [ ! -f "$MODEL_OUT/latest/model.safetensors" ] || \
     [ ! -f "$MODEL_OUT/latest/optimizer.safetensors" ] || \
     [ ! -f "$MODEL_OUT/latest/metadata.yaml" ] || \
     [ ! -f "$MODEL_OUT/best/model.safetensors" ] || \
     [ ! -f "$MODEL_OUT/best/metadata.yaml" ]; then
    echo "v0.50 RESUME requires latest + best checkpoints under $MODEL_OUT" >&2
    exit 2
  fi
else
  if [ -e "$OUT" ]; then
    echo "v0.50 $MODE output must be fresh: $OUT" >&2
    exit 2
  fi
  mkdir -p "$OUT"
fi

if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required for the Codon v0.50 workflow" >&2
  exit 127
fi
ENGINE="$(command -v singularity)"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "gpu_name=$GPU_NAME"
echo "cuda_compute_capability=$GPU_CC"
echo "mode=$MODE"
echo "sif=$SIF"
echo "run_v0260=$RUN_V0260"
echo "parent_v0350=$PARENT"
echo "experiment=$OUT"
echo "batch_size=$BATCH_SIZE"
echo "max_epochs=$MAX_EPOCHS"
echo "learning_rate=$LEARNING_RATE"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"

if [ "$GPU_CC" != "80" ]; then
  echo "v0.50 image is built for A100 compute capability 8.0; observed $GPU_NAME sm_$GPU_CC" >&2
  exit 2
fi

bind_dirs="$({
  dirname "$RUN_V0260"
  dirname "$PARENT"
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
GPU_USAGE_CSV="$OUT/gpu_usage_${RUN_TAG}.csv"
GPU_USAGE_SUMMARY="$OUT/gpu_usage_${RUN_TAG}_summary.tsv"
printf '%s\n' 'timestamp,index,gpu_util_percent,gpu_memory_util_percent,gpu_memory_used_mib,gpu_memory_total_mib' > "$GPU_USAGE_CSV"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits -l 2 >> "$GPU_USAGE_CSV" 2> "$OUT/gpu_usage_${RUN_TAG}.stderr" &
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
  foundation_train_multimodal_v0500 \
    "$RUN_V0260" \
    "$MODEL_OUT" \
    "$PARENT" \
    "$MAX_EPOCHS" \
    "$BATCH_SIZE" \
    "$PATIENCE" \
    "$MIN_DELTA" \
    "$SEED" \
    "$LEARNING_RATE" \
    "$MODE" \
  2>&1 | tee "$LOG"
run_rc=${PIPESTATUS[0]}
exit "$run_rc"
