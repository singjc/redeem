#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0500-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=redeem-v0500-smoke-%j.out

module load singularityce/4.1.0

if [ -z "${SIF:-}" ]; then
  echo "Set SIF to the v0.50 smoke image" >&2
  exit 2
fi
if [ -z "${OUT:-}" ]; then
  echo "Set OUT to a fresh v0.50 smoke output directory" >&2
  exit 2
fi
if [ ! -f "$SIF" ]; then
  echo "Missing v0.50 smoke SIF: $SIF" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "v0.50 smoke output must be fresh: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"

if [ -n "${REDEEM_SINGULARITY_BIN:-}" ]; then
  ENGINE="$REDEEM_SINGULARITY_BIN"
elif command -v singularity >/dev/null 2>&1; then
  ENGINE="$(command -v singularity)"
else
  echo "singularity is not available" >&2
  exit 127
fi

SMOKE_MODE="${SMOKE_MODE:-a100-full}"
PROFILE_BATCH_SIZE="${PROFILE_BATCH_SIZE:-32}"
PROFILE_STEPS="${PROFILE_STEPS:-12}"
case "$SMOKE_MODE" in
  a100)
    EXPECTED_BLOCKS=2
    ;;
  a100-full)
    EXPECTED_BLOCKS=8
    ;;
  *)
    echo "Codon v0.50 smoke mode must be a100 or a100-full; observed $SMOKE_MODE" >&2
    exit 2
    ;;
esac

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "gpu_name=$GPU_NAME"
echo "cuda_compute_capability=$GPU_CC"
echo "sif=$SIF"
echo "out=$OUT"
echo "v0500_smoke_mode=$SMOKE_MODE"
echo "v0500_smoke_blocks=$EXPECTED_BLOCKS"
echo "profile_batch_size=$PROFILE_BATCH_SIZE"
echo "profile_steps=$PROFILE_STEPS"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"

if [ "$GPU_CC" != "80" ]; then
  echo "v0.50 smoke image is built for A100 compute capability 8.0; observed $GPU_NAME sm_$GPU_CC" >&2
  exit 2
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

GPU_USAGE_CSV="$OUT/gpu_usage_smoke.csv"
GPU_USAGE_SUMMARY="$OUT/gpu_usage_smoke_summary.tsv"
printf '%s\n' 'timestamp,index,gpu_util_percent,gpu_memory_util_percent,gpu_memory_used_mib,gpu_memory_total_mib' > "$GPU_USAGE_CSV"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits -l 1 >> "$GPU_USAGE_CSV" 2> "$OUT/gpu_usage_smoke.stderr" &
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

LOG="$OUT/smoke.log"
TIME_FILE="$OUT/smoke.time.txt"
srun "$ENGINE" exec --nv --bind "$OUT" "$SIF" \
  /usr/bin/time -v -o "$TIME_FILE" \
  foundation_v0500_smoke "$SMOKE_MODE" "$PROFILE_BATCH_SIZE" "$PROFILE_STEPS" \
  2>&1 | tee "$LOG"
run_rc=${PIPESTATUS[0]}
exit "$run_rc"
