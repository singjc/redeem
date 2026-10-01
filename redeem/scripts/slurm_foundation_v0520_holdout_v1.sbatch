#!/usr/bin/env bash
#SBATCH --job-name=redeem-v052-holdout
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=redeem-v0520-holdout-%j.out

module load singularityce/4.1.0

: "${SIF:?Set SIF to the v0.52 HOLDOUT evaluator SIF}"
: "${RUN_YAML:?Set RUN_YAML to prepared run_v0260.yaml}"
: "${V052_BEST:?Set V052_BEST to the frozen selected v0.52 best checkpoint directory}"
: "${OUT:?Set OUT to a fresh one-time HOLDOUT output directory}"

[ -f "$SIF" ] || { echo "SIF not found: $SIF" >&2; exit 2; }
[ -f "$RUN_YAML" ] || { echo "RUN_YAML not found: $RUN_YAML" >&2; exit 2; }
[ -f "$V052_BEST/model.safetensors" ] || { echo "v0.52 model missing: $V052_BEST" >&2; exit 2; }
[ -f "$V052_BEST/metadata.yaml" ] || { echo "v0.52 metadata missing: $V052_BEST" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "HOLDOUT output must be fresh: $OUT" >&2; exit 2; }

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
if [ "$GPU_CC" != "80" ]; then
  echo "v0.52 HOLDOUT evaluator expects A100 sm80; observed $GPU_NAME sm_$GPU_CC" >&2
  exit 2
fi

mkdir -p "$OUT"
bind_dirs="$({ dirname "$RUN_YAML"; dirname "$V052_BEST"; dirname "$OUT"; } | sort -u | paste -sd, -)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "gpu_name=$GPU_NAME"
echo "cuda_compute_capability=$GPU_CC"
echo "evaluation_partition=TRAIN_HOLDOUT_ONCE"
echo "checkpoint=$V052_BEST"
echo "historical_validation_consumed_before_run=NO"
echo "historical_test_consumed_before_run=NO"

srun singularity exec --nv --bind "$bind_dirs" "$SIF" \
  /usr/bin/time -v -o "$OUT/holdout.time.txt" \
  foundation_evaluate_v0520_holdout \
    "$RUN_YAML" \
    "$V052_BEST" \
    "$OUT/results" \
    32 \
  2>&1 | tee "$OUT/holdout.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi

echo "===== HOLDOUT SUMMARY ====="
cat "$OUT/results/holdout_summary.tsv"

echo "===== HOLDOUT REPORT ====="
cat "$OUT/results/holdout_report.md"

echo "train_holdout_consumed=YES"
echo "train_holdout_consumed_for_selection=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"
echo "v0520_holdout_complete=YES"
echo "v0520_holdout_out=$OUT"
