#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0731
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0731-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0731-%j.err

module load singularityce/4.1.0

ROOT=/nfs/research/juan/DIA/singj/foundation_model
: "${SIF:?SIF is required}"
: "${RUN_V0260:?RUN_V0260 is required}"
: "${PARENT_V070:?PARENT_V070 is required}"
: "${OUT:?OUT is required}"
MODE="${MODE:-smoke}"

if [ "$MODE" != "smoke" ] && [ "$MODE" != "audit" ]; then
  echo "MODE must be smoke or audit" >&2
  exit 2
fi
for path in "$SIF" "$RUN_V0260" "$PARENT_V070/model.safetensors" "$PARENT_V070/metadata.yaml"; do
  [ -e "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done
if [ -e "$OUT" ]; then
  echo "v0.73.1 output directory must be fresh: $OUT" >&2
  exit 2
fi

AUX_OUT="${OUT}.launcher-${SLURM_JOB_ID:-manual}"
if [ -e "$AUX_OUT" ]; then
  echo "v0.73.1 launcher sidecar must be fresh: $AUX_OUT" >&2
  exit 2
fi
mkdir -p "$AUX_OUT"

RUN_TAG="$MODE"
GPU_USAGE_CSV="$AUX_OUT/gpu_usage_${RUN_TAG}.csv"
GPU_USAGE_SUMMARY="$AUX_OUT/gpu_usage_${RUN_TAG}_summary.tsv"
LOG="$AUX_OUT/${RUN_TAG}.log"
TIME_FILE="$AUX_OUT/${RUN_TAG}.time.txt"

printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf 'host=%s\n' "$(hostname)"
printf 'mode=%s\n' "$MODE"
printf 'sif=%s\n' "$SIF"
printf 'run_v0260=%s\n' "$RUN_V0260"
printf 'parent_v070=%s\n' "$PARENT_V070"
printf 'experiment=%s\n' "$OUT"
printf 'training_scope=NONE\nselection_scope=DEV_AUDIT\n'
printf 'rt_conditioning=NO\nccs_conditioning=NO\n'
printf 'train_holdout_consumed=NO\nhistorical_validation_consumed=NO\nhistorical_test_consumed=NO\n'

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits -l 5 > "$GPU_USAGE_CSV" 2> "$AUX_OUT/gpu_usage_${RUN_TAG}.stderr" &
GPU_PID=$!
cleanup() {
  if kill -0 "$GPU_PID" >/dev/null 2>&1; then
    kill "$GPU_PID" >/dev/null 2>&1 || true
    wait "$GPU_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

/usr/bin/time -v -o "$TIME_FILE" \
  singularity exec --nv \
    -B "$ROOT:$ROOT" \
    "$SIF" \
    foundation_audit_forward_ms2_attribution_v0731 \
      "$RUN_V0260" \
      "$OUT" \
      "$PARENT_V070" \
      "$MODE" \
  2>&1 | tee "$LOG"
run_rc=${PIPESTATUS[0]}

cleanup
trap - EXIT

if [ -s "$GPU_USAGE_CSV" ]; then
  awk -F',' '
    {
      for (i=3; i<=6; i++) gsub(/^ +| +$/, "", $i)
      guv=$3+0; muv=$4+0; memv=$5+0; totalv=$6+0
      pct=(totalv>0 ? 100*memv/totalv : 0)
      n++; gu+=guv; mu+=muv; mem+=memv; pctsum+=pct
      if (n==1 || guv>gumax) gumax=guv
      if (n==1 || muv>mumax) mumax=muv
      if (n==1 || memv>memmax) memmax=memv
      if (n==1 || pct>pctmax) pctmax=pct
    }
    END {
      print "samples\t" n
      if (n>0) {
        printf "mean_gpu_util_percent\t%.3f\nmax_gpu_util_percent\t%.3f\nmean_gpu_memory_controller_util_percent\t%.3f\nmax_gpu_memory_controller_util_percent\t%.3f\nmean_gpu_memory_used_mib\t%.3f\nmax_gpu_memory_used_mib\t%.3f\nmean_gpu_vram_used_percent\t%.3f\nmax_gpu_vram_used_percent\t%.3f\n", gu/n,gumax,mu/n,mumax,mem/n,memmax,pctsum/n,pctmax
      }
    }
  ' "$GPU_USAGE_CSV" > "$GPU_USAGE_SUMMARY"
fi

if [ -d "$OUT" ]; then
  find "$AUX_OUT" -mindepth 1 -maxdepth 1 -exec mv -t "$OUT" -- {} +
  rmdir "$AUX_OUT"
else
  echo "v0.73.1 audit did not create output root; launcher logs retained at $AUX_OUT" >&2
fi

exit "$run_rc"
