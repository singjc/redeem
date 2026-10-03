#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0711
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0711-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0711-%j.err

ROOT=/nfs/research/juan/DIA/singj/foundation_model
: "${SIF:?SIF is required}"
: "${RUN_V0260:?RUN_V0260 is required}"
: "${PARENT_V070:?PARENT_V070 is required}"
: "${PARENT_V0200:?PARENT_V0200 is required}"
: "${OUT:?OUT is required}"
MODE="${MODE:-smoke}"

if [ "$MODE" != "smoke" ] && [ "$MODE" != "probe" ]; then
  echo "MODE must be smoke or probe" >&2
  exit 2
fi
for path in "$SIF" "$RUN_V0260" "$PARENT_V070/model.safetensors" "$PARENT_V070/metadata.yaml" "$PARENT_V0200/model.safetensors" "$PARENT_V0200/metadata.yaml"; do
  [ -e "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done
if [ -e "$OUT" ]; then
  echo "v0.71.1 output directory must be fresh: $OUT" >&2
  exit 2
fi

AUX_OUT="${OUT}.launcher-${SLURM_JOB_ID:-manual}"
if [ -e "$AUX_OUT" ]; then
  echo "v0.71.1 launcher sidecar must be fresh: $AUX_OUT" >&2
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
printf 'parent_v0200=%s\n' "$PARENT_V0200"
printf 'experiment=%s\n' "$OUT"
printf 'training_scope=TRAIN\nselection_scope=DEV\n'
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
    foundation_train_spectrum_causal_warmstart_v0711 \
      "$RUN_V0260" \
      "$OUT" \
      "$PARENT_V070" \
      "$PARENT_V0200" \
      "$MODE" \
  2>&1 | tee "$LOG"
run_rc=${PIPESTATUS[0]}

cleanup
trap - EXIT

if [ -s "$GPU_USAGE_CSV" ]; then
  awk -F',' '
    { gsub(/^ +| +$/, "", $3); gsub(/^ +| +$/, "", $4); gsub(/^ +| +$/, "", $5); gsub(/^ +| +$/, "", $6);
      n++; gu+=$3; if($3>gum)gum=$3; mu+=$4; if($4>mum)mum=$4; mem+=$5; if($5>memm)memm=$5; total=$6 }
    END { print "samples\t" n; if(n>0){printf "mean_gpu_util_percent\t%.3f\nmax_gpu_util_percent\t%.3f\nmean_gpu_memory_controller_util_percent\t%.3f\nmax_gpu_memory_controller_util_percent\t%.3f\nmean_gpu_memory_used_mib\t%.3f\nmax_gpu_memory_used_mib\t%.3f\nmean_gpu_vram_used_percent\t%.3f\nmax_gpu_vram_used_percent\t%.3f\n", gu/n,gum,mu/n,mum,mem/n,memm,100*mem/n/total,100*memm/total} }
  ' "$GPU_USAGE_CSV" > "$GPU_USAGE_SUMMARY"
fi

if [ -d "$OUT" ]; then
  find "$AUX_OUT" -mindepth 1 -maxdepth 1 -exec mv -t "$OUT" -- {} +
  rmdir "$AUX_OUT"
else
  echo "v0.71.1 trainer did not create output root; launcher logs retained at $AUX_OUT" >&2
fi

exit "$run_rc"
