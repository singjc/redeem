#!/usr/bin/env bash
#SBATCH --job-name=redeem-predict-parity
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-predict-parity-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-predict-parity-%j.err

module load singularityce/4.1.0

ROOT=/nfs/research/juan/DIA/singj/foundation_model
: "${SIF:?SIF is required}"
: "${OUT:?OUT is required}"

RUN="$ROOT/experiments/multimodal_v0261_03_openptm_fragmentgeometryfix2/prepared/run_v0260.yaml"
RT_MS2="$ROOT/experiments/multimodal_v0520_02_mobility_aware_pair_representation_full/model/best"
CCS="$ROOT/experiments/multimodal_v0380_01_mobility_native_ccs/model/best"
APD="$ROOT/experiments/alphapeptdeep_comparison_v0520_01_frozen_historical_validation/export"
RT_REF="$APD/foundation_comparison_precursors.tsv"
MS2_REF="$APD/foundation_comparison_fragments.tsv"
CCS_REF="$ROOT/experiments/ccs_learnability_audit_v1_v038_v060/v038_dev_predictions.tsv"
REPORT="$OUT/checkpoint_parity.tsv"
LOG="$OUT/checkpoint_parity.log"
TIME="$OUT/checkpoint_parity.time.txt"

for required in \
  "$SIF" \
  "$RUN" \
  "$RT_MS2/model.safetensors" \
  "$RT_MS2/metadata.yaml" \
  "$CCS/model.safetensors" \
  "$CCS/metadata.yaml" \
  "$RT_REF" \
  "$MS2_REF" \
  "$CCS_REF"; do
  if [ ! -s "$required" ]; then
    echo "missing checkpoint-parity input: $required" >&2
    exit 2
  fi
done

if [ -e "$OUT" ]; then
  echo "checkpoint-parity output must be fresh: $OUT" >&2
  exit 2
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required" >&2
  exit 127
fi

mkdir -p "$OUT"
BIND="$ROOT:$ROOT"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  BIND="${BIND},${REDEEM_SINGULARITY_BINDS}"
fi

printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf 'host=%s\n' "$(hostname)"
printf 'execution_device=CPU\n'
printf 'rt_ms2_checkpoint=%s\n' "$RT_MS2"
printf 'ccs_checkpoint=%s\n' "$CCS"
printf 'historical_validation_reference_reused=YES\n'
printf 'alphapeptdeep_rerun=NO\n'
printf 'historical_test_compared=NO\n'
printf 'train_holdout_compared=NO\n'

/usr/bin/time -v -o "$TIME" \
  singularity exec --bind "$BIND" "$SIF" \
    foundation_predict_checkpoint_smoke \
      "$RUN" \
      "$RT_MS2" \
      "$CCS" \
      "$RT_REF" \
      "$MS2_REF" \
      "$CCS_REF" \
      "$REPORT" \
      32 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi

for required in "$REPORT" "$LOG" "$TIME"; do
  if [ ! -s "$required" ]; then
    echo "missing checkpoint-parity artifact: $required" >&2
    exit 2
  fi
done

if ! grep -q '^foundation_predict_checkpoint_parity=PASS$' "$LOG"; then
  echo "foundation predictor checkpoint parity did not pass" >&2
  exit 2
fi
if ! awk -F $'\t' 'NR > 1 && $5 != "PASS" { exit 1 } END { exit NR < 4 }' "$REPORT"; then
  echo "checkpoint parity report contains a failure" >&2
  cat "$REPORT" >&2
  exit 2
fi

echo "foundation_predict_checkpoint_smoke=PASS"
cat "$REPORT"
echo "historical_validation_reference_reused=YES"
echo "alphapeptdeep_rerun=NO"
echo "historical_test_compared=NO"
echo "train_holdout_compared=NO"
echo "output=$OUT"
