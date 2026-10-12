#!/bin/bash -l
#SBATCH --job-name=redeem-v0520-extval
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0520-extval-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0520-extval-%j.err

main() {
  local ROOT=/nfs/research/juan/DIA/singj/foundation_model
  local SIF="${NATIVE_SIF:-}"
  local SHA="${EXPECTED_NATIVE_SHA256:-}"
  local MEASUREMENTS="${MEASUREMENTS:-}"
  local MEAS_SHA="${EXPECTED_MEASUREMENTS_SHA256:-}"
  local TOOL="$ROOT/scripts/redeem_v0520_external_study_eval_20261012.py"
  local TOOL_SHA="4093c4624b57fe82dfb7d47c57e2aa2458e25384e03dc964dfd6e0fd082b2a7a"
  local FRAGMENTS="${OBSERVED_FRAGMENTS:-}"
  local FRAG_SHA="${EXPECTED_OBSERVED_FRAGMENTS_SHA256:-}"
  local V052="$ROOT/experiments/multimodal_v0520_02_mobility_aware_pair_representation_full/model/best"
  local V035="$ROOT/experiments/multimodal_v0350_01_trainable_forward_representation/model/best"
  local OUT="$ROOT/experiments/foundation_v0520_external_study_v1/study_${SLURM_JOB_ID:-manual}"
  local PANEL="$OUT/inference_panel.tsv"
  local META="$OUT/inference_panel.manifest.json"
  local PRED="$OUT/predictions.tsv"
  local REPORT="$OUT/evaluation.json"
  local CONFIG="$OUT/inference.json"
  local FILE
  if [[ -z "$SIF" || -z "$MEASUREMENTS" || ! "$SHA" =~ ^[0-9a-f]{64}$ || ! "$MEAS_SHA" =~ ^[0-9a-f]{64}$ ]]; then
    echo 'ERROR: NATIVE_SIF, SHA, MEASUREMENTS and its SHA are required' >&2; return 2
  fi
  for FILE in "$SIF" "$MEASUREMENTS" "$TOOL" "$V052/metadata.yaml" "$V052/model.safetensors" \
              "$V035/metadata.yaml" "$V035/model.safetensors"; do
    if [[ ! -s "$FILE" ]]; then echo "ERROR: missing $FILE" >&2; return 2; fi
  done
  if [[ "$(sha256sum "$SIF" | awk '{print $1}')" != "$SHA" || \
        "$(sha256sum "$MEASUREMENTS" | awk '{print $1}')" != "$MEAS_SHA" || \
        "$(sha256sum "$TOOL" | awk '{print $1}')" != "$TOOL_SHA" ]]; then
    echo 'ERROR: SIF, measurements, or evaluator checksum mismatch' >&2; return 2
  fi
  if [[ -n "$FRAGMENTS" ]]; then
    if [[ ! -s "$FRAGMENTS" || ! "$FRAG_SHA" =~ ^[0-9a-f]{64}$ || \
          "$(sha256sum "$FRAGMENTS" | awk '{print $1}')" != "$FRAG_SHA" ]]; then
      echo 'ERROR: observed fragment TSV missing or SHA mismatch' >&2; return 2
    fi
  fi
  if [[ -e "$OUT" ]]; then echo "ERROR: output directory already exists: $OUT" >&2; return 2; fi
  if ! mkdir -p "$OUT"; then echo 'ERROR: output directory cannot be created' >&2; return 2; fi
  echo '===== REDEEM V0.52 EXTERNAL MEASURED STUDY ====='
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "sif_sha256=$SHA"
  echo "measurements_sha256=$MEAS_SHA"
  echo "output=$OUT"
  if ! python3 "$TOOL" self-test > "$OUT/evaluator_selftest.log" 2>&1 || \
     ! python3 "$TOOL" prepare --measurements "$MEASUREMENTS" --panel "$PANEL" \
                           --manifest "$META" > "$OUT/prepare.log" 2>&1; then
    echo 'ERROR: evaluator self-test or label-blind panel preparation failed' >&2
    cat "$OUT/evaluator_selftest.log" "$OUT/prepare.log" 2>/dev/null; return 1
  fi
  if ! module load singularityce/4.1.0; then
    echo 'ERROR: singularity module unavailable' >&2; return 2
  fi
  cat > "$CONFIG" <<JSON
{
  "model_arch": "foundation",
  "foundation_backend": "v0520_peptide_panel",
  "model_path": "$V052",
  "foundation_v0350_teacher_checkpoint": "$V035",
  "inference_data": "$PANEL",
  "foundation_partition": "external",
  "foundation_spectra_only": false,
  "device": "cuda",
  "batch_size": 32,
  "output_file": "$PRED"
}
JSON
  nvidia-smi -L
  if ! singularity exec --nv --bind "$ROOT:$ROOT" "$SIF" \
      redeem properties inference "$CONFIG" > "$OUT/inference.log" 2>&1; then
    echo 'ERROR: native v0.52 inference failed' >&2
    tail -n 80 "$OUT/inference.log" >&2; return 1
  fi
  if ! grep -qx 'foundation_backend=v0520_peptide_panel' "$OUT/inference.log" || \
     ! grep -qx 'benchmark_partitions_read=NO' "$OUT/inference.log"; then
    echo 'ERROR: inference confirmation missing' >&2; return 1
  fi
  local OPTIONS=(--measurements "$MEASUREMENTS" --panel "$PANEL" --manifest "$META" \
                 --predictions "$PRED" --report "$REPORT")
  if [[ -n "$FRAGMENTS" ]]; then
    OPTIONS+=(--predicted-fragments "$PRED.predicted_fragments.tsv" --measured-fragments "$FRAGMENTS")
  fi
  if ! python3 "$TOOL" score "${OPTIONS[@]}" > "$OUT/score.log" 2>&1; then
    echo 'ERROR: measured-study scoring failed' >&2
    cat "$OUT/score.log" >&2; return 1
  fi
  cat "$OUT/prepare.log" "$OUT/score.log"
  sha256sum "$PANEL" "$META" "$PRED" "$PRED.predicted_fragments.tsv" "$REPORT"
  echo 'foundation_v0520_external_study_evaluation=PASS'
  echo 'historical_test_consumed=NO'
  echo 'train_holdout_read_by_this_job=NO'
}
main
