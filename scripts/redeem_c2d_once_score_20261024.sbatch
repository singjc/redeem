#!/usr/bin/env bash
#SBATCH --job-name=redeem-c2d-dev3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Frozen candidate ranking and label-free feature export. No labels or trained model on Codon.
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2_d_once_score_20261024"
CODE="$BASE/code"; INPUT="$BASE/input"
SIF="$ROOT/containers/redeem-foundation_v0770-streaming-identifier_f7b8246c7730.sif"
CAT="$ROOT/experiments/redeem_pt2017_synthesis_newdev512_20261023/input/catalog.yaml"
PIN='d46ed2bcc39ed50dec081762788ac60f4f3551658d44b555ce0aacf33504bf22'
CAT_SHA='a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f'
REQ_SHA='7c9c4e2df5cd7b496fc44b8aca5fa1bd6cfdf7b7245e894569e265240af1c194'
RUN="$BASE/runs/run_${SLURM_JOB_ID:-UNSET}"
value_sha() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }
if [ -z "${SLURM_JOB_ID:-}" ] || [ -e "$RUN" ]; then
  echo 'ERROR: job ID absent or run directory already exists'; false
elif [ ! -s "$SIF" ] || [ "$(value_sha "$SIF")" != "$PIN" ]; then
  echo 'ERROR: qualified SIF changed'; false
elif [ ! -s "$CAT" ] || [ "$(value_sha "$CAT")" != "$CAT_SHA" ]; then
  echo 'ERROR: frozen catalog missing or changed'; false
elif [ ! -s "$INPUT/requests.tsv" ] || [ "$(value_sha "$INPUT/requests.tsv")" != "$REQ_SHA" ]; then
  echo 'ERROR: frozen DEV3 requests mismatch'; false
elif [ -e "$INPUT/truth.tsv" ] || [ -e "$INPUT/evidence.tsv" ]; then
  echo 'ERROR: DEV3 labels on Codon; prohibited'; false
elif ! (cd "$CODE" && sha256sum -c SHA256_REMOTE.txt); then
  echo 'ERROR: scoring code checksum changed'; false
elif ! mkdir "$RUN"; then
  echo 'ERROR: could not create immutable new run directory'; false
elif ! singularity exec --bind "$ROOT:$ROOT" "$SIF" foundation_identify_stream_v0770 \
     identify-stream --catalog "$CAT" --spectra "$INPUT/requests.tsv" \
     --output-hits "$RUN/identifier_hits.tsv" --output-summary "$RUN/identifier_summary.tsv"; then
  echo 'ERROR: unchanged v0.77 DEV3 identification failed; inspect partial output'; false
elif ! python3 "$CODE/redeem_c2d_export_20261024.py" \
     --requests "$INPUT/requests.tsv" --hits "$RUN/identifier_hits.tsv" \
     --metadata "$INPUT/metadata.json" --output "$RUN/chemistry_features"; then
  echo 'ERROR: chemistry feature export failed; preserve partial run'; false
else
  printf 'status=PASS_C2D_DEV3_LABEL_BLIND_EXPORT\njob_id=%s\n' "$SLURM_JOB_ID" > "$RUN/SUCCESS.txt"
  echo "REDEEM_C2D_DEV3_EXPORT=PASS job_id=$SLURM_JOB_ID"
fi
