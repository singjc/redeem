#!/usr/bin/env bash
#SBATCH --job-name=redeem-c2e-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Frozen TEST confirmation; label-free candidate retrieval and chemistry only.
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2e_heldout_once_20261025"
CODE="$BASE/code"; INPUT="$BASE/input"
SIF="$ROOT/containers/redeem-foundation_v0770-streaming-identifier_f7b8246c7730.sif"
CAT="$ROOT/experiments/redeem_pt2017_synthesis_newdev512_20261023/input/catalog.yaml"
RUN="$BASE/runs/run_${SLURM_JOB_ID:-UNSET}"
if [ -z "${SLURM_JOB_ID:-}" ] || [ -e "$RUN" ]; then
 echo 'ERROR: missing Slurm ID or duplicate run'; false
elif [ "$(sha256sum "$SIF" | awk '{print $1}')" != 'd46ed2bcc39ed50dec081762788ac60f4f3551658d44b555ce0aacf33504bf22' ]; then
 echo 'ERROR: SIF SHA changed'; false
elif [ "$(sha256sum "$CAT" | awk '{print $1}')" != 'a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f' ]; then
 echo 'ERROR: catalog changed'; false
elif [ -e "$INPUT/truth.tsv" ] || [ -e "$INPUT/evidence.tsv" ]; then
 echo 'ERROR: forbidden TEST labels on Codon'; false
elif ! (cd "$CODE" && sha256sum -c SHA256_REMOTE.txt); then
 echo 'ERROR: source checksum mismatch'; false
elif ! python3 - "$CODE" "$INPUT" <<'PY_CHECK'
import sys
sys.path.insert(0,sys.argv[1]);from pathlib import Path
import redeem_c2e_lock_20261025 as lock
m=lock.test_metadata(Path(sys.argv[2])/'metadata.json',Path(sys.argv[1])/'LOCKED_PROTOCOL_20261025.json')
assert lock.sha(Path(sys.argv[2])/'requests.tsv')==m['requests_sha256']
print('REDEEM_C2E_STAGED_TEST=PASS')
PY_CHECK
then
 echo 'ERROR: staged TEST metadata invalid'; false
elif ! mkdir "$RUN"; then
 echo 'ERROR: cannot create immutable run directory'; false
elif ! singularity exec --bind "$ROOT:$ROOT" "$SIF" foundation_identify_stream_v0770 \
 identify-stream --catalog "$CAT" --spectra "$INPUT/requests.tsv" \
 --output-hits "$RUN/identifier_hits.tsv" --output-summary "$RUN/identifier_summary.tsv"; then
 echo 'ERROR: frozen identifier failed; preserve partial run'; false
elif ! python3 "$CODE/redeem_c2e_export_20261025.py" --requests "$INPUT/requests.tsv" \
 --hits "$RUN/identifier_hits.tsv" --metadata "$INPUT/metadata.json" \
 --protocol "$CODE/LOCKED_PROTOCOL_20261025.json" --output "$RUN/chemistry_features"; then
 echo 'ERROR: TEST chemistry export failed; preserve partial run'; false
else
 printf 'status=PASS_C2E_LABEL_BLIND_EXPORT\njob_id=%s\n' "$SLURM_JOB_ID" > "$RUN/SUCCESS.txt"
 echo "REDEEM_C2E_EXPORT=PASS job_id=$SLURM_JOB_ID"
fi
