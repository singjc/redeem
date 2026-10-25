#!/usr/bin/env bash
# Codon login: read-only preflight followed by at most one sbatch.
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2e_heldout_once_20261025"
CODE="$BASE/code"; INPUT="$BASE/input"
SIF="$ROOT/containers/redeem-foundation_v0770-streaming-identifier_f7b8246c7730.sif"
CAT="$ROOT/experiments/redeem_pt2017_synthesis_newdev512_20261023/input/catalog.yaml"
sha() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }
if [ -e "$BASE/submitted_job_id.txt" ]; then
 echo 'ERROR: once-only TEST job already submitted'
elif [ ! -s "$INPUT/requests.tsv" ] || [ ! -s "$INPUT/metadata.json" ]; then
 echo 'ERROR: TEST requests/metadata absent'
elif [ -e "$INPUT/truth.tsv" ] || [ -e "$INPUT/evidence.tsv" ]; then
 echo 'ERROR: TEST labels transferred to Codon; stop'
elif ! (cd "$CODE" && sha256sum -c SHA256_REMOTE.txt); then
 echo 'ERROR: remote package checksum mismatch'
elif [ "$(sha "$SIF")" != 'd46ed2bcc39ed50dec081762788ac60f4f3551658d44b555ce0aacf33504bf22' ]; then
 echo 'ERROR: parity-qualified identifier SIF changed'
elif [ "$(sha "$CAT")" != 'a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f' ]; then
 echo 'ERROR: immutable source catalog changed'
elif ! singularity exec --bind "$ROOT:$ROOT" "$SIF" /bin/sh -c 'test -r "$1" && test -s "$1"' sh "$CAT"; then
 echo 'ERROR: catalogue is unreadable inside SIF bind'
elif ! python3 - "$CODE" "$INPUT" <<'PY_CHECK'
import sys
sys.path.insert(0,sys.argv[1]);from pathlib import Path
import redeem_c2e_lock_20261025 as lock
m=lock.test_metadata(Path(sys.argv[2])/'metadata.json',Path(sys.argv[1])/'LOCKED_PROTOCOL_20261025.json')
assert lock.sha(Path(sys.argv[2])/'requests.tsv')==m['requests_sha256']
assert m['source_partition']=='test'
print('REDEEM_C2E_REMOTE_PREFLIGHT=PASS')
PY_CHECK
then
 echo 'ERROR: staged TEST cohort metadata invalid'
else
 ID=$(sbatch --parsable --output="$BASE/runs/c2e-%j.out" --error="$BASE/runs/c2e-%j.err" "$CODE/redeem_c2e_once_score_20261025.sbatch")
 if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: sbatch ID invalid: $ID"
 elif ! printf '%s\n' "$ID" > "$BASE/submitted_job_id.txt"; then
  echo "ERROR: job $ID was submitted; ID record failed; inspect sacct. Do not resubmit"
 else
  echo "REDEEM_C2E_SUBMITTED=PASS job_id=$ID"
 fi
fi
