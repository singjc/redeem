#!/usr/bin/env bash
# Execute in Codon interactive shell. No exit, logout, or shell-option mutation.
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2_d_once_score_20261024"
CODE="$BASE/code"; INPUT="$BASE/input"
SIF="$ROOT/containers/redeem-foundation_v0770-streaming-identifier_f7b8246c7730.sif"
CAT="$ROOT/experiments/redeem_pt2017_synthesis_newdev512_20261023/input/catalog.yaml"
value_sha() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }
if [ -e "$BASE/submitted_job_id.txt" ]; then
 echo 'ERROR: frozen DEV3 scoring already submitted; inspect the saved ID'
elif [ ! -s "$INPUT/requests.tsv" ] || [ "$(value_sha "$INPUT/requests.tsv")" != '7c9c4e2df5cd7b496fc44b8aca5fa1bd6cfdf7b7245e894569e265240af1c194' ]; then
 echo 'ERROR: DEV3 requests SHA mismatch'
elif [ -e "$INPUT/truth.tsv" ] || [ -e "$INPUT/evidence.tsv" ]; then
 echo 'ERROR: DEV3 truth/evidence found on Codon; stop'
elif ! (cd "$CODE" && sha256sum -c SHA256_REMOTE.txt); then
 echo 'ERROR: remote code SHA mismatch'
elif [ "$(value_sha "$SIF")" != 'd46ed2bcc39ed50dec081762788ac60f4f3551658d44b555ce0aacf33504bf22' ]; then
 echo 'ERROR: original parity-qualified SIF SHA mismatch'
elif [ "$(value_sha "$CAT")" != 'a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f' ]; then
 echo 'ERROR: source-defined synthesis catalog SHA mismatch'
elif ! singularity exec --bind "$ROOT:$ROOT" "$SIF" /bin/sh -c 'test -r "$1" && test -s "$1"' sh "$CAT"; then
 echo 'ERROR: catalog unreadable inside bound SIF; do not submit'
elif ! python3 - "$INPUT/metadata.json" <<'PY_CHECK'
import json,sys
m=json.load(open(sys.argv[1]))
expected={'status':'PASS_C2_D_PROSPECTIVE_DEV3_PREPARATION','queries':512,
 'source_partition':'validation','seed':20261024,'frozen_model_sha256':'47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca',
 'requests_sha256':'7c9c4e2df5cd7b496fc44b8aca5fa1bd6cfdf7b7245e894569e265240af1c194',
 'truth_sha256':'34d8c9ac18abd56e8c0e96cc6f952816d9ac2cf7c7e4934643881e85eecde07f',
 'protected_test_accessed':False}
assert all(m.get(k)==v for k,v in expected.items())
print('REDEEM_C2D_REMOTE_COHORT=PASS')
PY_CHECK
then
 echo 'ERROR: DEV3 cohort metadata changed'
else
 ID=$(sbatch --parsable --output="$BASE/runs/c2d-%j.out" --error="$BASE/runs/c2d-%j.err" "$CODE/redeem_c2d_once_score_20261024.sbatch")
 if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: sbatch did not produce a numeric job ID: $ID"
 elif ! printf '%s\n' "$ID" > "$BASE/submitted_job_id.txt"; then
  echo "ERROR: job $ID submitted but ID record could not be saved; inspect sacct"
 else
  echo "REDEEM_C2D_SUBMITTED=PASS job_id=$ID"
 fi
fi
