#!/usr/bin/env bash
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2_optional_inference_20261025"
CODE="$BASE/code"
if [ -e "$BASE/job_id.txt" ]; then
  echo 'ERROR: smoke job ID already present; inspect existing job'
elif [ ! -s "$BASE/model/model.json" ] || [ ! -s "$CODE/SHA256_CODE.txt" ]; then
  echo 'ERROR: model or source scripts are not staged'
elif ! (cd "$CODE" && sha256sum -c SHA256_CODE.txt); then
  echo 'ERROR: code checksum mismatch'
elif ! printf '%s  %s\n' '47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca' "$BASE/model/model.json" | sha256sum -c -; then
  echo 'ERROR: model SHA mismatch'
else
  ID=$(sbatch --parsable "$CODE/redeem_c2_optional_smoke_20261025.sbatch")
  if [[ "$ID" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$ID" > "$BASE/job_id.txt"
    echo "REDEEM_C2_OPTIONAL_SMOKE_SUBMITTED=PASS job_id=$ID"
  else
    echo "ERROR: sbatch failed or returned unexpected job ID: $ID"
  fi
fi
