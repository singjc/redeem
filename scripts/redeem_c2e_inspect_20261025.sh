#!/usr/bin/env bash
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2e_heldout_once_20261025"
if [ ! -s "$BASE/submitted_job_id.txt" ]; then
 echo 'REDEEM_C2E=NOT_SUBMITTED'
else
 ID=$(cat "$BASE/submitted_job_id.txt")
 if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
  echo 'ERROR: invalid job ID'
 else
  echo "REDEEM_C2E_JOB_ID=$ID"
  sacct -j "$ID" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
  seff "$ID"
  RUN="$BASE/runs/run_$ID"
  if [ -s "$RUN/SUCCESS.txt" ] && [ -s "$RUN/chemistry_features/feature_provenance.json" ]; then
   python3 -m json.tool "$RUN/chemistry_features/feature_provenance.json"
   echo 'REDEEM_C2E_EXPORT=PASS_LABEL_BLIND_ONLY'
  else
   echo 'REDEEM_C2E_EXPORT=INCOMPLETE_OR_FAILED'
   if [ -s "$BASE/runs/c2e-$ID.err" ]; then tail -n 80 "$BASE/runs/c2e-$ID.err"; fi
   if [ -s "$BASE/runs/c2e-$ID.out" ]; then tail -n 80 "$BASE/runs/c2e-$ID.out"; fi
  fi
 fi
fi
