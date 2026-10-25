#!/usr/bin/env bash
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2_optional_inference_20261025"
if [ ! -s "$BASE/job_id.txt" ]; then
  echo 'ERROR: job ID not recorded'
else
  ID=$(cat "$BASE/job_id.txt")
  echo "REDEEM_C2_OPTIONAL_JOB_ID=$ID"
  sacct -j "$ID" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
  if [ -s "$BASE/runs/smoke_${ID}/smoke_report.json" ]; then
    python3 -m json.tool "$BASE/runs/smoke_${ID}/smoke_report.json"
    grep -E 'Elapsed \(wall clock\) time|Maximum resident set size|User time \(seconds\)' \
      "$BASE/runs/smoke_${ID}/geometry_resources.txt" "$BASE/runs/smoke_${ID}/c2_resources.txt"
  else
    echo "REDEEM_C2_OPTIONAL_SMOKE=PENDING_OR_FAILED inspect $BASE/slurm-${ID}.out $BASE/slurm-${ID}.err"
  fi
fi
