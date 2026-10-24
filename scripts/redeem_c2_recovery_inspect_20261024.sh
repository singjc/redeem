#!/usr/bin/env bash
# Codon: inspect the recovery job, not the original failed submission.
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2_train_only_20261023"
REC="$BASE/recovery_catalog_path_20261024"
if [ ! -s "$REC/submitted_job_id.txt" ]; then
  echo 'REDEEM_C2_RECOVERY=NOT_SUBMITTED'
else
  ID=$(cat "$REC/submitted_job_id.txt")
  if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
    echo 'ERROR: saved recovery job ID is invalid'
  else
    echo "REDEEM_C2_RECOVERY_JOB_ID=$ID"
    sacct -j "$ID" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
    FILE="$BASE/fits/run_$ID/chemistry_features/feature_provenance.json"
    if [ -s "$FILE" ]; then
      python3 - "$FILE" <<'PY_VERIFY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m['status']=='PASS_TRAIN_ONLY_CHEMISTRY_FEATURE_EXPORT'
assert m['queries']==2048 and m['candidate_rows']==524288
assert m['candidate_rows_per_query']==256
assert m['reference_truth_accessed'] is False
assert m['protected_test_accessed'] is False
print(json.dumps(m,indent=2,sort_keys=True))
print('REDEEM_C2_RECOVERY_FEATURES=PASS')
PY_VERIFY
    else
      echo 'REDEEM_C2_RECOVERY_FEATURES=PENDING_OR_FAILED'
      if [ -s "$BASE/fits/c2-catalog-recovery-$ID.err" ]; then tail -40 "$BASE/fits/c2-catalog-recovery-$ID.err"; fi
      if [ -s "$BASE/fits/c2-catalog-recovery-$ID.out" ]; then tail -40 "$BASE/fits/c2-catalog-recovery-$ID.out"; fi
    fi
  fi
fi
