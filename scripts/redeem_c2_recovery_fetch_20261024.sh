#!/usr/bin/env bash
# Laptop: fetch only the recovery job's TRAIN feature output.
# Do not use the old local helper's fetch action; it reads failed job 6720538.
REMOTE='/nfs/research/juan/DIA/singj/foundation_model/experiments/redeem_c2_train_only_20261023'
LOCAL="$HOME/Downloads/redeem_c2_train2048_feature_export_20261023"
PY="$HOME/.venvs/redeem-external-dev-uv-20261022/bin/python"
if [ -e "$LOCAL" ] || [ -L "$LOCAL" ]; then
  echo 'ERROR: local feature export destination exists; preserve it'
elif [ ! -x "$PY" ]; then
  echo 'ERROR: existing uv-managed interpreter missing'
else
  ID=$(ssh codon "cat '$REMOTE/recovery_catalog_path_20261024/submitted_job_id.txt'" 2>/dev/null)
  if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
    echo 'ERROR: missing or invalid recovery job ID on Codon'
  elif ! ssh codon "test -s '$REMOTE/fits/run_$ID/chemistry_features/feature_provenance.json' && test -s '$REMOTE/fits/run_$ID/chemistry_features/candidate_chemistry_features.tsv'"; then
    echo 'ERROR: recovery feature export incomplete; inspect job on Codon'
  elif ! mkdir "$LOCAL"; then
    echo 'ERROR: cannot create local fetch destination'
  elif ! scp "codon:$REMOTE/fits/run_$ID/chemistry_features/feature_provenance.json" "$LOCAL/"; then
    echo 'ERROR: feature provenance transfer failed; preserve partial destination'
  elif ! scp "codon:$REMOTE/fits/run_$ID/chemistry_features/candidate_chemistry_features.tsv" "$LOCAL/"; then
    echo 'ERROR: feature table transfer failed; preserve partial destination'
  elif ! "$PY" - "$LOCAL" "$ID" <<'PY_VERIFY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]); jobid=sys.argv[2]
m=json.loads((root/'feature_provenance.json').read_text())
assert m['status']=='PASS_TRAIN_ONLY_CHEMISTRY_FEATURE_EXPORT'
assert m['source_scoring_job_id']==jobid
assert m['queries']==2048 and m['candidate_rows']==524288
assert m['candidate_rows_per_query']==256
assert m['reference_truth_accessed'] is False and m['protected_test_accessed'] is False
assert m['requests_sha256']=='26dc26a242c5cb039bdba91d7cbfa0ea6b0e14f27777ba1f54f0a8e7bf2c5a25'
actual=hashlib.sha256((root/'candidate_chemistry_features.tsv').read_bytes()).hexdigest()
assert actual==m['features_sha256']
print(f'REDEEM_C2_RECOVERY_FETCH=PASS job_id={jobid} rows={m["candidate_rows"]} features_sha256={actual}')
PY_VERIFY
  then
    echo 'ERROR: fetched feature provenance/hash mismatch; preserve partial destination'
  else
    echo 'NEXT_ACTION=bash "$HOME/Downloads/REDEEM_C2_TRAIN_ONLY_20261023/redeem_c2_train_local_20261023.sh" train'
  fi
fi
