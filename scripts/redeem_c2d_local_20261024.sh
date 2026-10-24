#!/usr/bin/env bash
# Laptop; do not use from Codon. All activities are guarded, immutable, and one-time.
ACTION="${1:-}"
PKG="$HOME/Downloads/REDEEM_C2_D_ONCE_SCORE_20261024"
COHORT="$HOME/Downloads/redeem_c2_d_prospective_dev512_20261024"
COVERAGE="$HOME/Downloads/redeem_c2_d_prospective_coverage_20261024.json"
MODEL="$HOME/Downloads/redeem_c2_frozen_ranker_20261023/model.json"
OUTPUT="$HOME/Downloads/redeem_c2_d_once_scoring_20261024"
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
REMOTE="$ROOT/experiments/redeem_c2_d_once_score_20261024"
PY="$HOME/.venvs/redeem-external-dev-uv-20261022/bin/python"
REQ_SHA='7c9c4e2df5cd7b496fc44b8aca5fa1bd6cfdf7b7245e894569e265240af1c194'
MODEL_SHA='47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca'
CODE_FILES=(redeem_c2d_export_20261024.py redeem_c2_train_features_20261023.py redeem_c2d_once_score_20261024.sbatch redeem_c2d_submit_20261024.sh redeem_c2d_inspect_20261024.sh)
actual_sha() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }

if [ "$ACTION" = transfer ]; then
 if [ ! -s "$COHORT/requests.tsv" ] || [ "$(actual_sha "$COHORT/requests.tsv")" != "$REQ_SHA" ]; then
  echo 'ERROR: immutable prospective DEV3 requests are missing/changed'
 elif [ ! -s "$COHORT/metadata.json" ]; then
  echo 'ERROR: DEV3 metadata missing'
 elif [ ! -s "$COVERAGE" ] || [ ! -s "$MODEL" ] || [ "$(actual_sha "$MODEL")" != "$MODEL_SHA" ]; then
  echo 'ERROR: coverage gate or frozen model missing/changed'
 elif [ ! -x "$PY" ]; then
  echo 'ERROR: uv-managed Python interpreter unavailable'
 elif ! "$PY" - "$COHORT/metadata.json" "$COVERAGE" <<'PY_VERIFY'
import json,sys
m=json.load(open(sys.argv[1]));c=json.load(open(sys.argv[2]))
assert m['status']=='PASS_C2_D_PROSPECTIVE_DEV3_PREPARATION'
assert m['queries']==512 and m['frozen_model_sha256']=='47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca'
assert m['requests_sha256']=='7c9c4e2df5cd7b496fc44b8aca5fa1bd6cfdf7b7245e894569e265240af1c194'
assert m['truth_sha256']=='34d8c9ac18abd56e8c0e96cc6f952816d9ac2cf7c7e4934643881e85eecde07f'
assert c['decision']=='ALLOW_ONCE_SCORING' and c['truth_in_catalog']==297 and c['catalog_sha256']=='a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f'
assert c['requests_sha256']==m['requests_sha256'] and c['truth_sha256']==m['truth_sha256']
print('REDEEM_C2D_LOCAL_GATE=PASS 297/512')
PY_VERIFY
 then
  echo 'ERROR: immutability/coverage gate failed'
 elif ! (cd "$PKG" && sha256sum -c SHA256_REMOTE.txt); then
  echo 'ERROR: packaged remote-code SHA mismatch'
 elif ! ssh codon "test ! -e '$REMOTE' && test ! -L '$REMOTE'"; then
  echo 'ERROR: remote scoring directory exists; preserve it'
 elif ! ssh codon "mkdir -p '$REMOTE/code' '$REMOTE/input' '$REMOTE/runs'"; then
  echo 'ERROR: could not create remote experiment root'
 elif ! scp "$PKG/redeem_c2d_export_20261024.py" \
           "$PKG/redeem_c2_train_features_20261023.py" \
           "$PKG/redeem_c2d_once_score_20261024.sbatch" \
           "$PKG/redeem_c2d_submit_20261024.sh" \
           "$PKG/redeem_c2d_inspect_20261024.sh" \
           "$PKG/SHA256_REMOTE.txt" "codon:$REMOTE/code/"; then
  echo 'ERROR: partial remote code transfer; preserve and inspect'
 elif ! scp "$COHORT/requests.tsv" "$COHORT/metadata.json" "codon:$REMOTE/input/"; then
  echo 'ERROR: remote request transfer incomplete; preserve directory'
 elif ! ssh codon "cd '$REMOTE/code' && sha256sum -c SHA256_REMOTE.txt && printf '%s  %s\n' '$REQ_SHA' '$REMOTE/input/requests.tsv' | sha256sum -c - && test ! -e '$REMOTE/input/truth.tsv' && test ! -e '$REMOTE/input/evidence.tsv'"; then
  echo 'ERROR: Codon verification failed; no job submitted'
 else
  echo 'REDEEM_C2D_TRANSFER=PASS inputs=label_free next=remote_submit'
 fi
elif [ "$ACTION" = fetch ]; then
 if [ -e "$OUTPUT" ] || [ -L "$OUTPUT" ]; then
  echo 'ERROR: local scoring output already exists; preserve it'
 elif [ ! -x "$PY" ]; then
  echo 'ERROR: uv Python missing'
 else
  ID=$(ssh codon "cat '$REMOTE/submitted_job_id.txt'" 2>/dev/null)
  if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
   echo 'ERROR: missing/invalid remote DEV3 scoring job ID'
  elif ! ssh codon "test -s '$REMOTE/runs/run_$ID/SUCCESS.txt' && test -s '$REMOTE/runs/run_$ID/identifier_hits.tsv' && test -s '$REMOTE/runs/run_$ID/chemistry_features/candidate_chemistry_features.tsv' && test -s '$REMOTE/runs/run_$ID/chemistry_features/feature_provenance.json'"; then
   echo 'ERROR: scoring job incomplete; inspect Codon log'
  elif ! mkdir "$OUTPUT"; then
   echo 'ERROR: cannot create local immutable scoring output'
  elif ! scp "codon:$REMOTE/runs/run_$ID/identifier_hits.tsv" "$OUTPUT/identifier_hits.tsv"; then
   echo 'ERROR: hits transfer incomplete'
  elif ! scp "codon:$REMOTE/runs/run_$ID/chemistry_features/candidate_chemistry_features.tsv" "$OUTPUT/candidate_chemistry_features.tsv"; then
   echo 'ERROR: features transfer incomplete'
  elif ! scp "codon:$REMOTE/runs/run_$ID/chemistry_features/feature_provenance.json" "$OUTPUT/feature_provenance.json"; then
   echo 'ERROR: feature provenance transfer incomplete'
  elif ! "$PY" - "$OUTPUT" "$ID" <<'PY_CHECK'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);job=sys.argv[2]
m=json.loads((p/'feature_provenance.json').read_text())
assert m['status']=='PASS_LABEL_BLIND_C2D_DEV3_FEATURE_EXPORT' and m['source_scoring_job_id']==job
assert m['queries']==512 and m['candidate_rows']==131072 and m['reference_truth_accessed'] is False
assert m['model_fit_performed'] is False and m['protected_test_accessed'] is False
for name,key in [('identifier_hits.tsv','hits_sha256'),('candidate_chemistry_features.tsv','features_sha256')]:
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==m[key]
print('REDEEM_C2D_FETCH=PASS label_blind_rows=131072 job='+job)
PY_CHECK
  then
   echo 'ERROR: fetched input integrity check failed; preserve output'
  else
   echo 'REDEEM_C2D_FETCH=PASS next=once_evaluate'
  fi
 fi
elif [ "$ACTION" = evaluate ]; then
 if [ ! -x "$PY" ] || [ "$(actual_sha "$MODEL")" != "$MODEL_SHA" ]; then
  echo 'ERROR: uv Python or frozen C2 model missing/changed'
 elif [ ! -s "$OUTPUT/feature_provenance.json" ] || [ ! -s "$OUTPUT/candidate_chemistry_features.tsv" ] || [ ! -s "$OUTPUT/identifier_hits.tsv" ]; then
  echo 'ERROR: verified label-blind outputs missing; fetch first'
 elif [ "$(actual_sha "$PKG/redeem_c2_ranker_20261023.py")" != '9128c3db1d2996059afc81d8422b9ec33f7b629d5627b56b352fe67700b8d498' ]; then
  echo 'ERROR: TRAIN-fitted C2 apply implementation changed'
 elif [ -e "$OUTPUT/frozen_c2_ranking.tsv" ] || [ -e "$OUTPUT/final_comparison.json" ]; then
  echo 'ERROR: comparison already exists; never re-evaluate or overwrite'
 elif ! "$PY" - "$OUTPUT" <<'PY_VALIDATE'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/'feature_provenance.json').read_text())
assert m['status']=='PASS_LABEL_BLIND_C2D_DEV3_FEATURE_EXPORT' and m['queries']==512
for file,key in [('identifier_hits.tsv','hits_sha256'),('candidate_chemistry_features.tsv','features_sha256')]:
 assert hashlib.sha256((p/file).read_bytes()).hexdigest()==m[key]
print('REDEEM_C2D_BEFORE_APPLY=PASS')
PY_VALIDATE
 then
  echo 'ERROR: DEV3 feature/hit hashes do not match recorded provenance'
 else
  FEATURES_SHA=$("$PY" - "$OUTPUT/feature_provenance.json" <<'PY_GET'
import json,sys
print(json.load(open(sys.argv[1]))['features_sha256'])
PY_GET
)
  if ! "$PY" "$PKG/redeem_c2_ranker_20261023.py" apply \
      --model "$MODEL" --features "$OUTPUT/candidate_chemistry_features.tsv" \
      --features-sha "$FEATURES_SHA" --queries 512 \
      --output "$OUTPUT/frozen_c2_ranking.tsv"; then
   echo 'ERROR: frozen C2 apply failed; preserve incomplete results'
  elif ! "$PY" "$PKG/redeem_c2d_evaluate_20261024.py" \
      --metadata "$COHORT/metadata.json" --truth "$COHORT/truth.tsv" \
      --model "$MODEL" --coverage "$COVERAGE" \
      --hits "$OUTPUT/identifier_hits.tsv" --features "$OUTPUT/candidate_chemistry_features.tsv" \
      --feature-provenance "$OUTPUT/feature_provenance.json" \
      --ranked "$OUTPUT/frozen_c2_ranking.tsv" --output "$OUTPUT/final_comparison.json"; then
   echo 'ERROR: one-time DEV3 evaluation failed; preserve results for inspection'
  else
   "$PY" -m json.tool "$OUTPUT/final_comparison.json"
  fi
 fi
else
 echo 'Usage: bash redeem_c2d_local_20261024.sh transfer|fetch|evaluate'
fi
