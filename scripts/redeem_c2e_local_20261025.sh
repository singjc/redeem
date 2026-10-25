#!/usr/bin/env bash
# Laptop only. Preserve one-time TEST embargo. Never call in Codon SSH shell.
ACTION="${1:-}"
PKG="$HOME/Downloads/REDEEM_C2E_HELDOUT_TEST_20261025"
PROTOCOL="$PKG/LOCKED_PROTOCOL_20261025.json"
COHORT="$HOME/Downloads/redeem_c2e_heldout_test512_20261025"
RESULT="$HOME/Downloads/redeem_c2e_test_once_result_20261025"
MODEL="$HOME/Downloads/redeem_c2_frozen_ranker_20261023/model.json"
CATALOG="$HOME/Downloads/redeem_pt_synthesis_catalog_3sheets_20261023/catalog.yaml"
DEV1="$HOME/Downloads/redeem_external_dev_validation_20261022/truth.tsv"
DEV2="$HOME/Downloads/redeem_pt_2017_new_validation_20261022/truth.tsv"
DEV3="$HOME/Downloads/redeem_c2_d_prospective_dev512_20261024/truth.tsv"
TRAIN="$HOME/Downloads/redeem_c2_train2048_20261023/truth.tsv"
PY="$HOME/.venvs/redeem-external-dev-uv-20261022/bin/python"
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
REMOTE="$ROOT/experiments/redeem_c2e_heldout_once_20261025"
REPO='InstaDeepAI/ms_proteometools'
REV='ffde6e4a2925f94dc28c87e33917db19a28f5869'
FILE='data/test-00000-of-00001-5f85882133d4ee76.parquet'
CODE_FILES=(LOCKED_PROTOCOL_20261025.json redeem_c2e_lock_20261025.py \
 redeem_c2e_export_20261025.py redeem_c2_train_features_20261023.py \
 redeem_c2e_once_score_20261025.sbatch redeem_c2e_submit_20261025.sh \
 redeem_c2e_inspect_20261025.sh)
sha() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }

if [ ! -d "$PKG" ]; then
 echo 'ERROR: extracted package missing'
elif ! (cd "$PKG" && sha256sum -c SHA256SUMS.txt >/dev/null); then
 echo 'ERROR: package checksum verification failed'
elif [ ! -x "$PY" ] || [ ! -s "$PROTOCOL" ]; then
 echo 'ERROR: uv-managed interpreter or frozen protocol not found'
elif ! "$PY" - "$PKG" <<'PY_PREFLIGHT'
import sys
sys.path.insert(0,sys.argv[1]);from pathlib import Path
import redeem_c2e_lock_20261025 as lock
lock.protocol(Path(sys.argv[1])/'LOCKED_PROTOCOL_20261025.json')
print('REDEEM_C2E_LOCK=PASS preregistered-before-test-access')
PY_PREFLIGHT
then
 echo 'ERROR: preregistration hash verification failed'
elif [ "$ACTION" = download ] || [ "$ACTION" = locate ]; then
 # Cache lookups avoid touching any PSM label; online download uses normal verified TLS.
 if [ "$ACTION" = download ]; then
  HF_HUB_OFFLINE=0 "$PY" - "$REPO" "$FILE" "$REV" <<'PY_HF'
import sys
from huggingface_hub import hf_hub_download
path=hf_hub_download(repo_id=sys.argv[1],filename=sys.argv[2],revision=sys.argv[3],repo_type='dataset')
print('REDEEM_C2E_PINNED_TEST_SNAPSHOT='+path)
PY_HF
 else
  HF_HUB_OFFLINE=1 "$PY" - "$REPO" "$FILE" "$REV" <<'PY_HF'
import sys
from huggingface_hub import hf_hub_download
try:
 path=hf_hub_download(repo_id=sys.argv[1],filename=sys.argv[2],revision=sys.argv[3],repo_type='dataset',local_files_only=True)
except Exception as exc:
 print('ERROR: pinned TEST shard not in snapshot cache; run download over verified TLS. '+str(exc));raise SystemExit(1)
print('REDEEM_C2E_PINNED_TEST_SNAPSHOT='+path)
PY_HF
 fi
elif [ "$ACTION" = prepare ]; then
 if [ -e "$COHORT" ] || [ -L "$COHORT" ]; then
  echo 'ERROR: TEST cohort already exists. Never reselect or overwrite'
 elif [ "$(sha "$MODEL")" != '47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca' ]; then
  echo 'ERROR: C2 model SHA differs'
 else
  TEST_PATH=$(HF_HUB_OFFLINE=1 "$PY" - "$REPO" "$FILE" "$REV" <<'PY_CACHE'
import sys
from huggingface_hub import hf_hub_download
try:print(hf_hub_download(repo_id=sys.argv[1],filename=sys.argv[2],revision=sys.argv[3],repo_type='dataset',local_files_only=True))
except Exception as exc:print('ERROR:'+str(exc),file=sys.stderr);raise SystemExit(1)
PY_CACHE
)
  if [ ! -s "$TEST_PATH" ]; then
   echo 'ERROR: pinned TEST shard missing; do not open TEST'
  else
   "$PY" "$PKG/redeem_c2e_prepare_20261025.py" \
    --protocol "$PROTOCOL" --parquet "$TEST_PATH" \
    --model-dir "$(dirname "$MODEL")" --dev1 "$DEV1" --dev2 "$DEV2" \
    --dev3 "$DEV3" --train "$TRAIN" --output "$COHORT"
  fi
 fi
elif [ "$ACTION" = transfer ]; then
 if [ ! -s "$COHORT/requests.tsv" ] || [ ! -s "$COHORT/metadata.json" ]; then
  echo 'ERROR: frozen TEST cohort requests/metadata missing'
 elif ! "$PY" - "$PKG" "$COHORT" <<'PY_PRE'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import redeem_c2e_lock_20261025 as lock
p=Path(sys.argv[2]);m=lock.test_metadata(p/'metadata.json',Path(sys.argv[1])/'LOCKED_PROTOCOL_20261025.json')
assert lock.sha(p/'requests.tsv')==m['requests_sha256']
assert lock.sha(p/'truth.tsv')==m['truth_sha256']
print('REDEEM_C2E_LOCAL_COHORT=PASS requests_only_to_Codon')
PY_PRE
then
  echo 'ERROR: frozen TEST cohort verification failed'
 elif ! (cd "$PKG" && sha256sum -c SHA256_REMOTE.txt); then
  echo 'ERROR: remote-source manifest mismatch'
 elif ! ssh codon "test ! -e '$REMOTE' && test ! -L '$REMOTE'"; then
  echo 'ERROR: TEST remote root already exists; preserve, do not overwrite'
 elif ! ssh codon "mkdir -p '$REMOTE/code' '$REMOTE/input' '$REMOTE/runs'"; then
  echo 'ERROR: failed to create remote root'
 elif ! scp "${CODE_FILES[@]/#/$PKG/}" "$PKG/SHA256_REMOTE.txt" "codon:$REMOTE/code/"; then
  echo 'ERROR: incomplete source transfer; inspect partial directory'
 elif ! scp "$COHORT/requests.tsv" "$COHORT/metadata.json" "codon:$REMOTE/input/"; then
  echo 'ERROR: incomplete TEST request transfer; preserve partial directory'
 elif ! ssh codon "cd '$REMOTE/code' && sha256sum -c SHA256_REMOTE.txt && test -s '$REMOTE/input/requests.tsv' && test ! -e '$REMOTE/input/truth.tsv' && test ! -e '$REMOTE/input/evidence.tsv'"; then
  echo 'ERROR: transferred code/labels preflight failed'
 else
  echo 'REDEEM_C2E_TRANSFER=PASS label_free_only'
 fi
elif [ "$ACTION" = fetch ]; then
 if [ -e "$RESULT" ] || [ -L "$RESULT" ]; then
  echo 'ERROR: RESULT directory already exists; preserve it'
 else
  ID=$(ssh codon "cat '$REMOTE/submitted_job_id.txt'" 2>/dev/null)
  if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
   echo 'ERROR: TEST job ID absent or invalid'
  elif ! ssh codon "test -s '$REMOTE/runs/run_$ID/SUCCESS.txt' && test -s '$REMOTE/runs/run_$ID/identifier_hits.tsv' && test -s '$REMOTE/runs/run_$ID/chemistry_features/candidate_chemistry_features.tsv' && test -s '$REMOTE/runs/run_$ID/chemistry_features/feature_provenance.json'"; then
   echo 'ERROR: TEST feature export is incomplete'
  elif ! mkdir "$RESULT"; then
   echo 'ERROR: could not create local result dir'
  elif ! scp "codon:$REMOTE/runs/run_$ID/identifier_hits.tsv" "$RESULT/identifier_hits.tsv"; then
   echo 'ERROR: missing identifier hits; preserve partial download'
  elif ! scp "codon:$REMOTE/runs/run_$ID/chemistry_features/candidate_chemistry_features.tsv" "$RESULT/candidate_chemistry_features.tsv"; then
   echo 'ERROR: missing chemistry features; preserve partial download'
  elif ! scp "codon:$REMOTE/runs/run_$ID/chemistry_features/feature_provenance.json" "$RESULT/feature_provenance.json"; then
   echo 'ERROR: missing provenance; preserve partial download'
  elif ! "$PY" - "$PKG" "$RESULT" "$ID" <<'PY_CHECK'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import redeem_c2e_lock_20261025 as lock
p=Path(sys.argv[2]);job=sys.argv[3];m=json.loads((p/'feature_provenance.json').read_text())
assert m['status']=='PASS_LABEL_BLIND_C2E_TEST_FEATURE_EXPORT' and m['source_scoring_job_id']==job
assert m['queries']==512 and m['candidate_rows']==131072 and m['reference_truth_accessed'] is False
assert m['model_fit_performed'] is False
assert lock.sha(p/'identifier_hits.tsv')==m['hits_sha256']
assert lock.sha(p/'candidate_chemistry_features.tsv')==m['features_sha256']
print('REDEEM_C2E_FETCH=PASS rows=131072 job='+job)
PY_CHECK
  then
   echo 'ERROR: fetched output SHA/provenance mismatch; preserve'
  else
   echo 'REDEEM_C2E_FETCH=PASS next=single_final_evaluation'
  fi
 fi
elif [ "$ACTION" = evaluate ]; then
 if [ -e "$RESULT/frozen_c2_ranking.tsv" ] || [ -e "$RESULT/final_test_comparison.json" ]; then
  echo 'ERROR: TEST already applied/evaluated; never repeat or overwrite'
 elif [ "$(sha "$MODEL")" != '47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca' ]; then
  echo 'ERROR: frozen model changed'
 elif [ "$(sha "$PKG/redeem_c2_ranker_20261023.py")" != '9128c3db1d2996059afc81d8422b9ec33f7b629d5627b56b352fe67700b8d498' ]; then
  echo 'ERROR: C2 apply code changed'
 elif [ ! -s "$RESULT/feature_provenance.json" ] || [ ! -s "$RESULT/identifier_hits.tsv" ] || [ ! -s "$RESULT/candidate_chemistry_features.tsv" ]; then
  echo 'ERROR: TEST label-blind outputs not fetched'
 elif ! "$PY" - "$PKG" "$RESULT" <<'PY_SAFE'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import redeem_c2e_lock_20261025 as lock
p=Path(sys.argv[2]);m=json.loads((p/'feature_provenance.json').read_text())
assert m['status']=='PASS_LABEL_BLIND_C2E_TEST_FEATURE_EXPORT'
assert lock.sha(p/'identifier_hits.tsv')==m['hits_sha256']
assert lock.sha(p/'candidate_chemistry_features.tsv')==m['features_sha256']
print('REDEEM_C2E_EVAL_PREFLIGHT=PASS')
PY_SAFE
then
  echo 'ERROR: fetched label-blind results failed SHA verification'
 else
  FEATURE_SHA=$('$PY' - "$RESULT/feature_provenance.json" <<'PY_SHA'
import json,sys
print(json.load(open(sys.argv[1]))['features_sha256'])
PY_SHA
)
  if ! "$PY" "$PKG/redeem_c2_ranker_20261023.py" apply \
       --features "$RESULT/candidate_chemistry_features.tsv" \
       --features-sha "$FEATURE_SHA" --queries 512 \
       --model "$MODEL" --output "$RESULT/frozen_c2_ranking.tsv"; then
   echo 'ERROR: frozen ranker apply failed; preserve partial result'
  elif ! "$PY" "$PKG/redeem_c2e_evaluate_20261025.py" \
       --protocol "$PROTOCOL" --model "$MODEL" --catalog "$CATALOG" \
       --metadata "$COHORT/metadata.json" --truth "$COHORT/truth.tsv" \
       --requests "$COHORT/requests.tsv" --hits "$RESULT/identifier_hits.tsv" \
       --features "$RESULT/candidate_chemistry_features.tsv" \
       --provenance "$RESULT/feature_provenance.json" \
       --ranked "$RESULT/frozen_c2_ranking.tsv" --output "$RESULT/final_test_comparison.json"; then
   echo 'ERROR: final TEST evaluation failed; preserve. Do not retune or reselect'
  else
   "$PY" -m json.tool "$RESULT/final_test_comparison.json"
  fi
 fi
else
 echo 'Usage: bash redeem_c2e_local_20261025.sh download|locate|prepare|transfer|fetch|evaluate'
fi
