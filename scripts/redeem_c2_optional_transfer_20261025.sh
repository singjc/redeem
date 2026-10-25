#!/usr/bin/env bash
# Run from laptop. Exact source and trained model are verified before transfer.
# Remote destination is new and guarded against overwrites.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$HOME/Downloads/redeem_c2_frozen_ranker_20261023/model.json"
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2_optional_inference_20261025"
CODE=(
 redeem_identify_optional_c2_20261025.py
 redeem_c2_train_features_20261023.py
 redeem_c2_ranker_20261023.py
 redeem_c2_optional_smoke_20261025.sbatch
 redeem_c2_optional_submit_20261025.sh
 redeem_c2_optional_inspect_20261025.sh
)
if [ ! -s "$MODEL" ]; then
  echo 'ERROR: frozen model missing at expected laptop location'
elif ! printf '%s  %s\n' '47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca' "$MODEL" | sha256sum -c -; then
  echo 'ERROR: trained model checksum mismatch'
elif ! (cd "$DIR" && sha256sum -c SHA256SUMS.txt); then
  echo 'ERROR: optional inference package checksum mismatch'
elif ! (cd "$DIR" && sha256sum -c SHA256_CODE.txt); then
  echo 'ERROR: Codon source subset manifest mismatch'
elif ! ssh codon "test ! -e '$BASE' && test ! -L '$BASE'"; then
  echo 'ERROR: remote experiment path already exists; preserve it'
elif ! ssh codon "mkdir -p '$BASE/code' '$BASE/model' '$BASE/runs'"; then
  echo 'ERROR: remote experiment directory creation failed'
elif ! scp "${CODE[@]/#/$DIR/}" "$DIR/SHA256_CODE.txt" "codon:$BASE/code/"; then
  echo 'ERROR: failed to transfer code files; preserve partial remote state'
elif ! scp "$MODEL" "codon:$BASE/model/model.json"; then
  echo 'ERROR: failed to transfer frozen model; preserve partial remote state'
elif ! ssh codon "cd '$BASE/code' && sha256sum -c SHA256_CODE.txt && printf '%s  %s\\n' '47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca' '$BASE/model/model.json' | sha256sum -c -"; then
  echo 'ERROR: Codon SHA verification failed'
else
  echo 'REDEEM_C2_OPTIONAL_TRANSFER=PASS label_free_only'
  echo "NEXT_CODON_COMMAND=bash '$BASE/code/redeem_c2_optional_submit_20261025.sh'"
fi
