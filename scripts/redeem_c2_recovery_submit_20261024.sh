#!/usr/bin/env bash
# Run on a Codon login node. Recovery only; do not modify the frozen scripts.
# This script deliberately never calls realpath or Path.resolve on the catalog.
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2_train_only_20261023"
CODE="$BASE/code"
TRAIN="$BASE/train_2048"
SIF="$ROOT/containers/redeem-foundation_v0770-streaming-identifier_f7b8246c7730.sif"
SIF_SHA='d46ed2bcc39ed50dec081762788ac60f4f3551658d44b555ce0aacf33504bf22'
CAT="$ROOT/experiments/redeem_pt2017_synthesis_newdev512_20261023/input/catalog.yaml"
CAT_SHA='a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f'
REQUEST_SHA='26dc26a242c5cb039bdba91d7cbfa0ea6b0e14f27777ba1f54f0a8e7bf2c5a25'
REC="$BASE/recovery_catalog_path_20261024"
OLD_ID='6720538'
actual_sha() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }

if [ ! -s "$BASE/submitted_job_id.txt" ] || [ "$(cat "$BASE/submitted_job_id.txt")" != "$OLD_ID" ]; then
  echo 'ERROR: original C2 submission record differs; preserve it and inspect'
elif [ -e "$REC/submitted_job_id.txt" ]; then
  echo 'ERROR: recovery job already has a saved ID; inspect instead of resubmitting'
elif [ ! -s "$CAT" ] || [ "$(actual_sha "$CAT")" != "$CAT_SHA" ]; then
  echo 'ERROR: logical /nfs PT2017 catalog is missing or SHA-256 differs'
elif [ ! -s "$SIF" ] || [ "$(actual_sha "$SIF")" != "$SIF_SHA" ]; then
  echo 'ERROR: frozen identifier SIF missing or SHA-256 differs'
elif [ ! -s "$TRAIN/requests.tsv" ] || [ "$(actual_sha "$TRAIN/requests.tsv")" != "$REQUEST_SHA" ]; then
  echo 'ERROR: frozen TRAIN requests missing or SHA-256 differs'
elif [ -e "$TRAIN/truth.tsv" ] || [ -e "$TRAIN/evidence.tsv" ]; then
  echo 'ERROR: TRAIN truth or evidence unexpectedly present on Codon'
elif ! (cd "$CODE" && sha256sum -c SHA256_SOURCE.txt); then
  echo 'ERROR: frozen C2 source manifest changed'
elif [ ! -s "$CODE/redeem_c2_train_20261023.sbatch" ]; then
  echo 'ERROR: frozen C2 Slurm launcher is missing'
elif ! command -v singularity >/dev/null 2>&1; then
  echo 'ERROR: singularity is not available in PATH'
elif ! singularity exec --bind "$ROOT:$ROOT" "$SIF" /bin/sh -c 'test -r "$1" && test -s "$1"' sh "$CAT"; then
  echo 'ERROR: catalog is not readable at its logical /nfs path INSIDE the frozen SIF; do not submit'
elif [ ! -d "$BASE/fits" ]; then
  echo 'ERROR: original fits directory missing; do not recreate it silently'
else
  ORIGINAL_STATE=$(sacct -X -j "$OLD_ID" --noheader --parsable2 --format=State 2>/dev/null | head -n 1 | cut -d '|' -f 1)
  if [[ "$ORIGINAL_STATE" != FAILED* ]]; then
    echo "ERROR: original job state is not confirmed FAILED: $ORIGINAL_STATE"
  elif ! python3 - "$TRAIN" "$REQUEST_SHA" <<'PY_VERIFY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]); expected=sys.argv[2]
m=json.loads((root/'metadata.json').read_text())
assert m['status']=='PASS_TRAIN_ONLY_SOURCE_PREPARATION'
assert m['selected_queries']==2048 and m['source_partition']=='train'
assert m['source_revision']=='ffde6e4a2925f94dc28c87e33917db19a28f5869'
assert m['protected_test_accessed'] is False
assert m['requests_sha256']==expected
assert hashlib.sha256((root/'requests.tsv').read_bytes()).hexdigest()==expected
print('REDEEM_C2_RECOVERY_TRAIN=PASS queries=2048')
PY_VERIFY
  then
    echo 'ERROR: C2 TRAIN metadata did not pass provenance checks'
  elif [ ! -d "$REC" ] && ! mkdir "$REC"; then
    echo 'ERROR: cannot create separate recovery directory'
  elif [ -e "$REC/submitted_job_id.txt" ]; then
    echo 'ERROR: recovery job already submitted; preserve existing record'
  else
    echo "REDEEM_C2_RECOVERY_CATALOG_LOGICAL=$CAT"
    echo 'REDEEM_C2_RECOVERY_CONTAINER_CATALOG=PASS'
    ID=$(sbatch --parsable \
      --output="$BASE/fits/c2-catalog-recovery-%j.out" \
      --error="$BASE/fits/c2-catalog-recovery-%j.err" \
      --export="ALL,REDEEM_C2_CATALOG=$CAT" \
      "$CODE/redeem_c2_train_20261023.sbatch")
    if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
      echo "ERROR: sbatch did not return numeric job ID: $ID"
    elif ! printf '%s\n' "$ID" > "$REC/submitted_job_id.txt"; then
      echo "ERROR: submitted job $ID but could not save job ID; inspect sacct"
    else
      printf 'original_failed_job_id=%s\ncatalog_logical=%s\ncatalog_sha256=%s\nsif_sha256=%s\nrequests_sha256=%s\n' \
        "$OLD_ID" "$CAT" "$CAT_SHA" "$SIF_SHA" "$REQUEST_SHA" > "$REC/recovery_provenance.txt"
      echo "REDEEM_C2_RECOVERY_SUBMITTED=PASS job_id=$ID"
      echo "REDEEM_C2_RECOVERY_RECORD=$REC/submitted_job_id.txt"
    fi
  fi
fi
