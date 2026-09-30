#!/bin/bash
#SBATCH --job-name=redeem-stage-b-manifest
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=redeem-structure-stage-b-manifest-%j.out

if [ -z "${SCRIPT:-}" ]; then
  echo "ERROR: SCRIPT is required" >&2
  exit 2
fi
if [ -z "${IDENTITY_INVENTORY:-}" ]; then
  echo "ERROR: IDENTITY_INVENTORY is required" >&2
  exit 2
fi
if [ -z "${OUT:-}" ]; then
  echo "ERROR: OUT is required" >&2
  exit 2
fi

TARGET_SIZE="${TARGET_SIZE:-8000}"
MINIMUM_PER_STRATUM="${MINIMUM_PER_STRATUM:-4}"

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: Stage B manifest script not found: $SCRIPT" >&2
  exit 2
fi
if [ ! -f "$IDENTITY_INVENTORY" ]; then
  echo "ERROR: Stage A identity inventory not found: $IDENTITY_INVENTORY" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "ERROR: output path already exists; use a fresh Stage B manifest output: $OUT" >&2
  exit 2
fi

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "script=$SCRIPT"
echo "identity_inventory=$IDENTITY_INVENTORY"
echo "out=$OUT"
echo "target_size=$TARGET_SIZE"
echo "minimum_per_stratum=$MINIMUM_PER_STRATUM"
echo "partition_scope=TRAIN_identity_inventory_only"
echo "measured_ccs_used_for_selection=NO"
echo "mobility_labels_used_for_selection=NO"
echo "dev_labels_used=NO"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"

echo "===== selector self-test ====="
python3 "$SCRIPT" --self-test
SELFTEST_RC=$?
if [ "$SELFTEST_RC" -ne 0 ]; then
  echo "Stage B manifest selector self-test failed with rc=$SELFTEST_RC" >&2
  exit "$SELFTEST_RC"
fi

echo "===== Stage B manifest selection ====="
srun python3 "$SCRIPT" \
  --identity-inventory "$IDENTITY_INVENTORY" \
  --out-dir "$OUT" \
  --target-size "$TARGET_SIZE" \
  --minimum-per-stratum "$MINIMUM_PER_STRATUM"
RUN_RC=$?
if [ "$RUN_RC" -ne 0 ]; then
  echo "Stage B manifest selection failed with rc=$RUN_RC" >&2
  exit "$RUN_RC"
fi

if [ ! -s "$OUT/stage_b_manifest.tsv" ]; then
  echo "ERROR: Stage B manifest was not created" >&2
  exit 3
fi
if [ ! -s "$OUT/stage_b_summary.tsv" ]; then
  echo "ERROR: Stage B summary was not created" >&2
  exit 3
fi

SELECTED_ROWS=$(($(wc -l < "$OUT/stage_b_manifest.tsv") - 1))
SUMMARY_SELECTED=$(awk -F '\t' '$1 == "selected_identities" {print $2}' "$OUT/stage_b_summary.tsv")
if [ "$SELECTED_ROWS" != "$SUMMARY_SELECTED" ]; then
  echo "ERROR: manifest row count mismatch rows=$SELECTED_ROWS summary=$SUMMARY_SELECTED" >&2
  exit 3
fi

echo "===== summary ====="
cat "$OUT/stage_b_summary.tsv"

echo "===== by charge ====="
cat "$OUT/stage_b_by_charge.tsv"

echo "===== by length ====="
cat "$OUT/stage_b_by_length_bin.tsv"

echo "===== by PTM class ====="
cat "$OUT/stage_b_by_ptm_class.tsv"

echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"
echo "stage_b_manifest_complete=YES"
echo "stage_b_manifest_out=$OUT"
