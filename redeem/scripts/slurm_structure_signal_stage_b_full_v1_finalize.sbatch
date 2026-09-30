#!/bin/bash
#SBATCH --job-name=redeem-stageb-final
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=redeem-structure-stage-b-finalize-%j.out
#SBATCH --error=redeem-structure-stage-b-finalize-%j.err

if [ -z "${SIF:-}" ]; then
  echo "ERROR: SIF is required" >&2
  exit 2
fi
if [ -z "${MANIFEST:-}" ]; then
  echo "ERROR: MANIFEST is required" >&2
  exit 2
fi
if [ -z "${EXPECTED_MANIFEST_FINGERPRINT:-}" ]; then
  echo "ERROR: EXPECTED_MANIFEST_FINGERPRINT is required" >&2
  exit 2
fi
if [ -z "${OUT_ROOT:-}" ]; then
  echo "ERROR: OUT_ROOT is required" >&2
  exit 2
fi

SHARD_COUNT="${SHARD_COUNT:-80}"
EXPECTED_IDENTITIES="${EXPECTED_IDENTITIES:-8000}"
FOUNDATION_ROOT="${FOUNDATION_ROOT:-/nfs/research/juan/DIA/singj/foundation_model}"
SHARDS_ROOT="$OUT_ROOT/shards"
FINAL_OUT="$OUT_ROOT/final"

if [ ! -f "$SIF" ]; then
  echo "ERROR: missing SIF: $SIF" >&2
  exit 2
fi
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: missing manifest: $MANIFEST" >&2
  exit 2
fi
if [ ! -d "$SHARDS_ROOT" ]; then
  echo "ERROR: missing shards root: $SHARDS_ROOT" >&2
  exit 2
fi
if [ -e "$FINAL_OUT" ]; then
  echo "ERROR: final output already exists: $FINAL_OUT" >&2
  exit 2
fi

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "sif=$SIF"
echo "manifest=$MANIFEST"
echo "expected_manifest_fingerprint=$EXPECTED_MANIFEST_FINGERPRINT"
echo "shards_root=$SHARDS_ROOT"
echo "final_out=$FINAL_OUT"
echo "shard_count=$SHARD_COUNT"
echo "expected_identities=$EXPECTED_IDENTITIES"
echo "generation_method=fixed_seed_etkdgv3_no_forcefield"
echo "forcefield_minimization=NO"
echo "partition_scope=TRAIN_frozen_stage_b_manifest_only"
echo "measured_ccs_used=NO"
echo "mobility_labels_used=NO"
echo "dev_labels_used=NO"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"

srun singularity exec \
  --bind "$FOUNDATION_ROOT:$FOUNDATION_ROOT" \
  "$SIF" \
  structure_signal_stage_b_full_v1.py \
    finalize \
    --manifest "$MANIFEST" \
    --expected-manifest-fingerprint "$EXPECTED_MANIFEST_FINGERPRINT" \
    --shards-root "$SHARDS_ROOT" \
    --out-dir "$FINAL_OUT" \
    --shard-count "$SHARD_COUNT" \
    --expected-identities "$EXPECTED_IDENTITIES"
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "Stage B full finalizer failed with rc=$RC" >&2
  exit "$RC"
fi

echo "===== final summary ====="
cat "$FINAL_OUT/stage_b_summary.tsv"
echo "===== by PTM class ====="
cat "$FINAL_OUT/stage_b_by_ptm_class.tsv"
echo "===== by charge ====="
cat "$FINAL_OUT/stage_b_by_charge.tsv"
echo "===== by length ====="
cat "$FINAL_OUT/stage_b_by_length_bin.tsv"
echo "===== failures ====="
cat "$FINAL_OUT/stage_b_failures.tsv"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"
echo "stage_b_full_complete=YES"
echo "stage_b_full_out=$FINAL_OUT"
