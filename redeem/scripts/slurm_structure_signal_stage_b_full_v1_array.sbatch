#!/bin/bash
#SBATCH --job-name=redeem-stageb-full
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=redeem-structure-stage-b-full-%A_%a.out
#SBATCH --error=redeem-structure-stage-b-full-%A_%a.err

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
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "ERROR: this launcher must run as a Slurm array" >&2
  exit 2
fi

SHARD_COUNT="${SHARD_COUNT:-80}"
EXPECTED_IDENTITIES="${EXPECTED_IDENTITIES:-8000}"
NUM_CONFORMERS="${NUM_CONFORMERS:-3}"
MICROSTATES="${MICROSTATES:-2}"
MASS_TOLERANCE_DA="${MASS_TOLERANCE_DA:-0.002}"
FOUNDATION_ROOT="${FOUNDATION_ROOT:-/nfs/research/juan/DIA/singj/foundation_model}"
SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
SHARD_OUT="$OUT_ROOT/shards/shard_$(printf '%03d' "$SHARD_INDEX")_of_$(printf '%03d' "$SHARD_COUNT")"

if [ ! -f "$SIF" ]; then
  echo "ERROR: missing SIF: $SIF" >&2
  exit 2
fi
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: missing manifest: $MANIFEST" >&2
  exit 2
fi
if [ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]; then
  echo "ERROR: array index $SHARD_INDEX exceeds shard_count=$SHARD_COUNT" >&2
  exit 2
fi
if [ -e "$SHARD_OUT" ]; then
  echo "ERROR: shard output already exists: $SHARD_OUT" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/shards"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-unknown}"
echo "array_task_id=$SHARD_INDEX"
echo "host=$(hostname)"
echo "sif=$SIF"
echo "manifest=$MANIFEST"
echo "expected_manifest_fingerprint=$EXPECTED_MANIFEST_FINGERPRINT"
echo "out_root=$OUT_ROOT"
echo "shard_out=$SHARD_OUT"
echo "shard_index=$SHARD_INDEX"
echo "shard_count=$SHARD_COUNT"
echo "expected_identities=$EXPECTED_IDENTITIES"
echo "num_conformers=$NUM_CONFORMERS"
echo "microstates=$MICROSTATES"
echo "mass_tolerance_da=$MASS_TOLERANCE_DA"
echo "generation_method=fixed_seed_etkdgv3_no_forcefield"
echo "forcefield_minimization=NO"
echo "partition_scope=TRAIN_frozen_stage_b_manifest_only"
echo "measured_ccs_used=NO"
echo "mobility_labels_used=NO"
echo "dev_labels_used=NO"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"

echo "===== container gate ====="
singularity exec \
  "$SIF" \
  python -c 'import rdkit; print("rdkit_version=" + rdkit.__version__)'
RDKIT_RC=$?
if [ "$RDKIT_RC" -ne 0 ]; then
  exit "$RDKIT_RC"
fi

echo "===== full Stage B shard ====="
srun singularity exec \
  --bind "$FOUNDATION_ROOT:$FOUNDATION_ROOT" \
  "$SIF" \
  structure_signal_stage_b_full_v1.py \
    shard \
    --manifest "$MANIFEST" \
    --expected-manifest-fingerprint "$EXPECTED_MANIFEST_FINGERPRINT" \
    --out-dir "$SHARD_OUT" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --expected-identities "$EXPECTED_IDENTITIES" \
    --num-conformers "$NUM_CONFORMERS" \
    --microstates "$MICROSTATES" \
    --mass-tolerance-da "$MASS_TOLERANCE_DA"
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "Stage B full shard failed with rc=$RC" >&2
  exit "$RC"
fi

cat "$SHARD_OUT/shard_summary.tsv"
echo "stage_b_full_shard_complete=YES"
