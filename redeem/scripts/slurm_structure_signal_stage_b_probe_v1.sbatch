#!/bin/bash
#SBATCH --job-name=redeem-stageb-probe
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=redeem-structure-stage-b-probe-%j.out
#SBATCH --error=redeem-structure-stage-b-probe-%j.err

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
if [ -z "${OUT:-}" ]; then
  echo "ERROR: OUT is required" >&2
  exit 2
fi

PROBE_SIZE="${PROBE_SIZE:-42}"
MATCHED_CHARGE_PAIRS="${MATCHED_CHARGE_PAIRS:-4}"
MINIMUM_PER_PTM_CLASS="${MINIMUM_PER_PTM_CLASS:-3}"
NUM_CONFORMERS="${NUM_CONFORMERS:-3}"
MICROSTATES="${MICROSTATES:-2}"
MAX_MINIMIZE_ITERS="${MAX_MINIMIZE_ITERS:-250}"
REPRODUCIBILITY_CHECKS="${REPRODUCIBILITY_CHECKS:-6}"
MASS_TOLERANCE_DA="${MASS_TOLERANCE_DA:-0.002}"
FOUNDATION_ROOT="${FOUNDATION_ROOT:-/nfs/research/juan/DIA/singj/foundation_model}"

if [ ! -f "$SIF" ]; then
  echo "ERROR: missing SIF: $SIF" >&2
  exit 2
fi
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: missing frozen Stage B manifest: $MANIFEST" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "ERROR: OUT already exists: $OUT" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "sif=$SIF"
echo "manifest=$MANIFEST"
echo "expected_manifest_fingerprint=$EXPECTED_MANIFEST_FINGERPRINT"
echo "out=$OUT"
echo "probe_size=$PROBE_SIZE"
echo "matched_charge_pairs=$MATCHED_CHARGE_PAIRS"
echo "minimum_per_ptm_class=$MINIMUM_PER_PTM_CLASS"
echo "num_conformers=$NUM_CONFORMERS"
echo "microstates=$MICROSTATES"
echo "max_minimize_iters=$MAX_MINIMIZE_ITERS"
echo "reproducibility_checks=$REPRODUCIBILITY_CHECKS"
echo "mass_tolerance_da=$MASS_TOLERANCE_DA"
echo "partition_scope=TRAIN_frozen_stage_b_manifest_only"
echo "measured_ccs_used=NO"
echo "mobility_labels_used=NO"
echo "dev_labels_used=NO"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"

echo "===== container gates ====="
singularity exec \
  "$SIF" \
  python -c 'import rdkit; print("rdkit_version=" + rdkit.__version__)'
RDKIT_RC=$?
if [ "$RDKIT_RC" -ne 0 ]; then
  echo "Stage B probe RDKit import gate failed with rc=$RDKIT_RC" >&2
  exit "$RDKIT_RC"
fi

singularity exec \
  "$SIF" \
  structure_signal_stage_b_probe_v1.py --self-test
SELFTEST_RC=$?
if [ "$SELFTEST_RC" -ne 0 ]; then
  echo "Stage B probe self-test gate failed with rc=$SELFTEST_RC" >&2
  exit "$SELFTEST_RC"
fi

echo "===== tiny Stage B engine probe ====="
srun singularity exec \
  --bind "$FOUNDATION_ROOT:$FOUNDATION_ROOT" \
  "$SIF" \
  structure_signal_stage_b_probe_v1.py \
    --manifest "$MANIFEST" \
    --expected-manifest-fingerprint "$EXPECTED_MANIFEST_FINGERPRINT" \
    --out-dir "$OUT" \
    --probe-size "$PROBE_SIZE" \
    --matched-charge-pairs "$MATCHED_CHARGE_PAIRS" \
    --minimum-per-ptm-class "$MINIMUM_PER_PTM_CLASS" \
    --num-conformers "$NUM_CONFORMERS" \
    --microstates "$MICROSTATES" \
    --max-minimize-iters "$MAX_MINIMIZE_ITERS" \
    --reproducibility-checks "$REPRODUCIBILITY_CHECKS" \
    --mass-tolerance-da "$MASS_TOLERANCE_DA" \
  2>&1 | tee "${OUT}.stdout.log"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then
  echo "Stage B tiny engine probe failed with rc=$RC" >&2
  exit "$RC"
fi

echo "===== summary ====="
cat "$OUT/probe_summary.tsv"

echo "===== by PTM class ====="
cat "$OUT/probe_by_ptm_class.tsv"

echo "===== by charge ====="
cat "$OUT/probe_by_charge.tsv"

echo "===== by length ====="
cat "$OUT/probe_by_length_bin.tsv"

echo "===== failures ====="
cat "$OUT/probe_failures.tsv"

echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"
echo "stage_b_tiny_probe_complete=YES"
echo "stage_b_tiny_probe_out=$OUT"
