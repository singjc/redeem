#!/usr/bin/env bash
#SBATCH --job-name=redeem-structure-stage-a
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=redeem-structure-stage-a-%j.out

module load singularityce/4.1.0

for name in SIF RUN FOUNDATION_ROOT OUT; do
  eval "value=\${$name:-}"
  if [ -z "$value" ]; then
    echo "Set $name for structure_signal_feasibility_v1 Stage A" >&2
    exit 2
  fi
done

for file in "$SIF" "$RUN"; do
  if [ ! -f "$file" ]; then
    echo "Missing required file: $file" >&2
    exit 2
  fi
done

if [ ! -d "$FOUNDATION_ROOT" ]; then
  echo "Missing foundation root: $FOUNDATION_ROOT" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "Stage A OUT must be fresh: $OUT" >&2
  exit 2
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required" >&2
  exit 127
fi

ENGINE="$(command -v singularity)"
BIND="${FOUNDATION_ROOT}:${FOUNDATION_ROOT}"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  BIND="${BIND},${REDEEM_SINGULARITY_BINDS}"
fi

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

LOG="${OUT}.stdout.log"

echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "sif=$SIF"
echo "run=$RUN"
echo "foundation_root=$FOUNDATION_ROOT"
echo "out=$OUT"
echo "partition_scope=TRAIN_only"
echo "measured_ccs_used_for_analysis=NO"
echo "dev_labels_used=NO"
echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"

echo "===== container self-test ====="
"$ENGINE" exec --bind "$BIND" "$SIF" \
  foundation_audit_structure_signal_feasibility_v1 --self-test
self_test_rc=$?
if [ "$self_test_rc" -ne 0 ]; then
  echo "Container self-test failed with rc=$self_test_rc" >&2
  exit "$self_test_rc"
fi

echo "===== TRAIN Stage A inventory ====="
srun "$ENGINE" exec --bind "$BIND" "$SIF" \
  foundation_audit_structure_signal_feasibility_v1 \
    "$RUN" \
    "$OUT" \
  2>&1 | tee "$LOG"
audit_rc=${PIPESTATUS[0]}
if [ "$audit_rc" -ne 0 ]; then
  echo "Stage A audit failed with rc=$audit_rc" >&2
  exit "$audit_rc"
fi

for required in \
  summary.tsv \
  identity_inventory.tsv \
  ptm_inventory.tsv \
  coverage_by_charge.tsv \
  coverage_by_length_bin.tsv \
  failure_reasons.tsv \
  failure_by_ptm.tsv \
  failure_by_charge.tsv \
  failure_by_length_bin.tsv \
  report.md; do
  if [ ! -s "$OUT/$required" ]; then
    echo "Missing or empty Stage A output: $OUT/$required" >&2
    exit 2
  fi
done

if ! grep -q $'^partition_scope\tTRAIN_only$' "$OUT/summary.tsv"; then
  echo "Stage A summary does not certify TRAIN-only scope" >&2
  exit 2
fi
if ! grep -q $'^measured_ccs_used_for_analysis\tNO$' "$OUT/summary.tsv"; then
  echo "Stage A summary does not certify measured_ccs_used_for_analysis=NO" >&2
  exit 2
fi
if ! grep -q $'^dev_labels_used\tNO$' "$OUT/summary.tsv"; then
  echo "Stage A summary does not certify dev_labels_used=NO" >&2
  exit 2
fi
if ! grep -q $'^holdout_used\tNO$' "$OUT/summary.tsv"; then
  echo "Stage A summary does not certify holdout_used=NO" >&2
  exit 2
fi

echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"
echo "stage_a_complete=YES"
echo "stage_a_out=$OUT"
