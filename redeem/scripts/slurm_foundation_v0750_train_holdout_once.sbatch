#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0750-holdout
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0750-holdout-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0750-holdout-%j.err

module load singularityce/4.1.0

ROOT=/nfs/research/juan/DIA/singj/foundation_model
: "${SIF:?SIF is required}"
: "${RUN_V0260:?RUN_V0260 is required}"
: "${OUT:?OUT is required}"
: "${CONFIRM_TRAIN_HOLDOUT_CONSUMPTION:?Set CONFIRM_TRAIN_HOLDOUT_CONSUMPTION=YES to run the one-time protected evaluation}"

if [ "$CONFIRM_TRAIN_HOLDOUT_CONSUMPTION" != "YES" ]; then
  echo "Refusing protected evaluation: CONFIRM_TRAIN_HOLDOUT_CONSUMPTION must equal YES" >&2
  exit 2
fi
for path in "$SIF" "$RUN_V0260"; do
  [ -e "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done
if [ -e "$OUT" ]; then
  echo "v0.75 HOLDOUT output directory must be fresh: $OUT" >&2
  exit 2
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required" >&2
  exit 127
fi

AUX_OUT="${OUT}.launcher-${SLURM_JOB_ID:-manual}"
if [ -e "$AUX_OUT" ]; then
  echo "v0.75 HOLDOUT launcher sidecar must be fresh: $AUX_OUT" >&2
  exit 2
fi
mkdir -p "$AUX_OUT"

LOG="$AUX_OUT/holdout.log"
TIME_FILE="$AUX_OUT/holdout.time.txt"
BIND="$ROOT:$ROOT"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  BIND="${BIND},${REDEEM_SINGULARITY_BINDS}"
fi

printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf 'host=%s\n' "$(hostname)"
printf 'mode=holdout\n'
printf 'evaluation_partition=TRAIN_HOLDOUT_ONCE\n'
printf 'sif=%s\n' "$SIF"
printf 'run_v0260=%s\n' "$RUN_V0260"
printf 'experiment=%s\n' "$OUT"
printf 'execution_device=CPU\ntraining_scope=NONE\nselection_scope=FROZEN_IMPLEMENTATION_CONFIRMATION_ONLY\n'
printf 'candidate_policy=same_charge_nearest256\ntarget_forcing=NO\n'
printf 'v070_checkpoint_required=NO\nv052_checkpoint_required=NO\n'
printf 'train_holdout_consumed=YES\ntrain_holdout_consumed_for_selection=NO\n'
printf 'historical_validation_consumed=NO\nhistorical_test_consumed=NO\n'

echo "===== container self-test ====="
singularity exec --bind "$BIND" "$SIF" \
  foundation_identify_charge_mass_geometry_v0750 --self-test
self_test_rc=$?
if [ "$self_test_rc" -ne 0 ]; then
  echo "v0.75 container self-test failed with rc=$self_test_rc" >&2
  exit "$self_test_rc"
fi

echo "===== one-time v0.75 TRAIN-HOLDOUT evaluation ====="
/usr/bin/time -v -o "$TIME_FILE" \
  singularity exec --bind "$BIND" "$SIF" \
    foundation_identify_charge_mass_geometry_v0750 \
      "$RUN_V0260" \
      "$OUT" \
      holdout \
      CONFIRM_TRAIN_HOLDOUT_ONCE \
  2>&1 | tee "$LOG"
run_rc=${PIPESTATUS[0]}

if [ -d "$OUT" ]; then
  find "$AUX_OUT" -mindepth 1 -maxdepth 1 -exec mv -t "$OUT" -- {} +
  rmdir "$AUX_OUT"
else
  echo "v0.75 HOLDOUT evaluator did not create output root; launcher logs retained at $AUX_OUT" >&2
fi

if [ "$run_rc" -ne 0 ]; then
  exit "$run_rc"
fi

for required in \
  summary.tsv \
  identifier_metrics.tsv \
  candidate_catalog.tsv \
  query_diagnostics.tsv \
  ranked_candidates.tsv \
  metadata.yaml; do
  if [ ! -s "$OUT/$required" ]; then
    echo "missing or empty v0.75 HOLDOUT output: $OUT/$required" >&2
    exit 2
  fi
done

if ! grep -q $'^evaluation_partition\tTRAIN_HOLDOUT_ONCE$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT partition certification failed" >&2
  exit 2
fi
if ! grep -q $'^corpus_fingerprint\tfnv1a64:a2a6f57d31064ba6$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT corpus fingerprint gate failed" >&2
  exit 2
fi
if ! grep -q $'^benchmark_manifest_fingerprint\tfnv1a64:2133c039625f77df$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT benchmark fingerprint gate failed" >&2
  exit 2
fi
if ! grep -q $'^evaluation_queries\t2048$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT frozen query-count gate failed" >&2
  exit 2
fi
if ! grep -q $'^candidate_pool\t256$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT nearest256 policy certification failed" >&2
  exit 2
fi
if ! grep -q $'^target_forcing\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT target-forcing certification failed" >&2
  exit 2
fi
if ! grep -q $'^v070_checkpoint_required\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT v0.70 dependency certification failed" >&2
  exit 2
fi
if ! grep -q $'^v052_checkpoint_required\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT v0.52 dependency certification failed" >&2
  exit 2
fi
if ! grep -q $'^train_holdout_consumed\tYES$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT consumption certification failed" >&2
  exit 2
fi
if ! grep -q $'^train_holdout_consumed_for_selection\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT selection firewall failed" >&2
  exit 2
fi
if ! grep -q $'^dev_labels_used_for_analysis\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 DEV-label analysis firewall failed" >&2
  exit 2
fi
if ! grep -q $'^dev_partition_used_for_selection\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 DEV-selection firewall failed" >&2
  exit 2
fi
if ! grep -q $'^historical_validation_consumed\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 historical VALIDATION firewall failed" >&2
  exit 2
fi
if ! grep -q $'^historical_test_consumed\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 historical TEST firewall failed" >&2
  exit 2
fi
if ! grep -Eq $'^audit_decision\t(HOLDOUT_CONFIRMATION_PASS|HOLDOUT_CONFIRMATION_FAIL_DO_NOT_TUNE)$' "$OUT/summary.tsv"; then
  echo "v0.75 HOLDOUT decision marker missing" >&2
  exit 2
fi

HOLDOUT_DECISION=$(awk -F '\t' '$1 == "audit_decision" {print $2}' "$OUT/summary.tsv")
echo "train_holdout_consumed=YES"
echo "train_holdout_consumed_for_selection=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"
echo "v0750_holdout_evaluation_complete=YES"
echo "v0750_holdout_decision=$HOLDOUT_DECISION"
echo "v0750_holdout_out=$OUT"
