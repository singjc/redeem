#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0750
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0750-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0750-%j.err

module load singularityce/4.1.0

ROOT=/nfs/research/juan/DIA/singj/foundation_model
: "${SIF:?SIF is required}"
: "${RUN_V0260:?RUN_V0260 is required}"
: "${OUT:?OUT is required}"
MODE="${MODE:-smoke}"

if [ "$MODE" != "smoke" ] && [ "$MODE" != "audit" ]; then
  echo "MODE must be smoke or audit" >&2
  exit 2
fi
for path in "$SIF" "$RUN_V0260"; do
  [ -e "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done
if [ -e "$OUT" ]; then
  echo "v0.75 output directory must be fresh: $OUT" >&2
  exit 2
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required" >&2
  exit 127
fi

AUX_OUT="${OUT}.launcher-${SLURM_JOB_ID:-manual}"
if [ -e "$AUX_OUT" ]; then
  echo "v0.75 launcher sidecar must be fresh: $AUX_OUT" >&2
  exit 2
fi
mkdir -p "$AUX_OUT"

LOG="$AUX_OUT/${MODE}.log"
TIME_FILE="$AUX_OUT/${MODE}.time.txt"
BIND="$ROOT:$ROOT"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  BIND="${BIND},${REDEEM_SINGULARITY_BINDS}"
fi

printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf 'host=%s\n' "$(hostname)"
printf 'mode=%s\n' "$MODE"
printf 'sif=%s\n' "$SIF"
printf 'run_v0260=%s\n' "$RUN_V0260"
printf 'experiment=%s\n' "$OUT"
printf 'execution_device=CPU\ntraining_scope=NONE\nselection_scope=DEV_IMPLEMENTATION_REPRODUCTION_ONLY\n'
printf 'candidate_policy=same_charge_nearest256\ntarget_forcing=NO\n'
printf 'v070_checkpoint_required=NO\nv052_checkpoint_required=NO\n'
printf 'train_holdout_consumed=NO\nhistorical_validation_consumed=NO\nhistorical_test_consumed=NO\n'

echo "===== container self-test ====="
singularity exec --bind "$BIND" "$SIF" \
  foundation_identify_charge_mass_geometry_v0750 --self-test
self_test_rc=$?
if [ "$self_test_rc" -ne 0 ]; then
  echo "v0.75 container self-test failed with rc=$self_test_rc" >&2
  exit "$self_test_rc"
fi

echo "===== v0.75 identifier ====="
/usr/bin/time -v -o "$TIME_FILE" \
  singularity exec --bind "$BIND" "$SIF" \
    foundation_identify_charge_mass_geometry_v0750 \
      "$RUN_V0260" \
      "$OUT" \
      "$MODE" \
  2>&1 | tee "$LOG"
run_rc=${PIPESTATUS[0]}

if [ -d "$OUT" ]; then
  find "$AUX_OUT" -mindepth 1 -maxdepth 1 -exec mv -t "$OUT" -- {} +
  rmdir "$AUX_OUT"
else
  echo "v0.75 identifier did not create output root; launcher logs retained at $AUX_OUT" >&2
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
    echo "missing or empty v0.75 output: $OUT/$required" >&2
    exit 2
  fi
done

if ! grep -q $'^candidate_universe_fingerprint\tfnv1a64:a5699533553116e8$' "$OUT/summary.tsv"; then
  echo "v0.75 candidate-universe fingerprint gate failed" >&2
  exit 2
fi
if ! grep -q $'^target_forcing\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 target-forcing certification failed" >&2
  exit 2
fi
if ! grep -q $'^v070_checkpoint_required\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 v0.70 dependency certification failed" >&2
  exit 2
fi
if ! grep -q $'^v052_checkpoint_required\tNO$' "$OUT/summary.tsv"; then
  echo "v0.75 v0.52 dependency certification failed" >&2
  exit 2
fi
if [ "$MODE" = "audit" ]; then
  if ! grep -q $'^il_candidate_coverage\t1.000000000000$' "$OUT/summary.tsv"; then
    echo "v0.75 nearest256 I/L candidate coverage gate failed" >&2
    exit 2
  fi
  if ! grep -q $'^audit_decision\tIMPLEMENTATION_REPRODUCTION_PASS$' "$OUT/summary.tsv"; then
    echo "v0.75 implementation reproduction gate failed" >&2
    exit 2
  fi
fi

echo "train_holdout_consumed=NO"
echo "historical_validation_consumed=NO"
echo "historical_test_consumed=NO"
echo "v0750_complete=YES"
echo "v0750_out=$OUT"
