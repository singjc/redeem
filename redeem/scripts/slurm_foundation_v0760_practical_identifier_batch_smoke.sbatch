#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0760-smoke
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0760-smoke-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0760-smoke-%j.err

module load singularityce/4.1.0

ROOT=/nfs/research/juan/DIA/singj/foundation_model
: "${SIF:?SIF is required}"
: "${OUT:?OUT is required}"

if [ ! -f "$SIF" ]; then
  echo "missing v0.76.0 SIF: $SIF" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "v0.76.0 smoke output must be fresh: $OUT" >&2
  exit 2
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required" >&2
  exit 127
fi

mkdir -p "$OUT"
BIND="$ROOT:$ROOT"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  BIND="${BIND},${REDEEM_SINGULARITY_BINDS}"
fi

CATALOG="$OUT/catalog.yaml"
RESPONSE="$OUT/batch_response.yaml"
HITS="$OUT/batch_hits.tsv"
MATERIALIZE_LOG="$OUT/materialize.log"
BATCH_LOG="$OUT/batch.log"
TIME_FILE="$OUT/batch.time.txt"

printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf 'host=%s\n' "$(hostname)"
printf 'execution_device=CPU\n'
printf 'runtime=v0760_practical_identifier_batch\n'
printf 'scientific_core=v0751_frozen\n'
printf 'api_contract=v0752_frozen\n'
printf 'candidate_policy=same_charge_nearest256\n'
printf 'target_forcing=NO\n'
printf 'protected_data_required=NO\n'
printf 'train_holdout_accessed=NO\n'
printf 'historical_test_accessed=NO\n'
printf 'sif=%s\n' "$SIF"
printf 'out=%s\n' "$OUT"

echo "===== container self-test ====="
singularity exec --bind "$BIND" "$SIF" \
  foundation_identify_batch_v0760 --self-test
self_test_rc=$?
if [ "$self_test_rc" -ne 0 ]; then
  echo "v0.76.0 container self-test failed with rc=$self_test_rc" >&2
  exit "$self_test_rc"
fi

echo "===== search-space materialization smoke ====="
singularity exec --bind "$BIND" "$SIF" \
  foundation_identify_batch_v0760 materialize-catalog \
    --search-space /opt/redeem/share/v0760/search_space.yaml \
    --output-catalog "$CATALOG" \
  2>&1 | tee "$MATERIALIZE_LOG"
materialize_rc=${PIPESTATUS[0]}
if [ "$materialize_rc" -ne 0 ]; then
  exit "$materialize_rc"
fi

echo "===== ordered batch identification smoke ====="
/usr/bin/time -v -o "$TIME_FILE" \
  singularity exec --bind "$BIND" "$SIF" \
    foundation_identify_batch_v0760 identify-batch \
      --catalog "$CATALOG" \
      --batch /opt/redeem/share/v0760/batch.yaml \
      --output-yaml "$RESPONSE" \
      --output-tsv "$HITS" \
  2>&1 | tee "$BATCH_LOG"
batch_rc=${PIPESTATUS[0]}
if [ "$batch_rc" -ne 0 ]; then
  exit "$batch_rc"
fi

for required in "$CATALOG" "$RESPONSE" "$HITS" "$MATERIALIZE_LOG" "$BATCH_LOG" "$TIME_FILE"; do
  if [ ! -s "$required" ]; then
    echo "missing or empty v0.76.0 smoke artifact: $required" >&2
    exit 2
  fi
done

if ! grep -q '^schema: redeem.foundation.practical_identifier.v0752$' "$CATALOG"; then
  echo "v0.76.0 materialized catalog schema gate failed" >&2
  exit 2
fi
if ! grep -q '^schema: redeem.foundation.practical_identifier.batch_response.v0760$' "$RESPONSE"; then
  echo "v0.76.0 batch response schema gate failed" >&2
  exit 2
fi
if ! grep -q '^catalog_size: 3$' "$RESPONSE"; then
  echo "v0.76.0 fixture catalog-size gate failed" >&2
  exit 2
fi
if ! grep -q '^request_count: 2$' "$RESPONSE"; then
  echo "v0.76.0 request-count gate failed" >&2
  exit 2
fi
if ! grep -q '^total_candidates_scored: 3$' "$RESPONSE"; then
  echo "v0.76.0 candidate-count gate failed" >&2
  exit 2
fi
if ! grep -q 'batch-q2' "$HITS" || ! grep -q 'batch-q3' "$HITS"; then
  echo "v0.76.0 flattened hits gate failed" >&2
  exit 2
fi
if grep -Eq 'FoundationPartition|TRAIN_HOLDOUT|historical_test|model\.safetensors' "$MATERIALIZE_LOG" "$BATCH_LOG"; then
  echo "v0.76.0 smoke unexpectedly references protected/model-selection state" >&2
  exit 2
fi

{
  printf 'key\tvalue\n'
  printf 'status\tPASS\n'
  printf 'runtime\tv0760_practical_identifier_batch\n'
  printf 'scientific_core\tv0751_frozen\n'
  printf 'api_contract\tv0752_frozen\n'
  printf 'candidate_pool\t256\n'
  printf 'materialized_candidates\t3\n'
  printf 'batch_requests\t2\n'
  printf 'target_forcing\tNO\n'
  printf 'protected_data_required\tNO\n'
  printf 'train_holdout_accessed\tNO\n'
  printf 'historical_test_accessed\tNO\n'
} > "$OUT/smoke_summary.tsv"

echo "v0760_practical_identifier_batch_codon_smoke=PASS"
echo "protected_data_required=NO"
echo "train_holdout_accessed=NO"
echo "historical_test_accessed=NO"
echo "v0760_out=$OUT"
