#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0753-smoke
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0753-smoke-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0753-smoke-%j.err

module load singularityce/4.1.0

ROOT=/nfs/research/juan/DIA/singj/foundation_model
: "${SIF:?SIF is required}"
: "${OUT:?OUT is required}"

if [ ! -f "$SIF" ]; then
  echo "missing v0.75.3 SIF: $SIF" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "v0.75.3 smoke output must be fresh: $OUT" >&2
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

RESPONSE="$OUT/response.yaml"
HITS="$OUT/hits.tsv"
LOG="$OUT/cli.log"
TIME_FILE="$OUT/cli.time.txt"

printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf 'host=%s\n' "$(hostname)"
printf 'execution_device=CPU\n'
printf 'runtime=v0753_practical_identifier_cli\n'
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
  foundation_identify_practical_v0753 --self-test
self_test_rc=$?
if [ "$self_test_rc" -ne 0 ]; then
  echo "v0.75.3 container self-test failed with rc=$self_test_rc" >&2
  exit "$self_test_rc"
fi

echo "===== file-oriented production smoke ====="
/usr/bin/time -v -o "$TIME_FILE" \
  singularity exec --bind "$BIND" "$SIF" \
    foundation_identify_practical_v0753 \
      --catalog /opt/redeem/share/v0753/catalog.yaml \
      --request /opt/redeem/share/v0753/request.yaml \
      --output-yaml "$RESPONSE" \
      --output-tsv "$HITS" \
  2>&1 | tee "$LOG"
run_rc=${PIPESTATUS[0]}
if [ "$run_rc" -ne 0 ]; then
  exit "$run_rc"
fi

for required in "$RESPONSE" "$HITS" "$LOG" "$TIME_FILE"; do
  if [ ! -s "$required" ]; then
    echo "missing or empty v0.75.3 smoke artifact: $required" >&2
    exit 2
  fi
done

if ! grep -q '^schema: redeem.foundation.practical_identifier.v0752$' "$RESPONSE"; then
  echo "v0.75.3 response schema gate failed" >&2
  exit 2
fi
if ! grep -q '^query_id: codon-smoke-query$' "$RESPONSE"; then
  echo "v0.75.3 query-id gate failed" >&2
  exit 2
fi
if ! grep -q '^candidate_pool: 256$' "$RESPONSE"; then
  echo "v0.75.3 frozen candidate-pool gate failed" >&2
  exit 2
fi
if ! grep -q '^catalog_size: 1$' "$RESPONSE"; then
  echo "v0.75.3 fixture catalog-size gate failed" >&2
  exit 2
fi
if ! grep -q '^candidates_scored: 1$' "$RESPONSE"; then
  echo "v0.75.3 fixture candidates-scored gate failed" >&2
  exit 2
fi
if ! grep -q '^returned_hits: 1$' "$RESPONSE"; then
  echo "v0.75.3 returned-hit gate failed" >&2
  exit 2
fi
if ! grep -q $'^1\t0\tPEPTIDE|z2\t2\t' "$HITS"; then
  echo "v0.75.3 expected top-hit gate failed" >&2
  exit 2
fi
if grep -Eq 'FoundationPartition|TRAIN_HOLDOUT|historical_test|model\.safetensors' "$LOG"; then
  echo "v0.75.3 smoke log unexpectedly references protected/model-selection state" >&2
  exit 2
fi

{
  printf 'key\tvalue\n'
  printf 'status\tPASS\n'
  printf 'runtime\tv0753_practical_identifier_cli\n'
  printf 'scientific_core\tv0751_frozen\n'
  printf 'api_contract\tv0752_frozen\n'
  printf 'candidate_pool\t256\n'
  printf 'target_forcing\tNO\n'
  printf 'protected_data_required\tNO\n'
  printf 'train_holdout_accessed\tNO\n'
  printf 'historical_test_accessed\tNO\n'
} > "$OUT/smoke_summary.tsv"

echo "v0753_practical_identifier_codon_smoke=PASS"
echo "protected_data_required=NO"
echo "train_holdout_accessed=NO"
echo "historical_test_accessed=NO"
echo "v0753_out=$OUT"
