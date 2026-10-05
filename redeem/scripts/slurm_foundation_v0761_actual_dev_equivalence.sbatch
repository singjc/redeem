#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0761-actual-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0761-actual-dev-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0761-actual-dev-%j.err

module load singularityce/4.1.0

ROOT=/nfs/research/juan/DIA/singj/foundation_model
: "${SIF:?SIF is required}"
: "${RUN_V0260:?RUN_V0260 is required}"
: "${REF:?REF is required}"
: "${OUT:?OUT is required}"

if [ ! -f "$SIF" ]; then
  echo "missing v0.76.1 SIF: $SIF" >&2
  exit 2
fi
if [ ! -f "$RUN_V0260" ]; then
  echo "missing run_v0260: $RUN_V0260" >&2
  exit 2
fi
if [ ! -s "$REF/candidate_catalog.tsv" ] || \
   [ ! -s "$REF/query_diagnostics.tsv" ] || \
   [ ! -s "$REF/ranked_candidates.tsv" ]; then
  echo "missing frozen v0.75 DEV reference artifacts under $REF" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "v0.76.1 output must be fresh: $OUT" >&2
  exit 2
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required" >&2
  exit 127
fi

mkdir -p "$OUT"
EXPORT="$OUT/export"
CATALOG="$OUT/catalog.yaml"
RESPONSE="$OUT/batch_response.yaml"
HITS="$OUT/batch_hits.tsv"
EXPORT_LOG="$OUT/export.log"
MATERIALIZE_LOG="$OUT/materialize.log"
BATCH_LOG="$OUT/batch.log"
EXPORT_TIME="$OUT/export.time.txt"
BATCH_TIME="$OUT/batch.time.txt"
REF_QUERY_MANIFEST="$OUT/reference_query_manifest.tsv"
REF_NORMALIZED="$OUT/reference_ranked_normalized.tsv"
PROD_NORMALIZED="$OUT/production_ranked_normalized.tsv"

BIND="$ROOT:$ROOT"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  BIND="${BIND},${REDEEM_SINGULARITY_BINDS}"
fi

printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf 'host=%s\n' "$(hostname)"
printf 'execution_device=CPU\n'
printf 'runtime=v0761_actual_dev_production_equivalence\n'
printf 'scientific_core=v0751_frozen\n'
printf 'api_contract=v0752_frozen\n'
printf 'batch_contract=v0760_frozen\n'
printf 'evaluation_partition=DEV_ENGINEERING_EQUIVALENCE_ONLY\n'
printf 'candidate_policy=same_charge_nearest256\n'
printf 'target_forcing=NO\n'
printf 'train_holdout_accessed=NO\n'
printf 'historical_test_accessed=NO\n'
printf 'sif=%s\n' "$SIF"
printf 'run_v0260=%s\n' "$RUN_V0260"
printf 'reference=%s\n' "$REF"
printf 'out=%s\n' "$OUT"

echo "===== container self-tests ====="
singularity exec --bind "$BIND" "$SIF" foundation_export_dev_batch_v0761 --self-test
export_self_test_rc=$?
if [ "$export_self_test_rc" -ne 0 ]; then
  exit "$export_self_test_rc"
fi
singularity exec --bind "$BIND" "$SIF" foundation_identify_batch_v0760 --self-test
batch_self_test_rc=$?
if [ "$batch_self_test_rc" -ne 0 ]; then
  exit "$batch_self_test_rc"
fi

echo "===== actual DEV export ====="
/usr/bin/time -v -o "$EXPORT_TIME" \
  singularity exec --bind "$BIND" "$SIF" \
    foundation_export_dev_batch_v0761 \
      "$RUN_V0260" \
      "$EXPORT" \
  2>&1 | tee "$EXPORT_LOG"
export_rc=${PIPESTATUS[0]}
if [ "$export_rc" -ne 0 ]; then
  exit "$export_rc"
fi

for required in \
  "$EXPORT/search_space.yaml" \
  "$EXPORT/batch.yaml" \
  "$EXPORT/candidate_catalog.tsv" \
  "$EXPORT/query_manifest.tsv" \
  "$EXPORT/export_metadata.yaml" \
  "$EXPORT/export_summary.tsv" \
  "$EXPORT_LOG" \
  "$EXPORT_TIME"; do
  if [ ! -s "$required" ]; then
    echo "missing or empty v0.76.1 export artifact: $required" >&2
    exit 2
  fi
done

if ! grep -q $'^candidate_universe\t63332$' "$EXPORT/export_summary.tsv"; then
  echo "v0.76.1 actual DEV candidate-universe gate failed" >&2
  exit 2
fi
if ! grep -q $'^candidate_universe_fingerprint\tfnv1a64:a5699533553116e8$' "$EXPORT/export_summary.tsv"; then
  echo "v0.76.1 candidate-universe fingerprint gate failed" >&2
  exit 2
fi
if ! grep -q $'^query_count\t512$' "$EXPORT/export_summary.tsv"; then
  echo "v0.76.1 query-count gate failed" >&2
  exit 2
fi
if ! grep -q $'^query_cohort_fingerprint\tfnv1a64:bc6d4b8b3090af2b$' "$EXPORT/export_summary.tsv"; then
  echo "v0.76.1 query-cohort fingerprint gate failed" >&2
  exit 2
fi

if ! cmp -s "$REF/candidate_catalog.tsv" "$EXPORT/candidate_catalog.tsv"; then
  echo "v0.76.1 candidate catalog is not byte-identical to frozen v0.75 DEV reference" >&2
  exit 2
fi

cut -f1-6 "$REF/query_diagnostics.tsv" > "$REF_QUERY_MANIFEST"
if ! cmp -s "$REF_QUERY_MANIFEST" "$EXPORT/query_manifest.tsv"; then
  echo "v0.76.1 query cohort/order is not byte-identical to frozen v0.75 DEV reference" >&2
  exit 2
fi

echo "===== production catalog materialization ====="
singularity exec --bind "$BIND" "$SIF" \
  foundation_identify_batch_v0760 materialize-catalog \
    --search-space "$EXPORT/search_space.yaml" \
    --output-catalog "$CATALOG" \
  2>&1 | tee "$MATERIALIZE_LOG"
materialize_rc=${PIPESTATUS[0]}
if [ "$materialize_rc" -ne 0 ]; then
  exit "$materialize_rc"
fi

echo "===== production batch identification ====="
/usr/bin/time -v -o "$BATCH_TIME" \
  singularity exec --bind "$BIND" "$SIF" \
    foundation_identify_batch_v0760 identify-batch \
      --catalog "$CATALOG" \
      --batch "$EXPORT/batch.yaml" \
      --output-yaml "$RESPONSE" \
      --output-tsv "$HITS" \
  2>&1 | tee "$BATCH_LOG"
batch_rc=${PIPESTATUS[0]}
if [ "$batch_rc" -ne 0 ]; then
  exit "$batch_rc"
fi

for required in "$CATALOG" "$RESPONSE" "$HITS" "$MATERIALIZE_LOG" "$BATCH_LOG" "$BATCH_TIME"; do
  if [ ! -s "$required" ]; then
    echo "missing or empty v0.76.1 production artifact: $required" >&2
    exit 2
  fi
done

if ! grep -q '^schema: redeem.foundation.practical_identifier.batch_response.v0760$' "$RESPONSE"; then
  echo "v0.76.1 production batch response schema gate failed" >&2
  exit 2
fi
if ! grep -q '^catalog_size: 63332$' "$RESPONSE"; then
  echo "v0.76.1 production catalog-size gate failed" >&2
  exit 2
fi
if ! grep -q '^request_count: 512$' "$RESPONSE"; then
  echo "v0.76.1 production request-count gate failed" >&2
  exit 2
fi
candidate_pool_lines=$(grep -c '^  candidate_pool: 256$' "$RESPONSE")
if [ "$candidate_pool_lines" -ne 512 ]; then
  echo "v0.76.1 fixed nearest256 response gate failed: observed=$candidate_pool_lines expected=512" >&2
  exit 2
fi

awk -F '\t' 'BEGIN { OFS="\t"; print "query_id","rank","candidate_index","key","charge","theoretical_neutral_mass","absolute_neutral_mass_error","geometry_score" }
NR > 1 { printf "%s\t%s\t%s\t%s\t%s\t%.8f\t%.8f\t%.8f\n", $1,$2,$3,$4,$6,$7,$8,$9 }' \
  "$REF/ranked_candidates.tsv" > "$REF_NORMALIZED"

awk -F '\t' 'BEGIN { OFS="\t"; print "query_id","rank","candidate_index","key","charge","theoretical_neutral_mass","absolute_neutral_mass_error","geometry_score" }
NR > 1 { printf "%s\t%s\t%s\t%s\t%s\t%.8f\t%.8f\t%.8f\n", $1,$2,$3,$4,$5,$6,$7,$8 }' \
  "$HITS" > "$PROD_NORMALIZED"

if ! cmp -s "$REF_NORMALIZED" "$PROD_NORMALIZED"; then
  echo "v0.76.1 production rankings differ from frozen actual DEV reference" >&2
  diff -u "$REF_NORMALIZED" "$PROD_NORMALIZED" | head -120 >&2
  exit 2
fi

if grep -Eq 'TRAIN_HOLDOUT|historical_test|model\.safetensors|target_forcing=YES' \
  "$EXPORT_LOG" "$MATERIALIZE_LOG" "$BATCH_LOG"; then
  echo "v0.76.1 unexpectedly references protected/model-selection state" >&2
  exit 2
fi

reference_rows=$(($(wc -l < "$REF_NORMALIZED") - 1))
production_rows=$(($(wc -l < "$PROD_NORMALIZED") - 1))
if [ "$reference_rows" -ne "$production_rows" ]; then
  echo "v0.76.1 ranked-row count mismatch: reference=$reference_rows production=$production_rows" >&2
  exit 2
fi

{
  printf 'key\tvalue\n'
  printf 'status\tPASS\n'
  printf 'runtime\tv0761_actual_dev_production_equivalence\n'
  printf 'scientific_core\tv0751_frozen\n'
  printf 'api_contract\tv0752_frozen\n'
  printf 'batch_contract\tv0760_frozen\n'
  printf 'candidate_universe\t63332\n'
  printf 'candidate_universe_fingerprint\tfnv1a64:a5699533553116e8\n'
  printf 'query_count\t512\n'
  printf 'query_cohort_fingerprint\tfnv1a64:bc6d4b8b3090af2b\n'
  printf 'candidate_pool\t256\n'
  printf 'candidate_catalog_byte_identical\tYES\n'
  printf 'query_manifest_byte_identical\tYES\n'
  printf 'normalized_rankings_byte_identical\tYES\n'
  printf 'ranked_rows\t%s\n' "$production_rows"
  printf 'target_forcing\tNO\n'
  printf 'train_holdout_accessed\tNO\n'
  printf 'historical_test_accessed\tNO\n'
} > "$OUT/equivalence_summary.tsv"

echo "v0761_actual_dev_production_equivalence=PASS"
echo "candidate_catalog_byte_identical=YES"
echo "query_manifest_byte_identical=YES"
echo "normalized_rankings_byte_identical=YES"
echo "train_holdout_accessed=NO"
echo "historical_test_accessed=NO"
echo "v0761_out=$OUT"
