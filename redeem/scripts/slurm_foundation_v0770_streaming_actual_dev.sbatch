#!/usr/bin/env bash
#SBATCH --job-name=redeem-v0770-stream
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0770-stream-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-v0770-stream-%j.err

module load singularityce/4.1.0

ROOT=/nfs/research/juan/DIA/singj/foundation_model
: "${SIF:?SIF is required}"
: "${REF:?REF is required}"
: "${OUT:?OUT is required}"

if [ ! -f "$SIF" ]; then
  echo "missing v0.77.0 SIF: $SIF" >&2
  exit 2
fi
for required in \
  "$REF/catalog.yaml" \
  "$REF/export/batch.yaml" \
  "$REF/batch_hits.tsv" \
  "$REF/equivalence_summary.tsv"; do
  if [ ! -s "$required" ]; then
    echo "missing validated v0.76.1 reference artifact: $required" >&2
    exit 2
  fi
done
if [ -e "$OUT" ]; then
  echo "v0.77.0 output must be fresh: $OUT" >&2
  exit 2
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity is required" >&2
  exit 127
fi

mkdir -p "$OUT"
SPECTRA="$OUT/requests.tsv"
HITS="$OUT/stream_hits.tsv"
SUMMARY="$OUT/stream_summary.tsv"
CONVERT_LOG="$OUT/convert.log"
STREAM_LOG="$OUT/stream.log"
CONVERT_TIME="$OUT/convert.time.txt"
STREAM_TIME="$OUT/stream.time.txt"

BIND="$ROOT:$ROOT"
if [ -n "${REDEEM_SINGULARITY_BINDS:-}" ]; then
  BIND="${BIND},${REDEEM_SINGULARITY_BINDS}"
fi

printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf 'host=%s\n' "$(hostname)"
printf 'execution_device=CPU\n'
printf 'runtime=v0770_streaming_actual_dev\n'
printf 'scientific_core=v0751_frozen\n'
printf 'api_contract=v0752_frozen\n'
printf 'candidate_policy=same_charge_nearest256\n'
printf 'target_forcing=NO\n'
printf 'train_holdout_accessed=NO\n'
printf 'historical_test_accessed=NO\n'
printf 'reference=%s\n' "$REF"
printf 'out=%s\n' "$OUT"

echo "===== container self-test ====="
singularity exec --bind "$BIND" "$SIF" foundation_identify_stream_v0770 --self-test
self_test_rc=$?
if [ "$self_test_rc" -ne 0 ]; then
  exit "$self_test_rc"
fi

echo "===== validated batch -> streaming TSV ====="
/usr/bin/time -v -o "$CONVERT_TIME" \
  singularity exec --bind "$BIND" "$SIF" \
    foundation_identify_stream_v0770 batch-to-tsv \
      --batch "$REF/export/batch.yaml" \
      --output-spectra "$SPECTRA" \
  2>&1 | tee "$CONVERT_LOG"
convert_rc=${PIPESTATUS[0]}
if [ "$convert_rc" -ne 0 ]; then
  exit "$convert_rc"
fi

if [ ! -s "$SPECTRA" ]; then
  echo "v0.77.0 spectra TSV is missing or empty" >&2
  exit 2
fi
if ! grep -q '^request_count=512$' "$CONVERT_LOG"; then
  echo "v0.77.0 conversion request-count gate failed" >&2
  exit 2
fi

echo "===== streaming identification ====="
/usr/bin/time -v -o "$STREAM_TIME" \
  singularity exec --bind "$BIND" "$SIF" \
    foundation_identify_stream_v0770 identify-stream \
      --catalog "$REF/catalog.yaml" \
      --spectra "$SPECTRA" \
      --output-hits "$HITS" \
      --output-summary "$SUMMARY" \
  2>&1 | tee "$STREAM_LOG"
stream_rc=${PIPESTATUS[0]}
if [ "$stream_rc" -ne 0 ]; then
  exit "$stream_rc"
fi

for required in "$HITS" "$SUMMARY" "$CONVERT_LOG" "$STREAM_LOG" "$CONVERT_TIME" "$STREAM_TIME"; do
  if [ ! -s "$required" ]; then
    echo "missing or empty v0.77.0 artifact: $required" >&2
    exit 2
  fi
done

if ! grep -q $'^status\tPASS$' "$SUMMARY"; then
  echo "v0.77.0 stream summary status gate failed" >&2
  exit 2
fi
if ! grep -q $'^catalog_size\t63332$' "$SUMMARY"; then
  echo "v0.77.0 catalog-size gate failed" >&2
  exit 2
fi
if ! grep -q $'^request_count\t512$' "$SUMMARY"; then
  echo "v0.77.0 request-count gate failed" >&2
  exit 2
fi
if ! grep -q $'^total_candidates_scored\t130931$' "$SUMMARY"; then
  echo "v0.77.0 candidate-count gate failed" >&2
  exit 2
fi
if ! grep -q $'^total_returned_hits\t130931$' "$SUMMARY"; then
  echo "v0.77.0 returned-hit-count gate failed" >&2
  exit 2
fi
if ! cmp -s "$REF/batch_hits.tsv" "$HITS"; then
  echo "v0.77.0 streamed hits differ from validated v0.76.1 actual-DEV hits" >&2
  diff -u "$REF/batch_hits.tsv" "$HITS" | head -80 >&2
  exit 2
fi
if grep -Eiq \
  'train_holdout_(accessed|consumed)(=|[[:space:]])YES|historical_(validation|test)_(accessed|consumed)(=|[[:space:]])YES|target_forcing(=|[[:space:]])YES|model\.safetensors|CONFIRM_TRAIN_HOLDOUT' \
  "$CONVERT_LOG" "$STREAM_LOG"; then
  echo "v0.77.0 unexpectedly uses protected/model-selection state" >&2
  exit 2
fi

{
  printf 'key\tvalue\n'
  printf 'status\tPASS\n'
  printf 'runtime\tv0770_streaming_actual_dev\n'
  printf 'scientific_core\tv0751_frozen\n'
  printf 'api_contract\tv0752_frozen\n'
  printf 'catalog_size\t63332\n'
  printf 'request_count\t512\n'
  printf 'candidate_pool\t256\n'
  printf 'stream_hits_byte_identical_to_v0761\tYES\n'
  printf 'target_forcing\tNO\n'
  printf 'train_holdout_accessed\tNO\n'
  printf 'historical_test_accessed\tNO\n'
} > "$OUT/equivalence_summary.tsv"

echo "v0770_streaming_actual_dev=PASS"
echo "stream_hits_byte_identical_to_v0761=YES"
echo "train_holdout_accessed=NO"
echo "historical_test_accessed=NO"
echo "v0770_out=$OUT"
