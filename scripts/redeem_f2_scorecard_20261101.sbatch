#!/bin/bash
#SBATCH --job-name=redeem-f2-scorecard
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Timing estimate (15179621: 3,024 records in 749 s; 15223424: 2,016 in 525 s): about 4 records/s
# per checkpoint on CPU, load included. 5 checkpoints x up to 16,896 records is about 6 h at most.
# F2-A fixed scorecard (CPU). select -> score (v0.52 + 4 unified checkpoints) -> summary.
# Required env: OUT_ROOT, EXPECTED_BIN_SHA256, EXPECTED_SUMMARY_SHA256. Validation only, no TEST.
# Standalone batch script: normal failure exits are allowed.
set -euo pipefail
ROOT=/nfs/research/juan/DIA/singj/foundation_model
EXP="$ROOT/experiments/redeem_f2_scorecard_20261101"
BIN="$EXP/bin/foundation_f2_scorecard_dea11c0"
SUMMARY="$EXP/scripts/redeem_f2_scorecard_summary_20261101.py"
RUN_YAML="$ROOT/experiments/multimodal_v0261_03_openptm_fragmentgeometryfix2/prepared/run_v0260.yaml"
RUN_SHA=7214b9cdd3836ec028655d5d6b8bb088280558626b24c9afc63ec06d582b4201
CORPUS_FNV=fnv1a64:a2a6f57d31064ba6
BENCH_FNV=fnv1a64:2133c039625f77df
SEED=20261101
PER_TASK=256
TEACHER="$ROOT/experiments/multimodal_v0350_01_trainable_forward_representation/model/best"
AB="$ROOT/experiments/redeem_f1_rtobj_ab_20261030/runs"
U51="$ROOT/experiments/foundation_research_sampling_convergence_v1/learning_3e4f07322218_62709585/model"
CKPTS=(
  "v052=$ROOT/experiments/multimodal_v0520_02_mobility_aware_pair_representation_full/model/best"
  "armA_ep6=$AB/armA_a832c3a40225_15102195/model/best_forward"
  "armB_ep6=$AB/armB_a832c3a40225_15102196/model/best_forward"
  "u51_ep8=$U51/best_forward"
  "u51_ep10=$U51"
)
test -n "${OUT_ROOT:-}" || { echo "BLOCKED: OUT_ROOT unset"; exit 2; }
OUT="$OUT_ROOT/scorecard_${SLURM_JOB_ID}"
test ! -e "$OUT" || { echo "BLOCKED: output exists $OUT"; exit 2; }
echo "$EXPECTED_BIN_SHA256  $BIN" | sha256sum -c - || { echo "BLOCKED: binary SHA mismatch"; exit 2; }
echo "$EXPECTED_SUMMARY_SHA256  $SUMMARY" | sha256sum -c - || { echo "BLOCKED: summary SHA mismatch"; exit 2; }
for spec in "${CKPTS[@]}" "teacher=$TEACHER"; do
  c="${spec#*=}"; test -s "$c/model.safetensors" -a -s "$c/metadata.yaml" || { echo "BLOCKED: $c"; exit 2; }
done
"$BIN" --self-test
mkdir -p "$OUT"
{
  echo "job_id=$SLURM_JOB_ID"; echo "host=$(hostname)"; echo "bin=$BIN"; echo "bin_sha256=$EXPECTED_BIN_SHA256"
  echo "source_commit=dea11c0"; echo "summary_sha256=$EXPECTED_SUMMARY_SHA256"; echo "run_yaml_sha256=$RUN_SHA"
  echo "seed=$SEED"; echo "per_task=$PER_TASK"; echo "excluded_validation_prefix=2048"; echo "test_labels_read=NO"
  for spec in "${CKPTS[@]}" "teacher=$TEACHER"; do c="${spec#*=}"; echo "ckpt ${spec%%=*}"; sha256sum "$c/model.safetensors" "$c/metadata.yaml"; done
  date -u +started=%FT%TZ
} > "$OUT/provenance.txt"
/usr/bin/time -v "$BIN" select "$RUN_YAML" "$RUN_SHA" "$CORPUS_FNV" "$BENCH_FNV" "$SEED" "$PER_TASK" \
  > "$OUT/cohort.tsv" 2> "$OUT/select.stderr"
grep -q '^#F2_SCORECARD_SELECT=COMPLETE$' "$OUT/cohort.tsv"
sha256sum "$OUT/cohort.tsv" >> "$OUT/provenance.txt"
/usr/bin/time -v "$BIN" score "$RUN_YAML" "$RUN_SHA" "$CORPUS_FNV" "$BENCH_FNV" "$OUT/cohort.tsv" "$TEACHER" "${CKPTS[@]}" \
  > "$OUT/scores.tsv" 2> "$OUT/score.stderr"
grep -q '^#F2_SCORECARD_SCORE=COMPLETE$' "$OUT/scores.tsv"
python3 "$SUMMARY" "$OUT/scores.tsv" --reference v052 \
  --epochs "armA=$AB/armA_a832c3a40225_15102195/model/validation_metrics.tsv" \
           "armB=$AB/armB_a832c3a40225_15102196/model/validation_metrics.tsv" \
           "u51=$U51/validation_metrics.tsv" \
  > "$OUT/summary.tsv"
grep -q '^F2_SUMMARY=COMPLETE$' "$OUT/summary.tsv"
date -u +finished=%FT%TZ >> "$OUT/provenance.txt"
sha256sum "$OUT/cohort.tsv" "$OUT/scores.tsv" "$OUT/summary.tsv" > "$OUT/output_sha256.txt"
echo "F2_SCORECARD_JOB=PASS out=$OUT"
