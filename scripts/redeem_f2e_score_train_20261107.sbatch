#!/bin/bash
#SBATCH --job-name=redeem-f2e-train
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-f2e-train-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-f2e-train-%j.err
# F2-E TRAIN-fit scoring. Copy of redeem_f2d_train_fit_20261107.sbatch (da0d9b72...) with only
# the checkpoints (epoch 30 of 23501944/5/6), the seen-record exposure files and the output
# root changed. Same binary 2a9d48f, same TRAIN cohort rule (seed 20261101, 256 per task).
# The benchmark Test partition (protected TRAIN-HOLDOUT) is not reachable.
set -euo pipefail
ROOT=/nfs/research/juan/DIA/singj/foundation_model
EXP="$ROOT/experiments/redeem_f2d_train_fit_20261107"
BIN="$EXP/bin/foundation_f2_scorecard_2a9d48f"
OUT_EXP="$ROOT/experiments/redeem_f2e_budget_20261107"
SUMMARY="$ROOT/experiments/redeem_f2_scorecard_20261101/scripts/redeem_f2_scorecard_summary_20261101.py"
SUMMARY_SHA=2bd3a71efe0b2b7f659fb2044f561d50b46106b8e4b4441c6ad9c5b83b5946a1
SEEN="$EXP/scripts/redeem_f2d_seen_20261107_v2.py"
BENCH_TSV="$ROOT/experiments/multimodal_v0261_03_openptm_fragmentgeometryfix2/prepared/benchmark_train_derived_v0260.tsv"
RUN_YAML="$ROOT/experiments/multimodal_v0261_03_openptm_fragmentgeometryfix2/prepared/run_v0260.yaml"
RUN_SHA=7214b9cdd3836ec028655d5d6b8bb088280558626b24c9afc63ec06d582b4201
CORPUS_FNV=fnv1a64:a2a6f57d31064ba6
BENCH_FNV=fnv1a64:2133c039625f77df
TEACHER="$ROOT/experiments/multimodal_v0350_01_trainable_forward_representation/model/best"
RUNS="$ROOT/experiments/redeem_f2e_budget_20261107/runs"
E0="$RUNS/ep30_seed20261006_initdefault_82f490de1178_23501944/model"
E1="$RUNS/ep30_seed20261007_init1_82f490de1178_23501945/model"
E2="$RUNS/ep30_seed20261008_init2_82f490de1178_23501946/model"
CKPTS=(
  "v052=$ROOT/experiments/multimodal_v0520_02_mobility_aware_pair_representation_full/model/best"
  "e0_ep30=$E0/epoch_030"
  "e1_ep30=$E1/epoch_030"
  "e2_ep30=$E2/epoch_030"
)
: "${EXPECTED_BIN_SHA256:?}" "${EXPECTED_SEEN_SHA256:?}"
export F2_PARTITION=train
OUT="$OUT_EXP/score/train_${SLURM_JOB_ID}"
test ! -e "$OUT" || { echo "BLOCKED: output exists $OUT"; exit 2; }
echo "$EXPECTED_BIN_SHA256  $BIN" | sha256sum -c - || { echo "BLOCKED: binary SHA mismatch"; exit 2; }
echo "$SUMMARY_SHA  $SUMMARY" | sha256sum -c - || { echo "BLOCKED: summary SHA mismatch"; exit 2; }
echo "$EXPECTED_SEEN_SHA256  $SEEN" | sha256sum -c - || { echo "BLOCKED: seen script SHA mismatch"; exit 2; }
for spec in "${CKPTS[@]}" "teacher=$TEACHER"; do
  c="${spec#*=}"; test -s "$c/model.safetensors" -a -s "$c/metadata.yaml" || { echo "BLOCKED: $c"; exit 2; }
done
"$BIN" --self-test
mkdir -p "$OUT"
{
  echo "job_id=$SLURM_JOB_ID"; echo "host=$(hostname)"; echo "bin_sha256=$EXPECTED_BIN_SHA256"
  echo "source_commit=2a9d48f"; echo "F2_PARTITION=$F2_PARTITION"; echo "seed=20261101"; echo "per_task=256"
  echo "test_partition_read=NO"; echo "train_holdout_read=NO"
  for spec in "${CKPTS[@]}" "teacher=$TEACHER"; do c="${spec#*=}"; echo "ckpt ${spec%%=*}"; sha256sum "$c/model.safetensors"; done
  date -u +started=%FT%TZ
} > "$OUT/provenance.txt"
"$BIN" select "$RUN_YAML" "$RUN_SHA" "$CORPUS_FNV" "$BENCH_FNV" 20261101 256 > "$OUT/cohort.tsv" 2> "$OUT/select.stderr"
grep -q '^#F2_SCORECARD_SELECT=COMPLETE$' "$OUT/cohort.tsv"
grep -qx '#partition	train' "$OUT/cohort.tsv"
sha256sum "$OUT/cohort.tsv" >> "$OUT/provenance.txt"
# Mark per seed which TRAIN cohort records the property stages saw (exact rebuild, checked
# against each run's training_exposure.tsv; fails on any mismatch).
python3 "$SEEN" "$BENCH_TSV" "$OUT/cohort.tsv" \
  20261006="$E0/training_exposure.tsv" \
  20261007="$E1/training_exposure.tsv" \
  20261008="$E2/training_exposure.tsv" \
  > "$OUT/seen.tsv"
grep -q '^F2D_SEEN=COMPLETE$' "$OUT/seen.tsv"
/usr/bin/time -v "$BIN" score "$RUN_YAML" "$RUN_SHA" "$CORPUS_FNV" "$BENCH_FNV" "$OUT/cohort.tsv" "$TEACHER" "${CKPTS[@]}" \
  > "$OUT/scores.tsv" 2> "$OUT/score.stderr"
grep -q '^#F2_SCORECARD_SCORE=COMPLETE$' "$OUT/scores.tsv"
python3 "$SUMMARY" "$OUT/scores.tsv" --reference v052 > "$OUT/summary.tsv"
grep -q '^F2_SUMMARY=COMPLETE$' "$OUT/summary.tsv"
date -u +finished=%FT%TZ >> "$OUT/provenance.txt"
sha256sum "$OUT/cohort.tsv" "$OUT/seen.tsv" "$OUT/scores.tsv" "$OUT/summary.tsv" > "$OUT/output_sha256.txt"
echo "F2E_SCORE_TRAIN_JOB=PASS out=$OUT"
