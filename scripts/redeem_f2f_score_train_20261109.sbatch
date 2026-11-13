#!/bin/bash
#SBATCH --job-name=redeem-f2f-train
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/research/juan/singj/foundation_model/redeem-f2f-train-%j.out
#SBATCH --error=/nfs/research/juan/singj/foundation_model/redeem-f2f-train-%j.err
# F2-F TRAIN-fit scoring. Copy of redeem_f2e_score_train_20261107.sbatch (dd13ca68...) with the
# checkpoints taken from env: F2F_ARM, F2F_RUN0/1/2 (run folders), F2F_LAST_EPOCH. Same binary
# 2a9d48f, same TRAIN cohort rule (seed 20261101, 256 per task). Test (TRAIN-HOLDOUT) unreachable.
set -euo pipefail
ROOT=/nfs/research/juan/singj/foundation_model
EXP="$ROOT/experiments/redeem_f2d_train_fit_20261107"
BIN="${F2F_BIN:-$EXP/bin/foundation_f2_scorecard_2a9d48f}"
OUT_EXP="$ROOT/experiments/${F2F_SCORE_EXPERIMENT:-redeem_f2f_arms_20261109}"
case "${F2F_SCORE_EXPERIMENT:-redeem_f2f_arms_20261109}" in redeem_f2f_arms_20261109|redeem_f2h_rt_harmonized_20261113) ;; *) echo "BLOCKED: unknown F2F_SCORE_EXPERIMENT"; exit 2;; esac
SUMMARY="$ROOT/experiments/redeem_f2_scorecard_20261101/scripts/redeem_f2_scorecard_summary_20261101.py"
SUMMARY_SHA=2bd3a71efe0b2b7f659fb2044f561d50b46106b8e4b4441c6ad9c5b83b5946a1
SEEN="$EXP/scripts/redeem_f2d_seen_20261107_v2.py"
BENCH_TSV="$ROOT/experiments/multimodal_v0261_03_openptm_fragmentgeometryfix2/prepared/benchmark_train_derived_v0260.tsv"
# 2026-11-11 root move: re-rooted copy of run_v0260.yaml (prefix only; fingerprints are checked below).
RUN_YAML="$ROOT/experiments/redeem_f2f_arms_20261109/prepared/run_v0260_root20261111.yaml"
RUN_SHA=ca642a6f7fe53d5a2faa3830b3eeea75c430bfb2521acf20210968d35de276c7
CORPUS_FNV=fnv1a64:a2a6f57d31064ba6
BENCH_FNV=fnv1a64:2133c039625f77df
TEACHER="$ROOT/experiments/multimodal_v0350_01_trainable_forward_representation/model/best"
RUNS="$OUT_EXP/runs"
: "${F2F_ARM:?}" "${F2F_RUN0:?}" "${F2F_RUN1:?}" "${F2F_RUN2:?}" "${F2F_LAST_EPOCH:?}"
case "$F2F_ARM" in ms2tok|rtlong|rtharm) ;; *) echo "BLOCKED: F2F_ARM must be ms2tok, rtlong or rtharm"; exit 2;; esac
E0="$RUNS/$F2F_RUN0/model"
E1="$RUNS/$F2F_RUN1/model"
E2="$RUNS/$F2F_RUN2/model"
LP=$(printf '%03d' "$F2F_LAST_EPOCH")
CKPTS=(
  "v052=$ROOT/experiments/multimodal_v0520_02_mobility_aware_pair_representation_full/model/best"
  "${F2F_ARM}0_ep$F2F_LAST_EPOCH=$E0/epoch_$LP"
  "${F2F_ARM}1_ep$F2F_LAST_EPOCH=$E1/epoch_$LP"
  "${F2F_ARM}2_ep$F2F_LAST_EPOCH=$E2/epoch_$LP"
)
: "${EXPECTED_BIN_SHA256:?}" "${EXPECTED_SEEN_SHA256:?}"
export F2_PARTITION=train
OUT="$OUT_EXP/score/${F2F_ARM}_train_${SLURM_JOB_ID}"
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
echo "F2F_SCORE_TRAIN_JOB=PASS arm=$F2F_ARM out=$OUT"
