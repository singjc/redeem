#!/bin/bash
#SBATCH --job-name=redeem-f2f-score
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/nfs/research/juan/singj/foundation_model/redeem-f2f-score-%j.out
#SBATCH --error=/nfs/research/juan/singj/foundation_model/redeem-f2f-score-%j.err
# F2-F Validation scoring (CPU). Copy of redeem_f2e_score_val_20261107.sbatch (2a045532...) with
# the checkpoint list taken from env: F2F_ARM (label prefix), F2F_RUN0/1/2 (run folders under
# redeem_f2f_arms_20261109/runs), F2F_EPOCHS (space-separated). Same binary, summary and fixed
# F2-A cohort. Validation only, no TEST.
set -euo pipefail
ROOT=/nfs/research/juan/singj/foundation_model
EXP="$ROOT/experiments/redeem_f2_scorecard_20261101"
BIN="${F2F_BIN:-$EXP/bin/foundation_f2_scorecard_dea11c0}"
SUMMARY="$EXP/scripts/redeem_f2_scorecard_summary_20261101.py"
COHORT="$EXP/runs/scorecard_15264522/cohort.tsv"
COHORT_SHA=3f640cbdd3a5dc4183a9055a5892d8cd9a291a31f8a54055b7ad24afb8a98056
BIN_SHA="${F2F_BIN_SHA:-5cfb01c86aa7011821a801f7ebab1356fc7b0b89645222eb9048569cfa3569c5}"
SUMMARY_SHA=2bd3a71efe0b2b7f659fb2044f561d50b46106b8e4b4441c6ad9c5b83b5946a1
# 2026-11-11 root move: re-rooted copy of run_v0260.yaml (prefix only; fingerprints are checked below).
RUN_YAML="$ROOT/experiments/redeem_f2f_arms_20261109/prepared/run_v0260_root20261111.yaml"
RUN_SHA=ca642a6f7fe53d5a2faa3830b3eeea75c430bfb2521acf20210968d35de276c7
CORPUS_FNV=fnv1a64:a2a6f57d31064ba6
BENCH_FNV=fnv1a64:2133c039625f77df
TEACHER="$ROOT/experiments/multimodal_v0350_01_trainable_forward_representation/model/best"
RUNS="$ROOT/experiments/redeem_f2f_arms_20261109/runs"
: "${F2F_ARM:?}" "${F2F_RUN0:?}" "${F2F_RUN1:?}" "${F2F_RUN2:?}" "${F2F_EPOCHS:?}"
case "$F2F_ARM" in ms2tok|rtlong) ;; *) echo "BLOCKED: F2F_ARM must be ms2tok or rtlong"; exit 2;; esac
R0="$RUNS/$F2F_RUN0/model"
R1="$RUNS/$F2F_RUN1/model"
R2="$RUNS/$F2F_RUN2/model"
CKPTS=()
# Resumed runs (2026-11-11): optional F2F_LATE_RUN0/1/2 = the resume folder of each seed. An epoch
# is read from the late folder if it holds that epoch_NNN, else from the original folder. The
# two folders never share an epoch (the resume folder starts after the resume epoch).
pick() { local late="$1" main="$2" p="$3"
  if [ -n "$late" ] && [ -d "$RUNS/$late/model/epoch_$p" ]; then echo "$RUNS/$late/model/epoch_$p"
  else echo "$main/epoch_$p"; fi; }
for e in $F2F_EPOCHS; do
  p=$(printf '%03d' "$e")
  CKPTS+=("${F2F_ARM}0_ep$e=$(pick "${F2F_LATE_RUN0:-}" "$R0" "$p")" "${F2F_ARM}1_ep$e=$(pick "${F2F_LATE_RUN1:-}" "$R1" "$p")" "${F2F_ARM}2_ep$e=$(pick "${F2F_LATE_RUN2:-}" "$R2" "$p")")
done
LAST=${F2F_EPOCHS##* }
# Optional binary-equivalence control: one F2-E checkpoint, rows must equal the F2-E scores.
[ -n "${F2F_CONTROL_CKPT:-}" ] && CKPTS+=("ctl_e0_ep30=$F2F_CONTROL_CKPT")
OUT="$ROOT/experiments/redeem_f2f_arms_20261109/score/${F2F_ARM}_val_${SLURM_JOB_ID}"
test ! -e "$OUT" || { echo "BLOCKED: output exists $OUT"; exit 2; }
echo "$BIN_SHA  $BIN" | sha256sum -c - || { echo "BLOCKED: binary SHA mismatch"; exit 2; }
echo "$SUMMARY_SHA  $SUMMARY" | sha256sum -c - || { echo "BLOCKED: summary SHA mismatch"; exit 2; }
echo "$COHORT_SHA  $COHORT" | sha256sum -c - || { echo "BLOCKED: cohort SHA mismatch"; exit 2; }
for spec in "${CKPTS[@]}" "teacher=$TEACHER"; do
  c="${spec#*=}"; test -s "$c/model.safetensors" -a -s "$c/metadata.yaml" || { echo "BLOCKED: $c"; exit 2; }
done
"$BIN" --self-test
mkdir -p "$OUT"
{
  echo "job_id=$SLURM_JOB_ID"; echo "host=$(hostname)"; echo "bin=$BIN"; echo "bin_sha256=$BIN_SHA"
  echo "summary_sha256=$SUMMARY_SHA"; echo "cohort=$COHORT"; echo "cohort_sha256=$COHORT_SHA"
  echo "run_yaml_sha256=$RUN_SHA"; echo "test_labels_read=NO"
  for spec in "${CKPTS[@]}" "teacher=$TEACHER"; do c="${spec#*=}"; echo "ckpt ${spec%%=*}"; sha256sum "$c/model.safetensors" "$c/metadata.yaml"; done
  date -u +started=%FT%TZ
} > "$OUT/provenance.txt"
/usr/bin/time -v "$BIN" score "$RUN_YAML" "$RUN_SHA" "$CORPUS_FNV" "$BENCH_FNV" "$COHORT" "$TEACHER" "${CKPTS[@]}" \
  > "$OUT/scores.tsv" 2> "$OUT/score.stderr"
grep -q '^#F2_SCORECARD_SCORE=COMPLETE$' "$OUT/scores.tsv"
python3 "$SUMMARY" "$OUT/scores.tsv" --reference "${F2F_ARM}0_ep$LAST" \
  --epochs "${F2F_ARM}0=$R0/validation_metrics.tsv" "${F2F_ARM}1=$R1/validation_metrics.tsv" "${F2F_ARM}2=$R2/validation_metrics.tsv" \
  > "$OUT/summary.tsv"
grep -q '^F2_SUMMARY=COMPLETE$' "$OUT/summary.tsv"
date -u +finished=%FT%TZ >> "$OUT/provenance.txt"
sha256sum "$OUT/scores.tsv" "$OUT/summary.tsv" > "$OUT/output_sha256.txt"
echo "F2F_SCORE_VAL_JOB=PASS arm=$F2F_ARM out=$OUT"
