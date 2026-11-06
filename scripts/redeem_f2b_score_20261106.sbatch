#!/bin/bash
#SBATCH --job-name=redeem-f2b-score
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-f2b-score-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-f2b-score-%j.err
# F2-B step-1 scoring (CPU). Derived from redeem_f2_scorecard_20261101.sbatch (8ba41ca).
# Same pinned binary (dea11c0, 5cfb01c8...) and summary (2bd3a71e...). Reuses the FIXED F2-A
# cohort.tsv from 15264522 (SHA 3f640cbd...8056); no new selection. Scores epochs 6/8/10 of the
# three F2-B seed replicates. Each R0 checkpoint is byte-identical to the u51 checkpoint of the
# same epoch, which gives a scorer reproduction control. Validation only, no TEST.
# Timing: 15264522 scored 5 checkpoints in 3 h 51 min, so 9 checkpoints take about 7 h.
set -euo pipefail
ROOT=/nfs/research/juan/DIA/singj/foundation_model
EXP="$ROOT/experiments/redeem_f2_scorecard_20261101"
BIN="$EXP/bin/foundation_f2_scorecard_dea11c0"
SUMMARY="$EXP/scripts/redeem_f2_scorecard_summary_20261101.py"
COHORT="$EXP/runs/scorecard_15264522/cohort.tsv"
COHORT_SHA=3f640cbdd3a5dc4183a9055a5892d8cd9a291a31f8a54055b7ad24afb8a98056
BIN_SHA=5cfb01c86aa7011821a801f7ebab1356fc7b0b89645222eb9048569cfa3569c5
SUMMARY_SHA=2bd3a71efe0b2b7f659fb2044f561d50b46106b8e4b4441c6ad9c5b83b5946a1
RUN_YAML="$ROOT/experiments/multimodal_v0261_03_openptm_fragmentgeometryfix2/prepared/run_v0260.yaml"
RUN_SHA=7214b9cdd3836ec028655d5d6b8bb088280558626b24c9afc63ec06d582b4201
CORPUS_FNV=fnv1a64:a2a6f57d31064ba6
BENCH_FNV=fnv1a64:2133c039625f77df
TEACHER="$ROOT/experiments/multimodal_v0350_01_trainable_forward_representation/model/best"
RUNS="$ROOT/experiments/redeem_f2b_seed_replicates_20261101/runs"
R0="$RUNS/seed20261006_initdefault_82f490de1178_20306282/model"
R1="$RUNS/seed20261007_init1_82f490de1178_20306283/model"
R2="$RUNS/seed20261008_init2_82f490de1178_20306284/model"
CKPTS=()
for e in 6 8 10; do
  p=$(printf '%03d' "$e")
  CKPTS+=("r0_ep$e=$R0/epoch_$p" "r1_ep$e=$R1/epoch_$p" "r2_ep$e=$R2/epoch_$p")
done
OUT="$ROOT/experiments/redeem_f2b_seed_replicates_20261101/score/score_${SLURM_JOB_ID}"
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
python3 "$SUMMARY" "$OUT/scores.tsv" --reference r0_ep10 \
  --epochs "r0=$R0/validation_metrics.tsv" "r1=$R1/validation_metrics.tsv" "r2=$R2/validation_metrics.tsv" \
  > "$OUT/summary.tsv"
grep -q '^F2_SUMMARY=COMPLETE$' "$OUT/summary.tsv"
date -u +finished=%FT%TZ >> "$OUT/provenance.txt"
sha256sum "$OUT/scores.tsv" "$OUT/summary.tsv" > "$OUT/output_sha256.txt"
echo "F2B_SCORE_JOB=PASS out=$OUT"
