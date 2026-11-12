#!/bin/bash
#SBATCH --job-name=redeem-f2g-score
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/research/juan/singj/foundation_model/redeem-f2g-score-%j.out
#SBATCH --error=/nfs/research/juan/singj/foundation_model/redeem-f2g-score-%j.err
# F2-G: last-k checkpoint averaging (rule: rolling handoff 2026-11-12 21:52). CPU only.
# Builds A5 (epoch_056..060) and A10 (epoch_051..060) per seed from the rtlong resume runs,
# then scores them with the F2-A Validation scorer (binary dea11c0, fixed cohort 3f640cbd)
# plus ctl_ep60 (seed-0 epoch_060) as the binary-equivalence control. Validation only.
set -euo pipefail
ROOT=/nfs/research/juan/singj/foundation_model
G="$ROOT/experiments/redeem_f2g_ckpt_average_20261112"
AVG="$G/scripts/redeem_f2g_ckpt_average_20261112_v2.py"
AVG_SHA=4d5506631a8860131fd1f73ff6a5dc91a83b389485a1fa28a7d1d1a1748aa198
EXP="$ROOT/experiments/redeem_f2_scorecard_20261101"
BIN="$EXP/bin/foundation_f2_scorecard_dea11c0"
BIN_SHA=5cfb01c86aa7011821a801f7ebab1356fc7b0b89645222eb9048569cfa3569c5
COHORT="$EXP/runs/scorecard_15264522/cohort.tsv"
COHORT_SHA=3f640cbdd3a5dc4183a9055a5892d8cd9a291a31f8a54055b7ad24afb8a98056
RUN_YAML="$ROOT/experiments/redeem_f2f_arms_20261109/prepared/run_v0260_root20261111.yaml"
RUN_SHA=ca642a6f7fe53d5a2faa3830b3eeea75c430bfb2521acf20210968d35de276c7
CORPUS_FNV=fnv1a64:a2a6f57d31064ba6
BENCH_FNV=fnv1a64:2133c039625f77df
TEACHER="$ROOT/experiments/multimodal_v0350_01_trainable_forward_representation/model/best"
RUNS="$ROOT/experiments/redeem_f2f_arms_20261109/runs"
SEEDS=(rtlong_ep60_resume40_seed20261006_initdefault_f563ae37faf3_26433793
       rtlong_ep60_resume41_seed20261007_init1_f563ae37faf3_26433794
       rtlong_ep60_resume40_seed20261008_init2_f563ae37faf3_26433795)
OUT="$G/score/val_${SLURM_JOB_ID}"
CK="$G/ckpts/job_${SLURM_JOB_ID}"
test ! -e "$OUT" && test ! -e "$CK" || { echo "BLOCKED: output exists"; exit 2; }
echo "$AVG_SHA  $AVG" | sha256sum -c -
echo "$BIN_SHA  $BIN" | sha256sum -c -
echo "$COHORT_SHA  $COHORT" | sha256sum -c -
python3 "$AVG" --self-test
"$BIN" --self-test
mkdir -p "$OUT" "$CK"
CKPTS=("ctl_ep60=$RUNS/${SEEDS[0]}/model/epoch_060")
for s in 0 1 2; do
  M="$RUNS/${SEEDS[$s]}/model"
  for k in 5 10; do
    src=(); for e in $(seq $((61 - k)) 60); do src+=("$M/epoch_$(printf '%03d' "$e")"); done
    python3 "$AVG" "$CK/s${s}_a${k}" "${src[@]}"
    CKPTS+=("a${k}_s${s}=$CK/s${s}_a${k}")
  done
done
{
  echo "job_id=$SLURM_JOB_ID"; echo "host=$(hostname)"; echo "avg_sha256=$AVG_SHA"; echo "bin_sha256=$BIN_SHA"
  echo "test_partition_read=NO"; echo "train_holdout_read=NO"
  for spec in "${CKPTS[@]}"; do c="${spec#*=}"; echo "ckpt ${spec%%=*} $c"; sha256sum "$c/model.safetensors"; done
  for s in 0 1 2; do for k in 5 10; do echo "sources s${s}_a${k}:"; cat "$CK/s${s}_a${k}/average_sources.txt"; done; done
  date -u +started=%FT%TZ
} > "$OUT/provenance.txt"
/usr/bin/time -v "$BIN" score "$RUN_YAML" "$RUN_SHA" "$CORPUS_FNV" "$BENCH_FNV" "$COHORT" "$TEACHER" "${CKPTS[@]}" \
  > "$OUT/scores.tsv" 2> "$OUT/score.stderr"
grep -q '^#F2_SCORECARD_SCORE=COMPLETE$' "$OUT/scores.tsv"
date -u +finished=%FT%TZ >> "$OUT/provenance.txt"
sha256sum "$OUT/scores.tsv" > "$OUT/output_sha256.txt"
echo "F2G_SCORE_JOB=PASS out=$OUT"
