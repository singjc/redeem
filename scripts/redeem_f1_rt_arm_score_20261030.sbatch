#!/bin/bash
#SBATCH --job-name=redeem-f1-rt-arm-score
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:50:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-f1-rt-arm-score-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-f1-rt-arm-score-%j.err
# F1-D CPU scorer: RT of unified checkpoints on the harmonized coordinate, first 1,024 Validation.
# Required env: EXPECTED_BIN_SHA256, CKPTS (space-separated checkpoint dirs). No TEST.
# Standalone batch script: normal failure exits are allowed.
set -euo pipefail
ROOT=/nfs/research/juan/DIA/singj/foundation_model
EXP="$ROOT/experiments/redeem_f1_rt_contract_audit_20261030"
BIN="$EXP/bin/foundation_f1_rt_arm_score"
RUN_YAML="$ROOT/experiments/multimodal_v0261_03_openptm_fragmentgeometryfix2/prepared/run_v0260.yaml"
RUN_SHA=7214b9cdd3836ec028655d5d6b8bb088280558626b24c9afc63ec06d582b4201
CORPUS_FNV=fnv1a64:a2a6f57d31064ba6
BENCH_FNV=fnv1a64:2133c039625f77df
OUT="$EXP/runs/arm_score_${SLURM_JOB_ID}"
test -n "${EXPECTED_BIN_SHA256:-}" || { echo "BLOCKED: EXPECTED_BIN_SHA256 unset"; exit 2; }
test -n "${CKPTS:-}" || { echo "BLOCKED: CKPTS unset"; exit 2; }
test ! -e "$OUT" || { echo "BLOCKED: output exists $OUT"; exit 2; }
echo "$EXPECTED_BIN_SHA256  $BIN" | sha256sum -c - || { echo "BLOCKED: binary SHA mismatch"; exit 2; }
read -r -a CK <<< "$CKPTS"
for c in "${CK[@]}"; do test -s "$c/model.safetensors" -a -s "$c/metadata.yaml" || { echo "BLOCKED: $c"; exit 2; }; done
mkdir -p "$OUT"
{
  echo "job_id=$SLURM_JOB_ID"; echo "host=$(hostname)"; echo "bin_sha256=$EXPECTED_BIN_SHA256"
  echo "source_commit=3f17de4"; echo "run_yaml_sha256=$RUN_SHA"; echo "max_records=1024"
  echo "test_labels_read=NO"; for c in "${CK[@]}"; do sha256sum "$c/model.safetensors" "$c/metadata.yaml"; done
  date -u +started=%FT%TZ
} > "$OUT/provenance.txt"
/usr/bin/time -v "$BIN" "$RUN_YAML" "$RUN_SHA" "$CORPUS_FNV" "$BENCH_FNV" 1024 "${CK[@]}" \
  > "$OUT/rt_arm_score.tsv" 2> "$OUT/rt_arm_score.stderr"
date -u +finished=%FT%TZ >> "$OUT/provenance.txt"
grep -q '^F1_RT_ARM_SCORE=COMPLETE$' "$OUT/rt_arm_score.tsv"
sha256sum "$OUT/rt_arm_score.tsv" > "$OUT/output_sha256.txt"
echo "F1_RT_ARM_SCORE_JOB=PASS out=$OUT"
