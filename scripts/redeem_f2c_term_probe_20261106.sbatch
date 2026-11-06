#!/bin/bash
#SBATCH --job-name=redeem-f2c-probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-f2c-probe-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-f2c-probe-%j.err
# F2-C per-term gradient probe (diagnostic, option c). Commit b652ee6.
# Four short phases, each 1 epoch x 64 cycles, probe every 8 cycles, data seed 20261006
# (the same batches in every phase). Config = the R0 (20306282) foundation_train.json with only
# these fields changed: epochs, patience, cycles, validation size, inverse/generation eval
# records, term_gradient_probe_every, save_epoch_checkpoints, output_file, checkpoint_file.
#   P0 scratch (R0 init) | P1 warm R0 epoch_001 | P2 warm R0 epoch_010 | P3 warm R2 epoch_010
# Probe values at cycle 0 of P1 to P3 are on the exact checkpoint weights. Later cycles drift a
# little (fresh AdamW state, LR at most 64% of the base LR in warmup). Validation only.
set -uo pipefail
ROOT=/nfs/research/juan/DIA/singj/foundation_model
EXP="$ROOT/experiments/redeem_f2c_term_gradient_probe_20261106"
RUNS="$ROOT/experiments/redeem_f2b_seed_replicates_20261101/runs"
R0="$RUNS/seed20261006_initdefault_82f490de1178_20306282"
R2="$RUNS/seed20261008_init2_82f490de1178_20306284"
BASE_CFG="$R0/foundation_train.json"
EXPECTED_COMMIT=b652ee6
: "${SIF:?}" "${EXPECTED_SIF_SHA256:?}" "${EXPECTED_GIT_SHA:?}"
[ "${EXPECTED_GIT_SHA:0:7}" = "$EXPECTED_COMMIT" ] || { echo "ERROR: commit pin" >&2; exit 2; }
[ "$(sha256sum "$SIF" | awk '{print $1}')" = "$EXPECTED_SIF_SHA256" ] || { echo "ERROR: SIF SHA" >&2; exit 2; }
module load singularityce/4.1.0 >/dev/null 2>&1 || true
command -v singularity >/dev/null && command -v nvidia-smi >/dev/null || { echo "ERROR: no singularity/GPU" >&2; exit 2; }
for c in "$R0/model/epoch_001" "$R0/model/epoch_010" "$R2/model/epoch_010"; do
  [ -s "$c/model.safetensors" ] && [ -s "$c/metadata.yaml" ] || { echo "ERROR: missing $c" >&2; exit 2; }
done
OUT="$EXP/runs/probe_${EXPECTED_GIT_SHA:0:12}_${SLURM_JOB_ID}"
[ ! -e "$OUT" ] || { echo "ERROR: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT" || exit 2
{ echo "job_id=$SLURM_JOB_ID"; echo "git_sha=$EXPECTED_GIT_SHA"; echo "sif_sha256=$EXPECTED_SIF_SHA256"
  echo "base_cfg=$BASE_CFG"; sha256sum "$BASE_CFG"; nvidia-smi -L
  for c in "$R0/model/epoch_001" "$R0/model/epoch_010" "$R2/model/epoch_010"; do sha256sum "$c/model.safetensors"; done
  echo "test_consumed=NO"; echo "train_holdout_consumed=NO"; } > "$OUT/provenance.txt"
rc_all=0
for spec in "P0=" "P1=$R0/model/epoch_001" "P2=$R0/model/epoch_010" "P3=$R2/model/epoch_010"; do
  phase=${spec%%=*}; ckpt=${spec#*=}; P="$OUT/$phase"; mkdir -p "$P"
  python3 - "$BASE_CFG" "$P/foundation_train.json" "$P/model" "$ckpt" <<'PY' || { echo "ERROR: config $phase" >&2; exit 3; }
import json, sys
base, dest, model, ckpt = sys.argv[1:]
c = json.load(open(base))
f = c['foundation']
assert f['seed'] == 20261006 and f['init_seed'] is None and f['max_gradient_norm'] == 1.0
c.update(output_file=model, checkpoint_file=(ckpt or None), epochs=1, early_stopping_patience=1)
f.update(epochs=1, early_stopping_patience=1, max_train_batches_per_epoch=64,
         max_validation_batches=8, inverse_evaluation_records=16, generation_evaluation_records=0,
         term_gradient_probe_every=8, save_epoch_checkpoints=False)
json.dump(c, open(dest, 'w'), indent=2, sort_keys=True)
print('F2C_CONFIG=PASS', dest)
PY
  singularity exec --nv --bind "$ROOT:$ROOT" "$SIF" redeem properties train "$P/foundation_train.json" > "$P/train.log" 2>&1
  rc=$?
  n=$(grep -c '^foundation_term_gradients' "$P/train.log")
  bad=$(awk -F'\t' '/^foundation_term_gradients/{for(i=2;i<=NF;i++){split($i,kv,"=");v[kv[1]]=kv[2]} r=v["sum_residual"]; t=v["total_value"]; if(r<0)r=-r; if(t<0)t=-t; if(r>1e-3*(t>1?t:1)) b++} END{print b+0}' "$P/train.log")
  grep -q "Config Invalid value for 'foundation'" "$P/train.log" && { echo "ERROR: $phase config fell back to defaults" >&2; rc=5; }
  echo "PHASE $phase rc=$rc probe_lines=$n residual_violations=$bad ckpt=${ckpt:-scratch}"
  if [ "$rc" -ne 0 ] || [ "$n" -lt 40 ] || [ "$bad" -ne 0 ]; then rc_all=1; tail -n 40 "$P/train.log" >&2; fi
done
find "$OUT" -name '*.log' -o -name '*.json' -o -name 'provenance.txt' | sort | xargs sha256sum > "$OUT/output_sha256.txt"
[ "$rc_all" -eq 0 ] && echo "F2C_PROBE_JOB=PASS out=$OUT" || { echo "F2C_PROBE_JOB=FAIL out=$OUT"; exit 1; }
