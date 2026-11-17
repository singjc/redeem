#!/usr/bin/env bash
#SBATCH --job-name=redeem-f2e-budget
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/nfs/research/juan/singj/foundation_model/redeem-f2e-budget-%j.out
#SBATCH --error=/nfs/research/juan/singj/foundation_model/redeem-f2e-budget-%j.err

# F2-E training-budget arm. Derived from redeem_f2b_seed_replicates_20261101.sbatch
# (16ce5af6...). The ONLY recipe change is EPOCHS (env F2E_EPOCHS, default 30): the cosine
# LR schedule scales with the planned updates, so the decay is stretched over the longer run.
# Everything else is the 62709585 recipe. Same seed pairs as R0-R2.
# Random initialization. Validation only; TEST and TRAIN-HOLDOUT remain closed.
# Do not use set -e or set -u: explicitly check return codes.
module load singularityce/4.1.0
module_rc=$?
if [ "$module_rc" -ne 0 ]; then
  echo "ERROR: singularityce module load failed ($module_rc)" >&2
  exit "$module_rc"
fi

# 2026-11-11: the project root moved out of DIA/ (user). Every dataset path inside the original
# run_v0260.yaml is absolute, so a re-rooted copy (prefix replaced only, pinned by SHA) is used.
# Corpus and benchmark fingerprints do not hash paths, so they are unchanged.
ROOT=/nfs/research/juan/singj/foundation_model
PREPARED_RUN="$ROOT/experiments/redeem_f2f_arms_20261109/prepared/run_v0260_root20261111.yaml"
PREPARED_RUN_SHA256=${F2F_PREPARED_RUN_SHA256:-}
[ -n "$PREPARED_RUN_SHA256" ] || { echo "ERROR: F2F_PREPARED_RUN_SHA256 is required" >&2; exit 2; }
RUN_ROOT="$ROOT/experiments/${F2F_EXPERIMENT:-redeem_f2e_budget_20261107}/runs"
case "${F2F_EXPERIMENT:-redeem_f2e_budget_20261107}" in redeem_f2e_budget_20261107|redeem_f2f_arms_20261109|redeem_f2h_rt_harmonized_20261113|redeem_f2i_ccs_ms2_20261113) ;; *) echo "ERROR: unknown F2F_EXPERIMENT" >&2; exit 2;; esac
EPOCHS=${F2E_EPOCHS:-30}
case "$EPOCHS" in ''|*[!0-9]*) echo "ERROR: F2E_EPOCHS must be an integer" >&2; exit 2;; esac
[ "$EPOCHS" -ge 10 ] && [ "$EPOCHS" -le 60 ] || { echo "ERROR: F2E_EPOCHS out of range 10..60" >&2; exit 2; }
UPDATES=$((EPOCHS * 1024 * 5))
# Opt-in resume of an interrupted run (needs a build with f563ae3): F2F_RESUME_RUN = the old run
# folder name under RUN_ROOT, F2F_RESUME_EPOCH = N (its epoch_NNN checkpoint). Epochs 1..N are
# replayed for data order/exposure and LR only; AdamW moments restart from zero.
RESUME_RUN=${F2F_RESUME_RUN:-}
RESUME_EPOCH=${F2F_RESUME_EPOCH:-0}
case "$RESUME_EPOCH" in ''|*[!0-9]*) echo "ERROR: F2F_RESUME_EPOCH must be an integer" >&2; exit 2;; esac
if [ "$RESUME_EPOCH" -gt 0 ]; then
  [ -n "$RESUME_RUN" ] && [ "$RESUME_EPOCH" -lt "$EPOCHS" ] || { echo "ERROR: resume needs F2F_RESUME_RUN and F2F_RESUME_EPOCH < epochs" >&2; exit 2; }
  RESUME_CKPT="$RUN_ROOT/$RESUME_RUN/model/epoch_$(printf '%03d' "$RESUME_EPOCH")"
  [ -s "$RESUME_CKPT/model.safetensors" ] && grep -qx "completed_epochs: $RESUME_EPOCH" "$RESUME_CKPT/metadata.yaml" || { echo "ERROR: resume checkpoint invalid: $RESUME_CKPT" >&2; exit 2; }
  CHECKPOINT_JSON="\"$RESUME_CKPT\""
  RESUME_JSON=$RESUME_EPOCH
else
  [ -z "$RESUME_RUN" ] || { echo "ERROR: F2F_RESUME_RUN set without F2F_RESUME_EPOCH" >&2; exit 2; }
  RESUME_CKPT=none
  CHECKPOINT_JSON=null
  RESUME_JSON=null
fi
# Rows written by THIS run (validation, history, stage logs); exposure always has EPOCHS rows.
RUN_EPOCHS=$((EPOCHS - RESUME_EPOCH))
# F2-F arms: optional MS2 fragment tokens (needs commit 7c964c6+) and an arm tag for the folder.
MS2_TOKENS=${F2F_MS2_TOKENS:-0}
case "$MS2_TOKENS" in 0) MS2_TOKENS_JSON=false;; 1) MS2_TOKENS_JSON=true;; *) echo "ERROR: F2F_MS2_TOKENS must be 0 or 1" >&2; exit 2;; esac
ARM_TAG=${F2F_ARM_TAG:-}
case "$ARM_TAG" in ''|ms2tok|rtlong|rtharm|ccsx|ccslm) ;; *) echo "ERROR: F2F_ARM_TAG must be empty, ms2tok, rtlong, rtharm, ccsx or ccslm" >&2; exit 2;; esac
# F2-H: opt-in RT target coordinate. Default = the trainer default (IntrinsicAndObserved), which is
# what every earlier F2 run used; the field is then omitted from the JSON, so old renders are unchanged.
RT_OBJECTIVE=${F2H_RT_OBJECTIVE:-}
case "$RT_OBJECTIVE" in '') RT_OBJECTIVE_JSON='';; Harmonized) RT_OBJECTIVE_JSON='"rt_objective": "Harmonized",';; *) echo "ERROR: F2H_RT_OBJECTIVE must be empty or Harmonized" >&2; exit 2;; esac
# F2-I (needs a build with 585fe9d): opt-in CCS changes. Empty/0 = the historical recipe.
MOBILITY_BATCH=${F2I_MOBILITY_BATCH:-}
case "$MOBILITY_BATCH" in '') MOBILITY_BATCH_JSON=null;; *[!0-9]*|0) echo "ERROR: F2I_MOBILITY_BATCH must be a positive integer" >&2; exit 2;; *) MOBILITY_BATCH_JSON=$MOBILITY_BATCH;; esac
CCS_LENGTH_MASS=${F2I_CCS_LENGTH_MASS:-0}
case "$CCS_LENGTH_MASS" in 0) CCS_LENGTH_MASS_JSON=false;; 1) CCS_LENGTH_MASS_JSON=true;; *) echo "ERROR: F2I_CCS_LENGTH_MASS must be 0 or 1" >&2; exit 2;; esac
EXPECTED_COMMIT=${F2F_EXPECTED_COMMIT:-82f490de1178b71dbe98c7257b4a8687b0d1f898}
if [ "$MS2_TOKENS" = 1 ] && [ "$EXPECTED_COMMIT" = 82f490de1178b71dbe98c7257b4a8687b0d1f898 ]; then
  echo "ERROR: F2F_MS2_TOKENS=1 needs F2F_EXPECTED_COMMIT pointing at a build with 7c964c6" >&2
  exit 2
fi
FROZEN_SIF_SHA256=${F2B_FROZEN_SIF_SHA256:-}
F2B_SEED=${F2B_SEED:-}
F2B_INIT_SEED=${F2B_INIT_SEED:-}
case "$F2B_SEED" in
  ''|*[!0-9]*) echo "ERROR: F2B_SEED must be a non-negative integer" >&2; exit 2;;
esac
case "$F2B_INIT_SEED" in
  default) INIT_SEED_JSON=null;;
  ''|*[!0-9]*) echo "ERROR: F2B_INIT_SEED must be an integer or 'default'" >&2; exit 2;;
  *) INIT_SEED_JSON=$F2B_INIT_SEED;;
esac
if [ -z "$FROZEN_SIF_SHA256" ]; then
  echo "ERROR: F2B_FROZEN_SIF_SHA256 is required" >&2
  exit 2
fi

SIF=${SIF:-}
EXPECTED_SIF_SHA256=${EXPECTED_SIF_SHA256:-}
EXPECTED_GIT_SHA=${EXPECTED_GIT_SHA:-}

if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "$SIF" ] || [ -z "$EXPECTED_SIF_SHA256" ] || [ -z "$EXPECTED_GIT_SHA" ]; then
  echo "ERROR: run via sbatch with SIF, EXPECTED_SIF_SHA256, EXPECTED_GIT_SHA" >&2
  exit 2
fi
if [ "$EXPECTED_GIT_SHA" != "$EXPECTED_COMMIT" ]; then
  echo "ERROR: expected exact source commit $EXPECTED_COMMIT, got $EXPECTED_GIT_SHA" >&2
  exit 2
fi
if [ ! -s "$SIF" ] || [ ! -s "$PREPARED_RUN" ]; then
  echo "ERROR: SIF or prepared run missing" >&2
  exit 2
fi
if [ "$(sha256sum "$PREPARED_RUN" | awk '{print $1}')" != "$PREPARED_RUN_SHA256" ]; then
  echo "ERROR: prepared run SHA mismatch" >&2
  exit 2
fi
ACTUAL_SIF_SHA256=$(sha256sum "$SIF" | awk '{print $1}')
if [ "$EXPECTED_SIF_SHA256" != "$FROZEN_SIF_SHA256" ] || [ "$EXPECTED_SIF_SHA256" != "$ACTUAL_SIF_SHA256" ]; then
  echo "ERROR: SIF SHA mismatch: expected=$EXPECTED_SIF_SHA256 actual=$ACTUAL_SIF_SHA256" >&2
  exit 2
fi
if ! command -v singularity >/dev/null 2>&1 || ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: Singularity and a GPU allocation are required" >&2
  exit 127
fi

SHORT=${EXPECTED_GIT_SHA:0:12}
OUT="$RUN_ROOT/${ARM_TAG:+${ARM_TAG}_}ep${EPOCHS}${RESUME_RUN:+_resume${RESUME_EPOCH}}_seed${F2B_SEED}_init${F2B_INIT_SEED}_${SHORT}_${SLURM_JOB_ID}"
MODEL="$OUT/model"
CFG="$OUT/foundation_train.json"
TRAIN_LOG="$OUT/train.log"
GPU_TSV="$OUT/gpu_usage.tsv"
GPU_STOP="$OUT/.stop_gpu_telemetry"

if [ -e "$OUT" ]; then
  echo "ERROR: output already exists: $OUT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT" || exit 2
if [ "$(df -Pk "$RUN_ROOT" | awk 'NR==2{print $4}')" -lt 20971520 ]; then
  echo "ERROR: less than 20 GiB free at $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT"
mkdir_rc=$?
if [ "$mkdir_rc" -ne 0 ]; then exit "$mkdir_rc"; fi

# The explicit fields below are essential: Serde intentionally maps omitted
# historical model/training compatibility fields to legacy defaults.
cat > "$CFG" <<JSON
{
  "version": "0.1.2",
  "train_data": "$PREPARED_RUN",
  "validation_data": null,
  "output_file": "$MODEL",
  "normalization": "min_max",
  "model_arch": "foundation",
  "device": "cuda",
  "batch_size": 8,
  "validation_batch_size": null,
  "learning_rate": 0.00002,
  "epochs": $EPOCHS,
  "early_stopping_patience": $EPOCHS,
  "checkpoint_file": $CHECKPOINT_JSON,
  "instrument": "QE",
  "nce": 20,
  "head_type": "mlp",
  "head_learnable_scaler": false,
  "train_var_prefixes": null,
  "warmup_fraction": 0.12,
  "foundation": {
    "model": {
      "peptide_backbone": "residue_pair_task_tokens",
      $RT_OBJECTIVE_JSON
      "specialists": {
        "enabled": true,
        "rt": true,
        "ms2": true,
        "ms2_residual_blend": true,
        "mobility_ccs": true,
        "ms2_fragment_tokens": $MS2_TOKENS_JSON,
        "mobility_length_mass": $CCS_LENGTH_MASS_JSON
      },
      "auxiliary_supervision_heads": true,
      "peptide": {
        "max_sequence_len": 64,
        "max_atoms_per_residue": 24,
        "atom_feature_dim": 12,
        "graph_hidden_dim": 128,
        "graph_layers": 5,
        "model_dim": 320,
        "num_attention_heads": 8,
        "transformer_ff_dim": 1280,
        "transformer_layers": 8,
        "dropout": 0.05,
        "contrastive_dim": 128,
        "instrument_vocab_size": 16,
        "ms2_fragment_channels": 8,
        "ms2_output_activation": "softplus-v0138",
        "ccs_context_mode": "neutral-mass-charge",
        "ccs_physics_baseline": null
      },
      "inverse": {
        "max_tokens": 80,
        "model_dim": 320,
        "num_attention_heads": 8,
        "feed_forward_dim": 1280,
        "spectrum_layers": 6,
        "decoder_layers": 6,
        "dropout": 0.05
      }
    },
    "ms2_loss": {
      "pointwise_weight": 1.0,
      "cosine_weight": 0.25,
      "pearson_weight": 0.25,
      "cosine_epsilon": 1e-8
    },
    "strategy": "research_curriculum",
    "batch_size": 8,
    "property_batch_size": 16,
    "mobility_batch_size": $MOBILITY_BATCH_JSON,
    "learning_rate": 0.00002,
    "epochs": $EPOCHS,
    "early_stopping_patience": $EPOCHS,
    "max_train_batches_per_epoch": 1024,
    "max_validation_batches": 256,
    "inverse_evaluation_records": 256,
    "generation_evaluation_records": 16,
    "warmup_steps": 500,
    "min_learning_rate_ratio": 0.10,
    "max_gradient_norm": 1.0,
    "mobility_consensus_supervision": true,
    "pair_auxiliary_weight": 0.10,
    "chemistry_summary_weight": 0.10,
    "conformation_proxy_weight": 0.10,
    "seed": $F2B_SEED,
    "init_seed": $INIT_SEED_JSON,
    "save_epoch_checkpoints": true,
    "resume_from_epoch": $RESUME_JSON
  }
}
JSON
config_rc=$?
if [ "$config_rc" -ne 0 ]; then
  echo "ERROR: config write failed" >&2
  exit "$config_rc"
fi

# Fail before loading millions of TRAIN records if nested Serde defaults silently
# replaced deep-model dimensions or MS2 shape-loss controls.
python3 - "$CFG" "$EPOCHS" "$MS2_TOKENS" "$RESUME_EPOCH" "$RT_OBJECTIVE" "$MOBILITY_BATCH" "$CCS_LENGTH_MASS" <<'PY_CONFIG_CHECK'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    cfg = json.load(stream)["foundation"]
model = cfg["model"]
peptide = model["peptide"]
inverse = model["inverse"]
assert model["peptide_backbone"] == "residue_pair_task_tokens"
assert model.get("rt_objective") == (sys.argv[5] or None), model.get("rt_objective")
assert model["specialists"] == {
    "enabled": True,
    "rt": True,
    "ms2": True,
    "ms2_residual_blend": True,
    "mobility_ccs": True,
    "ms2_fragment_tokens": sys.argv[3] == "1",
    "mobility_length_mass": sys.argv[7] == "1",
}
assert model["auxiliary_supervision_heads"] is True
for key, expected in {
    "max_atoms_per_residue": 24,
    "graph_hidden_dim": 128,
    "graph_layers": 5,
    "model_dim": 320,
    "num_attention_heads": 8,
    "transformer_ff_dim": 1280,
    "transformer_layers": 8,
    "ms2_output_activation": "softplus-v0138",
}.items():
    assert peptide[key] == expected, (key, peptide[key])
for key, expected in {
    "model_dim": 320, "num_attention_heads": 8,
    "feed_forward_dim": 1280, "spectrum_layers": 6,
    "decoder_layers": 6,
}.items():
    assert inverse[key] == expected, (key, inverse[key])
assert cfg["strategy"] == "research_curriculum"
assert cfg["property_batch_size"] == 16
assert cfg["mobility_batch_size"] == (int(sys.argv[6]) if sys.argv[6] else None)
assert cfg["batch_size"] == 8
assert cfg["ms2_loss"]["cosine_weight"] == 0.25
assert cfg["ms2_loss"]["pearson_weight"] == 0.25
assert cfg["ms2_loss"]["pointwise_weight"] == 1.0
assert cfg["epochs"] == int(sys.argv[2])
assert cfg["max_train_batches_per_epoch"] == 1024
assert cfg["max_validation_batches"] == 256
assert cfg["warmup_steps"] == 500
assert cfg["min_learning_rate_ratio"] == 0.10
assert cfg["max_gradient_norm"] == 1.0
assert cfg["early_stopping_patience"] == int(sys.argv[2])
assert cfg["mobility_consensus_supervision"] is True
for name in ["pair_auxiliary_weight", "chemistry_summary_weight", "conformation_proxy_weight"]:
    assert cfg[name] == 0.10, (name, cfg[name])
assert cfg["epochs"] * cfg["max_train_batches_per_epoch"] * 5 == int(sys.argv[2]) * 1024 * 5
assert isinstance(cfg["seed"], int) and cfg["save_epoch_checkpoints"] is True
assert cfg["init_seed"] is None or isinstance(cfg["init_seed"], int)
resume = int(sys.argv[4])
assert cfg["resume_from_epoch"] == (resume or None)
assert (json.load(open(sys.argv[1]))["checkpoint_file"] is None) == (resume == 0)
print("explicit_deep_research_sampling_convergence_preflight=PASS")
PY_CONFIG_CHECK
CONFIG_CHECK_RC=$?
if [ "$CONFIG_CHECK_RC" -ne 0 ]; then
  echo "ERROR: preflight rejected model/training configuration" >&2
  exit "$CONFIG_CHECK_RC"
fi

{
  printf 'key\tvalue\n'
  printf 'job_id\t%s\n' "$SLURM_JOB_ID"
  printf 'git_sha\t%s\n' "$EXPECTED_GIT_SHA"
  printf 'sif\t%s\n' "$SIF"
  printf 'sif_sha256\t%s\n' "$ACTUAL_SIF_SHA256"
  printf 'prepared_run\t%s\n' "$PREPARED_RUN"
  printf 'random_init\t%s\n' "$([ "$RESUME_EPOCH" -gt 0 ] && echo NO_RESUMED || echo YES)"
  printf 'resume_from_epoch\t%s\n' "$RESUME_EPOCH"
  printf 'resume_checkpoint\t%s\n' "$RESUME_CKPT"
  printf 'resume_adam_moments\t%s\n' "$([ "$RESUME_EPOCH" -gt 0 ] && echo RESET || echo NA)"
  printf 'prepared_run_sha256\t%s\n' "$PREPARED_RUN_SHA256"
  printf 'ms2_fragment_tokens\t%s\n' "$MS2_TOKENS"
  printf 'arm_tag\t%s\n' "${ARM_TAG:-none}"
  printf 'rt_objective\t%s\n' "${RT_OBJECTIVE:-default_IntrinsicAndObserved}"
  printf 'mobility_batch_size\t%s\n' "${MOBILITY_BATCH:-default_batch_size}"
  printf 'mobility_length_mass\t%s\n' "$CCS_LENGTH_MASS"
  printf 'data_order_seed\t%s\n' "$F2B_SEED"
  printf 'cuda_init_seed\t%s\n' "$F2B_INIT_SEED"
  printf 'recipe_parent_job\t62709585\n'
  printf 'backbone\tresidue_pair_task_tokens\n'
    printf 'native_specialists\tRT,mobility_CCS,MS2 (gated)\n'
  printf 'auxiliary_heads\tYES\n'
  printf 'ms2_residual_blend\ttrue\n'
  printf 'checkpoint_initialization\tNO\n'
  printf 'source_sampling_gate_job\t62685397\n'
  printf 'training_strategy\tresearch_curriculum\n'
  printf 'mobility_consensus_supervision\tYES\n'
  printf 'batch_size\t8\n'
  printf 'property_batch_size\t16\n'
  printf 'property_records_per_epoch\t32768\n'
  printf 'property_records_total_selected\t%s\n' "$((EPOCHS * 32768))"
  printf 'epochs\t%s\n' "$EPOCHS"
  printf 'max_train_cycles_per_epoch\t1024\n'
  printf 'planned_optimizer_updates\t%s\n' "$UPDATES"
  printf 'max_validation_batches\t256\n'
  printf 'expected_forward_validation_records\t2048\n'
  printf 'ms2_activation\tsoftplus-v0138\n'
  printf 'ms2_cosine_weight\t0.25\n'
  printf 'ms2_pearson_weight\t0.25\n'
  printf 'planned_stage_visits\t%s\n' "$UPDATES"
  printf 'lr_warmup_updates\t500\n'
  printf 'historical_test_consumed\tNO\n'
  printf 'train_holdout_consumed\tNO\n'
  printf 'output\t%s\n' "$OUT"
} > "$OUT/provenance.tsv"

scontrol show job -dd "$SLURM_JOB_ID" > "$OUT/slurm_job_snapshot.txt" 2>&1
singularity inspect --json "$SIF" > "$OUT/sif_inspect.json" 2>&1
nvidia-smi -L > "$OUT/gpu_inventory.txt" 2>&1
printf 'timestamp,index,name,gpu_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_w\n' > "$GPU_TSV"
(
  while [ ! -f "$GPU_STOP" ]; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits >> "$GPU_TSV" 2>> "$OUT/gpu_usage.err"
    sleep 10
  done
) &
GPU_PID=$!

printf '===== RESEARCH PROPERTY-SAMPLING CONVERGENCE =====\n'
printf 'job_id=%s\nsource_git_sha=%s\nsif_sha256=%s\noutput=%s\n' \
  "$SLURM_JOB_ID" "$EXPECTED_GIT_SHA" "$ACTUAL_SIF_SHA256" "$OUT"
printf 'historical_test_consumed=NO\ntrain_holdout_consumed=NO\n'
printf 'total_optimizer_updates_planned=%s\n' "$UPDATES"
cat "$OUT/gpu_inventory.txt"

/usr/bin/time -v -o "$OUT/train.time.txt" \
  singularity exec --nv --bind "$ROOT:$ROOT" "$SIF" \
    redeem properties train "$CFG" \
  2>&1 | tee "$TRAIN_LOG"
TRAIN_RC=${PIPESTATUS[0]}

touch "$GPU_STOP"
wait "$GPU_PID"
GPU_RC=$?
rm -f "$GPU_STOP"
if [ "$GPU_RC" -ne 0 ]; then
  echo "WARNING: GPU telemetry process exited with rc=$GPU_RC" >&2
fi

awk -F',' '
  NR == 1 { next }
  { for (i=1; i<=NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
    count++; util=$4+0; used=$5+0; total=$6+0
    usum+=util; msum+=used
    if (count == 1 || util > umax) umax=util
    if (count == 1 || used > mmax) mmax=used
    if (total > mtot) mtot=total
  }
  END { print "metric\tvalue"; print "samples\t" count
    if (count > 0) {
      printf "mean_gpu_util_percent\t%.3f\n",usum/count
      printf "max_gpu_util_percent\t%.3f\n",umax
      printf "mean_gpu_memory_used_mib\t%.3f\n",msum/count
      printf "max_gpu_memory_used_mib\t%.3f\n",mmax
      printf "gpu_memory_total_mib\t%.3f\n",mtot
      if (mtot > 0) printf "max_gpu_memory_util_percent\t%.3f\n",100*mmax/mtot
    }
  }' "$GPU_TSV" > "$OUT/gpu_usage_summary.tsv"

if [ "$TRAIN_RC" -ne 0 ]; then
  echo "foundation_research_sampling_convergence=FAIL rc=$TRAIN_RC" >&2
  echo "output=$OUT" >&2
  exit "$TRAIN_RC"
fi

# Prove non-repeated, exposure-aware property-stage sampling across epochs.
EXPOSURE="$MODEL/training_exposure.tsv"
if [ ! -s "$EXPOSURE" ]; then
  echo "ERROR: missing property training exposure report" >&2
  exit 2
fi
if ! awk -F '\t' '
  NR==1 {for(i=1;i<=NF;i++) h[$i]=i; next}
  NR>=2 {
    n=split("epoch cycles property_updates property_batch_size property_records new_property_records cumulative_unique_property_records train_records rt_labelled ms2_labelled ccs_labelled inverse_spectrum_records", required," ");
    for(i=1;i<=n;i++) if(!h[required[i]] || $h[required[i]]=="") bad=1;
    if(($h["epoch"]+0)!=NR-1 || ($h["cycles"]+0)!=1024 ||
       ($h["property_updates"]+0)!=2048 || ($h["property_batch_size"]+0)!=16 ||
       ($h["property_records"]+0)!=32768 || ($h["train_records"]+0)<3650000 ||
       ($h["rt_labelled"]+0)<1 || ($h["ms2_labelled"]+0)<1 ||
       ($h["ccs_labelled"]+0)<1 || ($h["inverse_spectrum_records"]+0)<1) bad=1;
    cur=$h["cumulative_unique_property_records"]+0;
    fresh=$h["new_property_records"]+0;
    if(cur<prior || cur-prior!=fresh || fresh<=0 || fresh>32768) bad=1;
    prior=cur; rows++;
  }
  END {
    printf "property_sampling_epochs=%d\n",rows;
    printf "property_sampling_cumulative_unique=%d\n",prior;
    if(bad || rows!=epochs || prior<280000) exit 1;
  }
' epochs="$EPOCHS" "$EXPOSURE"; then
  echo "ERROR: research property exposure contract failed" >&2
  cat "$EXPOSURE" >&2
  exit 2
fi
if [ "$(grep -c '^foundation_stage_losses' "$TRAIN_LOG")" -ne "$RUN_EPOCHS" ]; then
  echo "ERROR: missing per-stage research curriculum losses" >&2
  exit 2
fi
if [ "$(grep -c '^foundation_stage_gradients' "$TRAIN_LOG")" -ne "$RUN_EPOCHS" ]; then
  echo "ERROR: missing per-stage gradient diagnostics" >&2
  exit 2
fi
if [ "$RESUME_EPOCH" -gt 0 ] && ! grep -q "^foundation_resume from_epoch=$RESUME_EPOCH global_update=$((RESUME_EPOCH * 1024 * 5)) planned_updates=$UPDATES " "$TRAIN_LOG"; then
  echo "ERROR: trainer did not report the expected resume point" >&2
  exit 2
fi
for e in $(seq -f '%03g' $((RESUME_EPOCH + 1)) "$EPOCHS"); do
  if [ ! -s "$MODEL/epoch_$e/model.safetensors" ] || [ ! -s "$MODEL/epoch_$e/metadata.yaml" ]; then
    echo "ERROR: missing per-epoch checkpoint epoch_$e" >&2
    exit 2
  fi
done
for spec in "seed: $F2B_SEED" "save_epoch_checkpoints: true"; do
  if ! grep -Fqx "$spec" "$MODEL/training.yaml"; then
    echo "ERROR: saved training config did not record $spec" >&2
    exit 2
  fi
done
# The token arm must really be active: an older binary would silently ignore the field.
if [ "$MS2_TOKENS" = 1 ] && ! grep -Eq '^\s*ms2_fragment_tokens:\s*true\s*$' "$MODEL/metadata.yaml"; then
  echo "ERROR: ms2_fragment_tokens=true requested but not recorded in model metadata" >&2
  exit 2
fi
printf 'research_property_sampling_exposure=PASS\n'
if [ -n "$RT_OBJECTIVE" ] && ! grep -Eq "^\s*rt_objective:\s*$RT_OBJECTIVE\s*$" "$MODEL/metadata.yaml"; then
  echo "ERROR: rt_objective=$RT_OBJECTIVE requested but not recorded in model metadata" >&2
  exit 2
fi

# Preserve the deterministic Validation prefix (2048 records); explicitly
# check RT/CCS/MS2 support before interpreting best-forward checkpointing.
if ! awk -F '\t' '
  NR==1 { for (i=1;i<=NF;i++) col[$i]=i; next }
  NR>=2 {
    if (!col["records"] || !col["rt_count"] || !col["ccs_count"] ||
        !col["ms2_spectrum_count"]) exit 1;
    if (($col["records"]+0)!=2048 || ($col["rt_count"]+0)<1 ||
        ($col["ccs_count"]+0)<1 || ($col["ms2_spectrum_count"]+0)<1) exit 1;
    rows++;
  }
  END { if (rows != epochs) exit 1 }
' epochs="$RUN_EPOCHS" "$MODEL/validation_metrics.tsv"; then
  echo "ERROR: insufficient Validation forward-label coverage; best_forward is not scoreable" >&2
  cat "$MODEL/validation_metrics.tsv" >&2
  exit 2
fi

# Detect renewed zero-spectrum collapse without confusing the research target
# with an engineering-gate pass. All epochs require defined metrics; at least
# one epoch must show useful MS2 signal. Nonmonotone learning is allowed.
if ! awk -F '\t' '
  NR==1 { for(i=1;i<=NF;i++) col[$i]=i; next }
  NR>=2 {
    keys[1]="ms2_cosine_similarity"; keys[2]="ms2_spectral_angle";
    keys[3]="ms2_pearson"; keys[4]="ms2_rmse";
    keys[5]="ms2_spectrum_count";
    for(i=1;i<=5;i++) {key=keys[i]; if(!col[key] || $col[key]=="") bad=1}
    cosine=$col["ms2_cosine_similarity"]+0;
    angle=$col["ms2_spectral_angle"]+0;
    pearson=$col["ms2_pearson"]+0;
    rmse=$col["ms2_rmse"]+0;
    spectra=$col["ms2_spectrum_count"]+0;
    if(spectra<1 || rmse<=0) bad=1;
    if(cosine>best_cosine) best_cosine=cosine;
    if(cosine>0.20 && angle>0.20 && pearson>0 && rmse<0.54) useful++;
    count++;
  }
  END {
    printf "ms2_convergence_epoch_count=%d\n",count;
    printf "ms2_convergence_best_cosine=%.8f\n",best_cosine;
    printf "ms2_convergence_useful_epochs=%d\n",useful;
    if(bad || count!=epochs || useful<1) exit 1;
  }
' epochs="$RUN_EPOCHS" "$MODEL/validation_metrics.tsv"; then
  echo "ERROR: gated MS2 convergence lost useful spectral signal or missing metrics" >&2
  cat "$MODEL/validation_metrics.tsv" >&2
  exit 2
fi
printf 'ms2_noncollapse_gate=PASS\n'


for path in \
  "$MODEL/model.safetensors" \
  "$MODEL/metadata.yaml" \
  "$MODEL/training.yaml" \
  "$MODEL/training_history.tsv" \
  "$MODEL/validation_metrics.tsv" \
  "$MODEL/mobility_source_reliability.tsv" \
  "$MODEL/mobility_consensus_summary.tsv" \
  "$MODEL/best_forward/model.safetensors" \
  "$MODEL/best_inverse/model.safetensors"; do
  if [ ! -s "$path" ]; then
    echo "ERROR: missing or empty learning artifact: $path" >&2
    exit 2
  fi
done

for spec in \
  'peptide_backbone:[[:space:]]*residue_pair_task_tokens' \
  'auxiliary_supervision_heads:[[:space:]]*true' \
  'enabled:[[:space:]]*true' \
  'mobility_ccs:[[:space:]]*true'; do
  if ! grep -Eq "$spec" "$MODEL/metadata.yaml"; then
    echo "ERROR: checkpoint model metadata did not record $spec" >&2
    exit 2
  fi
done
if ! grep -Eq 'ms2:[[:space:]]*true' "$MODEL/metadata.yaml"; then
  echo "ERROR: saved checkpoint does not enable native MS2 specialist" >&2
  exit 2
fi
if ! grep -Eq 'ms2_residual_blend:[[:space:]]*true' "$MODEL/metadata.yaml"; then
  echo "ERROR: saved checkpoint does not enable gated pair-head preservation" >&2
  exit 2
fi
if ! grep -Eq 'ms2_output_activation:[[:space:]]*softplus-v0138' "$MODEL/metadata.yaml"; then
  echo "ERROR: saved checkpoint is NOT configured for Softplus MS2" >&2
  exit 2
fi
for spec in \
  'cosine_weight:[[:space:]]*0.25' \
  'pearson_weight:[[:space:]]*0.25'; do
  if ! grep -Eq "$spec" "$MODEL/training.yaml"; then
    echo "ERROR: saved training config missing MS2 shape loss $spec" >&2
    exit 2
  fi
done

for spec in \
  'strategy:[[:space:]]*research_curriculum' \
  'property_batch_size:[[:space:]]*16' \
  'mobility_consensus_supervision:[[:space:]]*true' \
  'pair_auxiliary_weight:[[:space:]]*0.1' \
  'chemistry_summary_weight:[[:space:]]*0.1' \
  'conformation_proxy_weight:[[:space:]]*0.1'; do
  if ! grep -Eq "$spec" "$MODEL/training.yaml"; then
    echo "ERROR: saved training config did not record $spec" >&2
    exit 2
  fi
done

# Assert nonzero forward and inverse training losses for every completed epoch.
if ! awk -F '\t' 'NR==1 { for(i=1;i<=NF;i++) col[$i]=i; next }
    NR>=2 {if(!(col["train_forward"] && col["train_inverse"])) exit 1;
      if(($col["train_forward"]+0)<=0 || ($col["train_inverse"]+0)<=0) exit 1;
      rows++} END{if(rows != epochs) exit 1}' epochs="$RUN_EPOCHS" "$MODEL/training_history.tsv"; then
  echo "ERROR: forward/inverse training losses not both positive; five-stage exercise not demonstrated" >&2
  exit 2
fi
if ! awk -F '\t' '$1=="train_consensus_examples" && ($2+0)>0 {found=1} END{if(!found) exit 1}' \
  "$MODEL/mobility_consensus_summary.tsv"; then
  echo "ERROR: zero source-aware consensus examples" >&2
  exit 2
fi
if ! awk -F '\t' 'NR==1 {for(i=1;i<=NF;i++) col[$i]=i; next}
  NR>=2 {if(!col["inverse_sequence_count"] || ($col["inverse_sequence_count"]+0)<=0) exit 1;
         rows++} END{if(rows != epochs) exit 1}' epochs="$RUN_EPOCHS" "$MODEL/validation_metrics.tsv"; then
  echo "ERROR: inverse Validation evaluation did not exercise spectrum-bearing records" >&2
  exit 2
fi

# Reload the just-saved checkpoint via the public production inference CLI.
cat > "$OUT/foundation_inference.json" <<JSON
{
  "model_path": "$MODEL",
  "inference_data": "$PREPARED_RUN",
  "output_file": "$OUT/validation_predictions.tsv",
  "model_arch": "foundation",
  "device": "cuda",
  "batch_size": 4,
  "foundation_partition": "validation",
  "foundation_max_records": 16,
  "foundation_spectra_only": false
}
JSON
singularity exec --nv --bind "$ROOT:$ROOT" "$SIF" \
  redeem properties inference "$OUT/foundation_inference.json" \
  > "$OUT/inference.log" 2>&1
INFERENCE_RC=$?
if [ "$INFERENCE_RC" -ne 0 ]; then
  echo "ERROR: checkpoint reload / Validation inference failed rc=$INFERENCE_RC" >&2
  cat "$OUT/inference.log" >&2
  exit "$INFERENCE_RC"
fi
if ! grep -q '^foundation_inference=PASS' "$OUT/inference.log"; then
  echo "ERROR: inference CLI did not emit PASS" >&2
  cat "$OUT/inference.log" >&2
  exit 2
fi
if ! awk 'END{if(NR != 17) exit 1}' "$OUT/validation_predictions.tsv"; then
  echo "ERROR: unexpected Validation inference output size (expected header + 16 records)" >&2
  exit 2
fi

# Read back the independently selected balanced-forward and inverse checkpoints.
for selected in best_forward best_inverse; do
  cat > "$OUT/inference_${selected}.json" <<JSON
{
  "model_path": "$MODEL/$selected",
  "inference_data": "$PREPARED_RUN",
  "output_file": "$OUT/validation_predictions_${selected}.tsv",
  "model_arch": "foundation",
  "device": "cuda",
  "batch_size": 4,
  "foundation_partition": "validation",
  "foundation_max_records": 16,
  "foundation_spectra_only": false
}
JSON
  singularity exec --nv --bind "$ROOT:$ROOT" "$SIF" \
    redeem properties inference "$OUT/inference_${selected}.json" \
    > "$OUT/inference_${selected}.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] || ! grep -q '^foundation_inference=PASS' "$OUT/inference_${selected}.log"; then
    echo "ERROR: $selected checkpoint inference failed rc=$rc" >&2
    cat "$OUT/inference_${selected}.log" >&2
    exit 2
  fi
  if ! awk 'END{if(NR != 17) exit 1}' "$OUT/validation_predictions_${selected}.tsv"; then
    echo "ERROR: unexpected $selected Validation inference output size" >&2
    exit 2
  fi
done

# Report the actual forward-balanced selection, not independent minima
# assembled from different epochs. All metrics below come from Validation.
python3 - "$MODEL/validation_metrics.tsv" "$MODEL/metadata.yaml" "$MODEL/best_forward/metadata.yaml" "$MODEL/best_inverse/metadata.yaml" "$OUT/convergence_summary.tsv" "$RUN_EPOCHS" <<'PY_SUMMARY'
import csv, math, pathlib, re, sys
metrics, main_yaml, forward_yaml, inverse_yaml, dest = map(pathlib.Path, sys.argv[1:6])
with metrics.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != int(sys.argv[6]):
    raise SystemExit("unexpected number of Validation epochs")
# Rows are keyed by their epoch column (a resumed run starts after epoch 1).
by_epoch = {int(r["epoch"]): r for r in rows}
last_epoch = max(by_epoch)

def selected_epoch(path):
    txt = path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^completed_epochs:\s*(\d+)", txt)
    if not match:
        raise SystemExit(f"missing completed_epochs in {path}")
    epoch = int(match.group(1))
    if epoch not in by_epoch:
        raise SystemExit(f"invalid selected epoch {epoch} in {path}")
    return epoch

tep = selected_epoch(main_yaml)
fep = selected_epoch(forward_yaml)
iep = selected_epoch(inverse_yaml)
fields = ["rt_mae_native", "ccs_mae", "ms2_cosine_similarity", "ms2_spectral_angle", "ms2_pearson", "ms2_rmse", "inverse_mean_nll", "retrieval_mrr"]
with dest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["checkpoint", "epoch", *fields])
    for label, epoch in [("best_total", tep), ("best_forward", fep), ("best_inverse", iep), ("final_validation_only", last_epoch)]:
        row = by_epoch[epoch]
        values = []
        for field in fields:
            val = row.get(field, "")
            if val == "" or not math.isfinite(float(val)):
                raise SystemExit(f"invalid {field} at epoch {epoch}")
            values.append(val)
        writer.writerow([label, epoch, *values])
print("selected_total_epoch=", tep, sep="")
print("selected_forward_epoch=", fep, sep="")
print("selected_inverse_epoch=", iep, sep="")
print("selected_checkpoint_metric_summary=PASS")
PY_SUMMARY
SUMMARY_RC=$?
if [ "$SUMMARY_RC" -ne 0 ]; then
  echo "ERROR: cannot audit selected checkpoint metrics" >&2
  exit "$SUMMARY_RC"
fi

: > "$OUT/output_sha256.txt"
for path in \
  "$CFG" \
  "$OUT/foundation_inference.json" \
  "$OUT/validation_predictions.tsv" \
  "$OUT/inference.log" \
  "$OUT/inference_best_forward.json" \
  "$OUT/validation_predictions_best_forward.tsv" \
  "$OUT/inference_best_forward.log" \
  "$OUT/inference_best_inverse.json" \
  "$OUT/validation_predictions_best_inverse.tsv" \
  "$OUT/inference_best_inverse.log" \
  "$OUT/provenance.tsv" "$OUT/train.log" "$OUT/train.time.txt" \
  "$OUT/convergence_summary.tsv" \
  "$GPU_TSV" "$OUT/gpu_usage_summary.tsv" \
  "$MODEL/model.safetensors" "$MODEL/metadata.yaml" "$MODEL/training.yaml" \
  "$MODEL/training_history.tsv" "$MODEL/validation_metrics.tsv" \
  "$MODEL/training_exposure.tsv" \
  "$MODEL/mobility_consensus_summary.tsv" "$MODEL/mobility_source_reliability.tsv" \
  "$MODEL/best_forward/model.safetensors" "$MODEL/best_inverse/model.safetensors" \
  "$MODEL"/epoch_0*/model.safetensors; do
  sha256sum "$path" >> "$OUT/output_sha256.txt"
  hash_rc=$?
  if [ "$hash_rc" -ne 0 ]; then exit "$hash_rc"; fi
done

printf '\n===== SELECTED FORWARD/INVERSE METRICS =====\n'
cat "$OUT/convergence_summary.tsv"
printf '\n===== PROPERTY TRAIN EXPOSURE =====\n'
cat "$MODEL/training_exposure.tsv"
printf '\n===== TRAINING HISTORY =====\n'
cat "$MODEL/training_history.tsv"
printf '\n===== VALIDATION METRICS =====\n'
cat "$MODEL/validation_metrics.tsv"
printf '\n===== MOBILITY CONSENSUS =====\n'
cat "$MODEL/mobility_consensus_summary.tsv"
printf '\n===== GPU SUMMARY =====\n'
cat "$OUT/gpu_usage_summary.tsv"
printf '\n===== SELECTED CHECKPOINT METADATA =====\n'
for selected in "$MODEL" "$MODEL/best_forward" "$MODEL/best_inverse"; do
  echo "checkpoint_directory=$selected"
  grep -E '^completed_epochs:|^best_validation_loss:' "$selected/metadata.yaml" || true
done
printf '\n===== OUTPUT HASHES =====\n'
cat "$OUT/output_sha256.txt"
printf '\nfoundation_f2b_seed_replicate=PASS\n'
printf 'SCIENTIFIC_ADVANCEMENT=NOT_DECIDED_FROM_ENGINEERING_PASS\n'
printf 'historical_test_consumed=NO\ntrain_holdout_consumed=NO\n'
printf 'output=%s\n' "$OUT"
