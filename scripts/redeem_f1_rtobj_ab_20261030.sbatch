#!/usr/bin/env bash
#SBATCH --job-name=redeem-f1-rtobj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/redeem-f1-rtobj-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/redeem-f1-rtobj-%j.err
# F1-D paired RT-objective A/B, one arm per job. Derived from
# redeem_unified_disjoint512_learning_curve_20261020.sbatch (SHA a45c0098...).
# Changes: new commit/SIF pins, full 2686289 model block + rt_objective per arm (env F1_ARM),
# arm-aware auditor, new output root. Protocol otherwise identical: random init, 8 x 512,
# disjoint property batches of 8, seed 20261006. DEV/Validation only. No TRAIN-HOLDOUT or TEST.

main() {
  local root='/nfs/research/juan/DIA/singj/foundation_model'
  local exp="$root/experiments/redeem_f1_rtobj_ab_20261030"
  local source_sha='a832c3a40225149a9ce84779f796977768be9e70'
  local run="$root/experiments/multimodal_v0261_03_openptm_fragmentgeometryfix2/prepared/run_v0260.yaml"
  local expected_run_sha='7214b9cdd3836ec028655d5d6b8bb088280558626b24c9afc63ec06d582b4201'
  local report="$root/scripts/redeem_disjoint512_learning_curve_report_20261020.py"
  local expected_report_sha='16864361a0a99dff98ffc7949c42fc7a66b1868a86eed7281d5c683b3eba4e50'
  local audit="$exp/scripts/redeem_f1_rtobj_audit_20261030.py"
  local model_block="$exp/scripts/model_block_2686289.json"
  local expected_block_sha='3f53c0c85d2d3ef58bd6ca1af6020cbce01e6f03e375b5493eed427edc0aff25'
  local objective out config model_dir gpu_name checkpoint tag inf_cfg inf_tsv
  if [ -z "${SLURM_JOB_ID:-}" ]; then echo 'ERROR: submit through sbatch' >&2; return 2; fi
  case "${F1_ARM:-}" in
    A) objective='IntrinsicAndObserved';;
    B) objective='Harmonized';;
    *) echo 'ERROR: F1_ARM must be A or B' >&2; return 2;;
  esac
  if [ "${SOURCE_SHA:-}" != "$source_sha" ] || [ -z "${EXPECTED_SIF_SHA256:-}" ] || [ -z "${EXPECTED_AUDIT_SHA256:-}" ]; then
    echo 'ERROR: SOURCE_SHA, EXPECTED_SIF_SHA256 or EXPECTED_AUDIT_SHA256 missing or wrong' >&2; return 2
  fi
  if [ -z "${SIF:-}" ] || [ ! -s "$SIF" ] || [ "$(sha256sum "$SIF" | awk '{print $1}')" != "$EXPECTED_SIF_SHA256" ]; then
    echo 'ERROR: SIF missing or SHA mismatch' >&2; return 2
  fi
  for pair in "$run:$expected_run_sha" "$report:$expected_report_sha" "$audit:$EXPECTED_AUDIT_SHA256" "$model_block:$expected_block_sha"; do
    if [ ! -s "${pair%%:*}" ] || [ "$(sha256sum "${pair%%:*}" | awk '{print $1}')" != "${pair##*:}" ]; then
      echo "ERROR: pinned input missing or SHA mismatch: ${pair%%:*}" >&2; return 2
    fi
  done
  export F1_SIF_SHA256="$EXPECTED_SIF_SHA256" F1_RT_OBJECTIVE="$objective"
  if ! python3 "$audit" --self-test || ! python3 "$report" --self-test; then
    echo 'ERROR: auditor or reporter self-test failed' >&2; return 2
  fi
  if ! command -v singularity >/dev/null 2>&1; then
    if ! module load singularityce/4.1.0; then echo 'ERROR: SingularityCE cannot be loaded' >&2; return 2; fi
  fi
  if ! command -v singularity >/dev/null 2>&1 || ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 'ERROR: SingularityCE/CUDA unavailable' >&2; return 2
  fi
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  if [[ "$gpu_name" != *A100* ]]; then echo "ERROR: expected A100, got $gpu_name" >&2; return 2; fi
  # Output root is selectable because /nfs/research (gpfs) filled to 100% during 14976065/6.
  local out_root="${F1_OUT_ROOT:-$exp/runs}"
  if ! mkdir -p "$out_root"; then echo "ERROR: cannot create $out_root" >&2; return 2; fi
  if [ "$(df -Pk "$out_root" | awk 'NR==2{print $4}')" -lt 5242880 ]; then
    echo "ERROR: less than 5 GiB free at $out_root" >&2; return 2
  fi
  out="$out_root/arm${F1_ARM}_${source_sha:0:12}_${SLURM_JOB_ID}"
  if [ -e "$out" ]; then echo "ERROR: refusing to overwrite $out" >&2; return 2; fi
  if ! mkdir -p "$out"; then echo 'ERROR: unable to create output directory' >&2; return 2; fi
  config="$out/config.json"
  model_dir="$out/model"
  printf 'source_sha=%s\nsif_path=%s\nsif_sha256=%s\nrun_yaml=%s\nrun_yaml_sha256=%s\ngpu_name=%s\nseed=%s\nepochs=%s\ntrain_cycles_per_epoch=%s\nvalidation_max_records=%s\ninverse_evaluation_max_records=%s\ngeneration_attempts_per_epoch=%s\nf1_arm=%s\nrt_objective=%s\nhistorical_holdout_selected=NO\nhistorical_test_selected=NO\n' \
    "$source_sha" "$SIF" "$EXPECTED_SIF_SHA256" "$run" "$expected_run_sha" "$gpu_name" '20261006' '8' '512' '1024' '256' '64' "$F1_ARM" "$objective" > "$out/provenance.txt"
  sha256sum "$report" "$audit" "$model_block" > "$out/report_script_sha256.txt"
  nvidia-smi > "$out/gpu_before.txt" 2>&1
  if ! python3 - "$config" "$run" "$model_dir" "$model_block" "$objective" <<'PY'
import json,sys
path,run,model,block,objective = sys.argv[1:]
with open(block) as handle: model_block = json.load(handle)
model_block['rt_objective'] = objective
config = dict(model_arch='foundation', train_data=run, validation_data=None,
    output_file=model, checkpoint_file=None, device='cuda',
    batch_size=8, learning_rate=2e-5, epochs=8, early_stopping_patience=8,
    foundation=dict(model=model_block, batch_size=8, epochs=8, early_stopping_patience=8,
        seed=20261006, learning_rate=2e-5, strategy='research_curriculum',
        warmup_steps=10, min_learning_rate_ratio=0.1, max_gradient_norm=1.0,
        mobility_consensus_supervision=True, pair_auxiliary_weight=0.10,
        chemistry_summary_weight=0.10, conformation_proxy_weight=0.10,
        max_train_batches_per_epoch=512, property_batch_size=8, max_validation_batches=128,
        inverse_evaluation_records=256, generation_evaluation_records=64))
with open(path, 'w') as handle:
    json.dump(config, handle, sort_keys=True, indent=2)
PY
  then echo 'ERROR: training JSON generation failed' >&2; return 3; fi
  # Verify all intended inputs before spending GPU time (original preflight + arm/model block).
  if ! python3 - "$config" "$run" "$model_dir" "$model_block" "$objective" <<'PY'
import json,sys
path,run,model,block,objective=sys.argv[1:]
with open(path) as f: c=json.load(f)
with open(block) as f: b=json.load(f)
f=c['foundation']
assert c['model_arch']=='foundation' and c['train_data']==run and c['validation_data'] is None
assert c['output_file']==model and c['checkpoint_file'] is None and c['device']=='cuda'
assert c['batch_size']==8 and c['epochs']==8 and c['early_stopping_patience']==8
assert f['strategy']=='research_curriculum' and f['max_train_batches_per_epoch']==512
assert f['batch_size']==8 and f['property_batch_size']==8 and f['max_validation_batches']==128
assert f['epochs']==8 and f['inverse_evaluation_records']==256 and f['generation_evaluation_records']==64
assert f['seed']==20261006 and f['warmup_steps']==10 and f['learning_rate']==2e-5
assert f['min_learning_rate_ratio']==0.1 and f['max_gradient_norm']==1.0
assert f['mobility_consensus_supervision'] is True
assert all(f[k]==0.1 for k in ['pair_auxiliary_weight','chemistry_summary_weight','conformation_proxy_weight'])
m=dict(f['model']); assert m.pop('rt_objective')==objective and m==b
print(f'F1_RTOBJ_CONFIG_PREFLIGHT=PASS rt_objective={objective}')
PY
  then echo 'ERROR: training config preflight failed' >&2; return 3; fi
  sha256sum "$config" > "$out/config_sha256.txt"
  if ! singularity exec --nv --bind "$root:$root" --bind "$out_root:$out_root" "$SIF" \
      /opt/redeem/bin/redeem properties train "$config" > "$out/train.log" 2>&1; then
    echo 'ERROR: training failed; tail of log:' >&2; tail -n 100 "$out/train.log" >&2; return 5
  fi
  if grep -q "Config Invalid value for 'foundation'" "$out/train.log"; then
    echo 'ERROR: CLI rejected the foundation block and used defaults' >&2; return 5
  fi
  if [ "$(sha256sum "$run" | awk '{print $1}')" != "$expected_run_sha" ]; then
    echo 'ERROR: training source YAML changed during the run' >&2; return 5
  fi
  if ! python3 "$report" --out "$out" > "$out/report_audit.log" 2>&1; then
    echo 'ERROR: epoch denominators/checkpoint/milestone audit failed' >&2; tail -n 80 "$out/report_audit.log" >&2; return 5
  fi
  cat "$out/report_audit.log"
  for tag in best_validation best_forward best_inverse; do
    case "$tag" in
      best_validation) checkpoint="$model_dir";;
      best_forward) checkpoint="$model_dir/best_forward";;
      best_inverse) checkpoint="$model_dir/best_inverse";;
    esac
    if [ ! -s "$checkpoint/model.safetensors" ] || [ ! -s "$checkpoint/metadata.yaml" ]; then
      echo "ERROR: missing $tag checkpoint" >&2; return 5
    fi
    inf_cfg="$out/reload_${tag}.json"
    inf_tsv="$out/reload_${tag}.tsv"
    if ! python3 - "$inf_cfg" "$run" "$checkpoint" "$inf_tsv" <<'PY'
import json,sys
path,run,model,output=sys.argv[1:]
cfg=dict(model_arch='foundation',device='cuda',model_path=model,
    inference_data=run,output_file=output,batch_size=8,
    foundation_backend='unified',foundation_partition='validation',
    foundation_max_records=8,foundation_spectra_only=False)
with open(path,'w') as f: json.dump(cfg,f,sort_keys=True,indent=2)
PY
    then echo "ERROR: failed to write inference config $tag" >&2; return 5; fi
    if ! singularity exec --nv --bind "$root:$root" --bind "$out_root:$out_root" "$SIF" \
      /opt/redeem/bin/redeem properties inference "$inf_cfg" > "$out/reload_${tag}.log" 2>&1; then
      echo "ERROR: checkpoint reload failed for $tag" >&2; tail -n 80 "$out/reload_${tag}.log" >&2; return 5
    fi
    if ! grep -Fqx 'foundation_inference=PASS' "$out/reload_${tag}.log" || [ "$(wc -l < "$inf_tsv")" -ne 9 ]; then
      echo "ERROR: $tag inference did not return exactly 8 Validation records" >&2; return 5
    fi
    echo "LEARNING_CURVE_CHECKPOINT_RELOAD=PASS selector=$tag records=8"
  done
  if ! python3 "$audit" --model-dir "$model_dir" --provenance "$out/provenance.txt" \
      --source-yaml "$run" --json-report "$out/rtobj_audit.json" > "$out/independent_audit.log" 2>&1; then
    echo 'ERROR: independent protocol/coverage/arm audit failed' >&2; tail -n 100 "$out/independent_audit.log" >&2; return 5
  fi
  cat "$out/independent_audit.log"
  nvidia-smi > "$out/gpu_after.txt" 2>&1
  find "$out" -maxdepth 3 -type f ! -name output_sha256.tsv -print0 | xargs -0 -r sha256sum > "$out/output_sha256.tsv"
  echo "F1_RTOBJ_ARM=PASS arm=$F1_ARM rt_objective=$objective out=$out"
  echo 'SCIENTIFIC_ADVANCEMENT=NOT_DECIDED_FROM_ENGINEERING_PASS'
  return 0
}
main "$@"
