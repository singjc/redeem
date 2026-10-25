#!/usr/bin/env bash
#SBATCH --job-name=redeem-c2-opt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/nfs/research/juan/DIA/singj/foundation_model/experiments/redeem_c2_optional_inference_20261025/slurm-%j.out
#SBATCH --error=/nfs/research/juan/DIA/singj/foundation_model/experiments/redeem_c2_optional_inference_20261025/slurm-%j.err
# Synthetic-free label-blind SMOKE from eight already selected TRAIN spectra.
# This IS NOT an accuracy experiment; do not read TRAIN truth or DEV/TEST labels.
ROOT='/nfs/research/juan/DIA/singj/foundation_model'
BASE="$ROOT/experiments/redeem_c2_optional_inference_20261025"
CODE="$BASE/code"
SIF="$ROOT/containers/redeem-foundation_v0770-streaming-identifier_f7b8246c7730.sif"
CAT="$ROOT/experiments/redeem_pt2017_synthesis_newdev512_20261023/input/catalog.yaml"
REQUESTS="$ROOT/experiments/redeem_c2_train_only_20261023/train_2048/requests.tsv"
MODEL="$BASE/model/model.json"
RUN="$BASE/runs/smoke_${SLURM_JOB_ID:-UNKNOWN}"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo 'ERROR: no Slurm job identifier'
  false
elif [ -e "$RUN" ]; then
  echo 'ERROR: smoke output exists; preserve it'
  false
elif ! (cd "$CODE" && sha256sum -c SHA256_CODE.txt); then
  echo 'ERROR: optional-inference source manifest invalid'
  false
elif ! printf '%s  %s\n' '47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca' "$MODEL" | sha256sum -c -; then
  echo 'ERROR: frozen trained model does not match'
  false
elif ! printf '%s  %s\n' '26dc26a242c5cb039bdba91d7cbfa0ea6b0e14f27777ba1f54f0a8e7bf2c5a25' "$REQUESTS" | sha256sum -c -; then
  echo 'ERROR: original TRAIN requests were modified'
  false
elif ! printf '%s  %s\n' 'a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f' "$CAT" | sha256sum -c -; then
  echo 'ERROR: published synthesis catalog modified'
  false
elif ! printf '%s  %s\n' 'd46ed2bcc39ed50dec081762788ac60f4f3551658d44b555ce0aacf33504bf22' "$SIF" | sha256sum -c -; then
  echo 'ERROR: frozen v0.77 SIF modified'
  false
elif [ -e "$BASE/truth.tsv" ] || [ -e "$BASE/evidence.tsv" ]; then
  echo 'ERROR: do not stage label files in smoke directory'
  false
elif [ ! -x /usr/bin/time ]; then
  echo 'ERROR: /usr/bin/time unavailable; cannot collect CPU/memory metrics'
  false
elif ! mkdir "$RUN"; then
  echo 'ERROR: cannot create new smoke run directory'
  false
elif ! python3 - "$REQUESTS" "$RUN/requests8.tsv" <<'PY'
import csv,sys
from pathlib import Path
src,dest=map(Path,sys.argv[1:]); seen=set()
with src.open(newline='') as f, dest.open('x',newline='') as out:
    reader=csv.DictReader(f,delimiter='\t')
    required=['query_id','observed_precursor_mz','observed_charge','top_k','peak_mz','intensity']
    if reader.fieldnames != required: raise ValueError('TRAIN request header mismatch')
    writer=csv.DictWriter(out,fieldnames=required,delimiter='\t',lineterminator='\n');writer.writeheader()
    for row in reader:
        q=row['query_id']
        if q not in seen and len(seen)==8:break
        if q not in seen:seen.add(q)
        if row['top_k']!='256' or row['observed_charge'] not in ('2','3'):
            raise ValueError('Unexpected TRAIN request policy')
        writer.writerow(row)
if len(seen)!=8:raise ValueError('Need eight grouped TRAIN spectra')
print('REDEEM_C2_SMOKE_TRAIN_REQUESTS=PASS spectra=8 truth_accessed=NO')
PY
then
  echo 'ERROR: could not select the eight label-free TRAIN spectra'
  false
elif ! /usr/bin/time -v -o "$RUN/geometry_resources.txt" python3 "$CODE/redeem_identify_optional_c2_20261025.py" \
  --ranker geometry --sif "$SIF" --bind-root "$ROOT" --catalog "$CAT" \
  --spectra "$RUN/requests8.tsv" --output-hits "$RUN/geometry_hits.tsv" \
  --output-summary "$RUN/geometry_summary.tsv"; then
  echo 'ERROR: geometry wrapper failed; preserve run'
  false
elif ! /usr/bin/time -v -o "$RUN/c2_resources.txt" python3 "$CODE/redeem_identify_optional_c2_20261025.py" \
  --ranker c2-experimental --sif "$SIF" --bind-root "$ROOT" --catalog "$CAT" \
  --spectra "$RUN/requests8.tsv" --queries 8 --model "$MODEL" \
  --output-hits "$RUN/c2_hits.tsv" --output-summary "$RUN/c2_summary.tsv" \
  --output-native-hits "$RUN/c2_native_hits.tsv" --output-scores "$RUN/c2_scores.tsv"; then
  echo 'ERROR: C2 opt-in wrapper failed; preserve run'
  false
elif ! cmp -s "$RUN/geometry_hits.tsv" "$RUN/c2_native_hits.tsv"; then
  echo 'ERROR: original candidate list changed between scoring modes'
  false
else
  python3 - "$RUN" <<'PY'
import sys,csv,hashlib,json
from pathlib import Path
root=Path(sys.argv[1])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
legacy=read(root/'geometry_hits.tsv'); reranked=read(root/'c2_hits.tsv')
assert len(legacy)==len(reranked)==8*256
assert {(r['query_id'],r['key'],r['candidate_index']) for r in legacy}=={(r['query_id'],r['key'],r['candidate_index']) for r in reranked}
changed=sum(a['key'] != b['key'] for a,b in zip([r for r in legacy if r['rank']=='1'],[r for r in reranked if r['rank']=='1']))
report={'status':'PASS_C2_OPTIONAL_INFERENCE_SMOKE_NOT_ACCURACY','queries':8,
        'candidate_rows':len(legacy),'changed_top1_without_labels':changed,
        'geometry_hits_sha256':sha(root/'geometry_hits.tsv'),
        'c2_hits_sha256':sha(root/'c2_hits.tsv'),
        'native_parity':True,'truth_labels_read':False,'test_accessed':False,
        'note':'TRAIN spectra for inference engineering smoke only; no scientific gain claimed'}
(root/'smoke_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('REDEEM_C2_OPTIONAL_SMOKE=PASS\n'+json.dumps(report,indent=2))
PY
fi
