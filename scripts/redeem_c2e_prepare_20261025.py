#!/usr/bin/env python3
"""One-time held-out TEST cohort selection from the preregistered source.

Only use AFTER freezing LOCKED_PROTOCOL. Preserve TEST labels on the laptop.
No score-based selection or model fitting. TEST can be opened only once.
"""
import argparse, csv, hashlib, heapq, json, math, tempfile
import redeem_c2e_lock_20261025 as lock
from collections import Counter
from pathlib import Path

REV='ffde6e4a2925f94dc28c87e33917db19a28f5869'
MODEL_SHA='47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca'
EMBARGO=tuple((k,v[0],v[2],v[1]) for k,v in lock.EMBARGO.items())
POOLS={'TUM_first_pool','TUM_second_pool','SRMAtlas','TUM_isoform','TUM_second_addon','TUM_proteo_TMT'}
AA=set('ADEFGHIKLMNPQRSTVWY')
REQ=['query_id','observed_precursor_mz','observed_charge','top_k','peak_mz','intensity']
TRUTH=['query_id','truth_candidate_key','partition','evidence_source']
EXPECTED_MODEL={'schema':'redeem.c2.chemistry_pairwise_ranker.v1','status':'TRAIN_ONLY_FROZEN_COEFFICIENTS_NOT_DEV_EVALUATED',
 'train_query_count':2048,'train_truth_in_pool':1041,'train_truth_absent_from_pool':1007,
 'training_top1_chemistry_count':931,'train_label_source_sha256':lock.EMBARGO['train'][0],
 'train_requests_sha256':'26dc26a242c5cb039bdba91d7cbfa0ea6b0e14f27777ba1f54f0a8e7bf2c5a25',
 'train_features_sha256':'8e2ada36e42bf7874f2f65078d8b1c76007d3f5b4f7bb40918030b85c6438c3d'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for ch in iter(lambda:f.read(1024*1024),b''):h.update(ch)
 return h.hexdigest()

def freeze(model_dir):
 p=Path(model_dir)/'model.json'; check=Path(model_dir)/'model.sha256'
 if not p.is_file() or not check.is_file() or sha(p)!=MODEL_SHA:raise ValueError('Frozen model file SHA mismatch/missing')
 if check.read_text().strip()!=f'{MODEL_SHA}  model.json':raise ValueError('Frozen model manifest mismatch')
 m=json.loads(p.read_text())
 for k,v in EXPECTED_MODEL.items():
  if m.get(k)!=v:raise ValueError(f'Frozen model contract mismatch: {k}')
 if m.get('test_accessed') is not False or m.get('validation_labels_accessed') is not False:raise ValueError('Model has seen validation/TEST')
 if len(m.get('coefficients',[]))!=6 or len(m.get('standardization_mean',[]))!=6 or len(m.get('standardization_std_min_005',[]))!=6:raise ValueError('Invalid feature dimension')
 if not all(math.isfinite(x) for k in ('coefficients','standardization_mean','standardization_std_min_005') for x in m[k]):raise ValueError('Non-finite model value')
 return m

def embargo(paths):
 excluded=set(); ids=set()
 for (kind,expected,partition,n),(name,path) in zip(EMBARGO,paths.items()):
  if kind!=name:raise ValueError('Wrong embargo ledger order')
  path=Path(path)
  if sha(path)!=expected:raise ValueError(f'Immutable {kind} truth SHA mismatch')
  with path.open(newline='') as f:
   r=csv.DictReader(f,delimiter='\t')
   if r.fieldnames!=TRUTH:raise ValueError(f'Unexpected {kind} truth schema')
   rows=list(r)
  if len(rows)!=n or any(x['partition']!=partition for x in rows) or len({x['query_id'] for x in rows})!=n:raise ValueError('Embargo ledger malformed')
  for row in rows:
   key=row['truth_candidate_key']
   if '|z' not in key:raise ValueError('Bad peptide key')
   excluded.add(key.rsplit('|z',1)[0])
   ids.add(row['query_id'])
 return excluded,ids

def eligible(row,excluded,ids):
 try:
  sequence=str(row['sequence']).strip().upper();mod=str(row['modified_sequence']).strip().upper()
  charge=int(row['precursor_charge']);mz=float(row['precursor_mz'])
  peaks=list(zip(row['mz_array'],row['intensity_array']))
  ident=[str(row[k]) for k in ('experiment_name','evidence_index','scan_number')]
 except (TypeError,KeyError,ValueError):return None
 if (sequence!=mod or not 7<=len(sequence)<=64 or not set(sequence)<=AA or sequence in excluded
     or charge not in (2,3) or not 100<mz<5000 or not math.isfinite(mz)
     or str(row['experiment_name']) not in POOLS or not all(ident)):
  return None
 if len(row['mz_array'])!=len(row['intensity_array']):return None
 kept=[]
 for a,b in peaks:
  try:a=float(a);b=float(b)
  except (TypeError,ValueError):continue
  if math.isfinite(a) and math.isfinite(b) and a>0 and b>0:kept.append((a,b))
 if len(kept)<10:return None
 kept=sorted(sorted(kept,key=lambda p:(-p[1],p[0]))[:256])
 uid='pttest-'+hashlib.sha256('\x1f'.join(ident).encode()).hexdigest()[:20]
 if uid in ids:return None
 return {'seq':sequence,'z':charge,'precursor_mz':mz,'peaks':kept,'query_id':uid,
         'experiment':ident[0],'evidence_index':ident[1],'scan_number':ident[2]}

def select(stream,excluded,ids,limit=512,seed=20261025):
 if limit!=512 or seed!=20261025:raise ValueError('Only the preregistered TEST512/20261025 selection is permitted')
 selected={};heap=[];counts=Counter()
 for row in stream:
  counts['scanned']+=1
  r=eligible(row,excluded,ids)
  if r is None:counts['filtered']+=1;continue
  priority=int(hashlib.sha256(f'{seed}|{r["query_id"]}'.encode()).hexdigest(),16)
  seq=r['seq'];old=selected.get(seq)
  if old and (priority,r['query_id'])>=old[:2]:continue
  if old is None and len(selected)==limit:
   while heap:
    neg,uid,old_seq=heap[0];found=selected.get(old_seq)
    if found and found[0]==-neg and found[1]==uid:break
    heapq.heappop(heap)
   if not heap:raise ValueError('Selection heap damaged')
   if (priority,r['query_id'])>=(-heap[0][0],heap[0][1]):continue
   victim=heapq.heappop(heap)[2];selected.pop(victim)
  selected[seq]=(priority,r['query_id'],r)
  heapq.heappush(heap,(-priority,r['query_id'],seq))
  counts['eligible']+=1
 if len(selected)!=512:raise ValueError(f'Not enough disjoint TEST identities: {len(selected)}')
 results=[v[2] for v in sorted(selected.values(),key=lambda v:(v[0],v[1]))]
 if len({v['query_id'] for v in results})!=512:raise ValueError('Duplicate spectrum ID')
 return results,counts

def parquet_rows(path):
 import pyarrow.parquet as pq
 required=['sequence','modified_sequence','experiment_name','evidence_index','scan_number',
           'precursor_charge','precursor_mz','mz_array','intensity_array']
 pf=pq.ParquetFile(str(path))
 if not set(required).issubset(pf.schema_arrow.names):raise ValueError('Unexpected pinned TEST Parquet columns')
 for b in pf.iter_batches(batch_size=4096,columns=required):
  yield from b.to_pylist()

def prepare(args,stream_override=None):
 lock.protocol(args.protocol)
 out=Path(args.output)
 if out.exists() or not out.parent.is_dir():raise ValueError('Output exists or parent missing')
 model=freeze(args.model_dir)
 forbidden,ids=embargo({'dev1':args.dev1,'dev2':args.dev2,'dev3':args.dev3,'train':args.train})
 if stream_override is None:
  path=Path(args.parquet)
  if not path.is_file():raise ValueError('Pinned TEST parquet not available locally')
  # Enforce the immutable Hugging Face snapshot path instead of trusting a renamed file.
  expected_name=lock.TEST_NAME
  if path.name!=expected_name or f'/snapshots/{REV}/' not in path.as_posix():
   raise ValueError('Require exact pinned Hugging Face snapshot path and TEST shard filename')
  source_hash=sha(path)
  rows,counts=select(parquet_rows(path),forbidden,ids)
 else:
  source_hash='SYNTHETIC_FIXTURE_NOT_REAL';rows,counts=select(stream_override,forbidden,ids)
 tmp=Path(tempfile.mkdtemp(prefix='.redeem-c2e-',dir=out.parent))
 try:
  with (tmp/'requests.tsv').open('w',newline='') as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(REQ)
   for r in rows:
    for mass,intensity in r['peaks']:
     w.writerow([r['query_id'],format(r['precursor_mz'],'.12g'),r['z'],256,format(mass,'.12g'),format(intensity,'.12g')])
  with (tmp/'truth.tsv').open('w',newline='') as f,(tmp/'evidence.tsv').open('w',newline='') as e:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(TRUTH)
   x=csv.writer(e,delimiter='\t',lineterminator='\n');x.writerow(['query_id','experiment_name','evidence_index','scan_number','sequence','charge','peak_count'])
   for r in rows:
    w.writerow([r['query_id'],f'{r["seq"]}|z{r["z"]}','TEST',f'InstaDeepAI/ms_proteometools/test@{REV}'])
    x.writerow([r['query_id'],r['experiment'],r['evidence_index'],r['scan_number'],r['seq'],r['z'],len(r['peaks'])])
  meta={'status':'PASS_C2E_HELDOUT_TEST_PREPARATION','source_partition':'test','source_revision':REV,
        'source_file_sha256':source_hash,'source_file_scope':'pinned_public_heldout_test_shard_provided_locally',
        'queries':512,'seed':20261025,'disjoint_from_dev1_dev2_dev3_and_c2_train':True,
        'label_blind_catalog_and_feature_scoring':True,'selection_uses_scores_or_catalog':False,
        'test_partition_opened_for_predeclared_selection':True,'protocol_sha256':lock.PROTOCOL_SHA,'frozen_model_sha256':MODEL_SHA,
        'training_top1_geometry_count':model['training_top1_geometry_count'],
        'training_top1_chemistry_count':model['training_top1_chemistry_count'],
        'requests_sha256':sha(tmp/'requests.tsv'),'truth_sha256':sha(tmp/'truth.tsv'),
        'evidence_sha256':sha(tmp/'evidence.tsv'),'selection_counts':dict(counts),
        'cautions':['Heldout published peptide-disjoint TEST; PSM annotations and source family overlap remain limitations',
                    'One-time selection and scoring; no model or catalog changes and no reselection',
                    'PSM labels are not orthogonal experimental peptide confirmations',
                    'Historical ReDeeM foundation TRAIN overlap remains unknown']}
  (tmp/'metadata.json').write_text(json.dumps(meta,sort_keys=True,indent=2)+'\n')
  tmp.rename(out)
 except Exception:
  (tmp/'FAILED_DO_NOT_USE').write_text('Partial output; preserve for inspection.\n')
  raise
 print('REDEEM_C2E_PREPARE=PASS queries=512 disjoint_prior=YES')
 print('REDEEM_C2E_MODEL_SHA256='+MODEL_SHA)
 print('REDEEM_C2E_OUTPUT='+str(out))
 return meta

def cli():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model-dir',required=True);p.add_argument('--dev1',required=True);p.add_argument('--dev2',required=True)
 p.add_argument('--train',required=True);p.add_argument('--dev3',required=True);p.add_argument('--protocol',required=True);p.add_argument('--parquet',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();prepare(a)
if __name__=='__main__':cli()
