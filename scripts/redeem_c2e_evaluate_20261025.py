#!/usr/bin/env python3
"""One-time locked paired TEST evaluation. Never fit or choose a new cohort."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import redeem_c2e_lock_20261025 as lock
import redeem_c2d_evaluate_20261024 as frozen_dev_metrics

HITS=['query_id','rank','candidate_index','key','charge','theoretical_neutral_mass','absolute_neutral_mass_error','geometry_score']
RANKED=['query_id','new_rank','candidate_key','original_rank','original_geometry','frozen_c2_score']
TRUTH=['query_id','truth_candidate_key','partition','evidence_source']
METRICS_SOURCE_SHA='b9cd126a1e99745401b6e5530d813270012287cf9cf9f70601d55f45845b3d1e'

def keys_from_catalog(path):
 if lock.sha(path)!=lock.CAT_SHA:raise ValueError('Source catalog SHA mismatch')
 keys=set()
 with Path(path).open() as f:
  if f.readline().strip()!='schema: redeem.foundation.practical_identifier.v0752' or f.readline().strip()!='candidates:':
   raise ValueError('Unexpected source catalog header')
  for line in f:
   if line.startswith('  - key: '):
    k=line[9:].strip()
    if k in keys:raise ValueError('Catalog contains duplicate key')
    keys.add(k)
 if len(keys)!=486810:raise ValueError('Catalog candidate count differs')
 return keys

def evaluate(args):
 out=Path(args.output)
 if out.exists() or not out.parent.is_dir():raise ValueError('One-time TEST report already exists / bad parent')
 lock.protocol(args.protocol)
 meta=lock.test_metadata(args.metadata,args.protocol,True)
 if lock.sha(args.truth)!=meta['truth_sha256'] or lock.sha(args.requests)!=meta['requests_sha256']:
  raise ValueError('TEST cohort content changed')
 if lock.sha(args.model)!=lock.MODEL_SHA:raise ValueError('Fitted C2 model changed')
 if lock.sha(Path(frozen_dev_metrics.__file__))!=METRICS_SOURCE_SHA:raise ValueError('Original paired evaluator changed')
 prov=json.loads(Path(args.provenance).read_text())
 expected={'status':'PASS_LABEL_BLIND_C2E_TEST_FEATURE_EXPORT','protocol_sha256':lock.PROTOCOL_SHA,
  'queries':512,'candidate_rows':131072,'requests_sha256':meta['requests_sha256'],
  'catalog_sha256':lock.CAT_SHA,'model_sha256':lock.MODEL_SHA,
  'reference_truth_accessed':False,'model_fit_performed':False,'source_partition':'test'}
 if any(prov.get(k)!=v for k,v in expected.items()):raise ValueError('Label-blind export provenance differs')
 if lock.sha(args.hits)!=prov['hits_sha256'] or lock.sha(args.features)!=prov['features_sha256']:
  raise ValueError('Feature or hits SHA mismatch')
 truth={}
 for r in frozen_dev_metrics.rows(args.truth,TRUTH):
  if r['query_id'] in truth or r['partition']!='TEST':raise ValueError('Invalid TEST truth rows')
  truth[r['query_id']]=r['truth_candidate_key']
 if len(truth)!=512:raise ValueError('TEST truth count differs')
 catalog=keys_from_catalog(args.catalog);in_catalog=sum(k in catalog for k in truth.values())
 geometry=defaultdict(list);orig={}
 for r in frozen_dev_metrics.rows(args.hits,HITS):
  q=r['query_id'];i=int(r['rank']);key=r['key']
  if i!=len(geometry[q])+1 or i>256 or (q,key) in orig:raise ValueError('Original hit pool invalid')
  geometry[q].append(key);orig[(q,key)]=i
 c2=defaultdict(list)
 for r in frozen_dev_metrics.rows(args.ranked,RANKED):
  q=r['query_id'];i=int(r['new_rank']);key=r['candidate_key']
  if i!=len(c2[q])+1 or i>256 or orig.get((q,key))!=int(r['original_rank']):
   raise ValueError('C2 candidate pool differs')
  c2[q].append(key)
 m=frozen_dev_metrics.metrics(truth,geometry,c2,query_count=512,pool=256)
 n=m['truth_in_candidate_pool']
 if n>in_catalog:raise ValueError('Pool contains reference absent from catalog')
 delta=m['paired_delta_conditional'];deltas=[]
 for q in sorted(truth):
  if truth[q] in geometry[q]:deltas.append(int(c2[q][0]==truth[q])-int(geometry[q][0]==truth[q]))
 ci=frozen_dev_metrics.paired_ci(deltas,seed=20261025,n=10000)
 decision='INCONCLUSIVE_TOO_FEW_POOL_COVERED' if n<128 else ('CONFIRMED_FIXED_C2_TEST_GATE' if delta>=.03 and ci[0]>0 else 'FAILED_FIXED_C2_TEST_GATE_NO_RETUNING')
 m.update({'paired_bootstrap_95_ci_conditional':ci,'acceptance':decision,
  'acceptance_rule':{'minimum_eligible':128,'minimum_absolute_conditional_gain':0.03,'bootstrap_repeats':10000,'bootstrap_seed':20261025,'lower_95_ci_above_zero':True}})
 m.update({'status':'PASS_HELDOUT_ONE_TIME_EVALUATION','protocol':'redeem_c2e_protected_pt_test_fixed_once_v1',
  'source_partition':'test','queries':512,'truth_in_catalog':in_catalog,
  'catalog_coverage_fraction':in_catalog/512,'pool_recall_given_catalog':n/in_catalog if in_catalog else None,
  'protocol_sha256':lock.PROTOCOL_SHA,'model_sha256':lock.MODEL_SHA,'catalog_sha256':lock.CAT_SHA,
  'requests_sha256':meta['requests_sha256'],'truth_sha256':meta['truth_sha256'],
  'hits_sha256':prov['hits_sha256'],'features_sha256':prov['features_sha256'],
  'ranked_sha256':lock.sha(args.ranked),'test_partition_accessed_for_final_evaluation':True,
  'trained_or_tuned_on_test':False,'no_test_cohort_reselection':True,
  'limits':['PSM annotations are not orthogonal synthesis confirmation','Published peptide-disjoint TEST is same overall ProteomeTools source family','Historical ReDeeM foundation TRAIN overlap not independently checked','C2 is a candidate reranker, not inverse foundation decoder']})
 out.write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
 print(f'REDEEM_C2E_TEST=PASS decision={decision} conditional_top1={m["c2_top1_count"]}/{n} vs {m["legacy_top1_count"]}/{n}')
 return m

def main():
 p=argparse.ArgumentParser()
 for x in ('protocol','model','metadata','truth','requests','catalog','hits','features','provenance','ranked','output'):
  p.add_argument('--'+x,required=True)
 evaluate(p.parse_args())
if __name__=='__main__':main()
