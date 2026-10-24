#!/usr/bin/env python3
"""One locked DEV3 paired evaluation. No fitting/tuning/model selection."""
import argparse,csv,hashlib,json,math,random
from collections import defaultdict
from pathlib import Path

Q=512;POOL=256;COV=297
MODEL_SHA='47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca'
CAT_SHA='a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f'
REQUEST_SHA='7c9c4e2df5cd7b496fc44b8aca5fa1bd6cfdf7b7245e894569e265240af1c194'
TRUTH_SHA='34d8c9ac18abd56e8c0e96cc6f952816d9ac2cf7c7e4934643881e85eecde07f'
HIT_HEADER=['query_id','rank','candidate_index','key','charge','theoretical_neutral_mass','absolute_neutral_mass_error','geometry_score']
RANK_HEADER=['query_id','new_rank','candidate_key','original_rank','original_geometry','frozen_c2_score']
TRUTH_HEADER=['query_id','truth_candidate_key','partition','evidence_source']


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()


def rows(path,header):
 with Path(path).open(newline='') as f:
  r=csv.DictReader(f,delimiter='\t')
  if r.fieldnames!=header:raise ValueError('TSV schema differs: '+str(path))
  for x in r:
   if None in x or any(v is None for v in x.values()):raise ValueError('Malformed TSV')
   yield x


def paired_ci(deltas,seed=20261024,n=10000):
 if not deltas:raise ValueError('No paired observations')
 rng=random.Random(seed);length=len(deltas);out=[]
 for i in range(n):
  s=sum(deltas[rng.randrange(length)] for _ in range(length));out.append(s/length)
 out.sort()
 return [out[int(.025*n)],out[int(.975*n)]]


def metrics(truth,geometry,c2,query_count=Q,pool=POOL):
 if len(truth)!=query_count or set(truth)!=set(geometry) or set(truth)!=set(c2):
  raise ValueError('Missing or extra queries')
 covered=[];legacy_top1=c2_top1=legacy_top10=c2_top10=0
 sum_geom_rr=sum_c2_rr=0.0;corrected=[];regressed=[];differences=[]
 for q in sorted(truth):
  g=geometry[q];t=c2[q]
  if len(g)!=pool or len(t)!=pool or len(set(g))!=pool or len(set(t))!=pool or set(g)!=set(t):
   raise ValueError('Incomplete/different candidate pools')
  label=truth[q]
  if label in g:
   covered.append(q);r1=g.index(label)+1;r2=t.index(label)+1
   ga=int(r1==1);ca=int(r2==1)
   legacy_top1+=ga;c2_top1+=ca
   legacy_top10+=int(r1<=10);c2_top10+=int(r2<=10)
   sum_geom_rr+=1/r1;sum_c2_rr+=1/r2
   differences.append(ca-ga)
   if ca and not ga:corrected.append(q)
   if ga and not ca:regressed.append(q)
 count=len(covered)
 if count==0:raise ValueError('No reference identities in candidate pool')
 diff=(c2_top1-legacy_top1)/count
 ci=paired_ci(differences)
 return {'truth_in_candidate_pool':count,'pool_recall_unconditional':count/query_count,
         'legacy_top1_count':legacy_top1,'c2_top1_count':c2_top1,
         'legacy_top1_unconditional':legacy_top1/query_count,'c2_top1_unconditional':c2_top1/query_count,
         'legacy_top1_conditional':legacy_top1/count,'c2_top1_conditional':c2_top1/count,
         'legacy_top10_unconditional':legacy_top10/query_count,'c2_top10_unconditional':c2_top10/query_count,
         'legacy_mrr_unconditional':sum_geom_rr/query_count,'c2_mrr_unconditional':sum_c2_rr/query_count,
         'paired_delta_conditional':diff,'paired_bootstrap_95_ci_conditional':ci,
         'paired_corrected':len(corrected),'paired_regressed':len(regressed),
         'corrected_query_ids':corrected,'regressed_query_ids':regressed,
         'acceptance_rule':{'minimum_absolute_gain':.03,'lower_95_ci_above_zero':True},
         'acceptance':'PASS_FIXED_C2_DEV3_GATE' if diff>=.03 and ci[0]>0 else 'FAIL_FIXED_C2_DEV3_GATE_STOP_NO_RETUNING'}


def evaluate(args):
 out=Path(args.output)
 if out.exists() or not out.parent.is_dir():raise ValueError('Output exists or parent missing')
 meta=json.loads(Path(args.metadata).read_text())
 if any(meta.get(k)!=v for k,v in {'status':'PASS_C2_D_PROSPECTIVE_DEV3_PREPARATION',
 'queries':512,'source_partition':'validation','seed':20261024,
 'frozen_model_sha256':MODEL_SHA,'requests_sha256':REQUEST_SHA,
 'truth_sha256':TRUTH_SHA,'protected_test_accessed':False,
 'catalog_or_ranker_metrics_used_for_selection':False}.items()):raise ValueError('DEV3 cohort metadata mismatch')
 if sha(args.truth)!=TRUTH_SHA:raise ValueError('Frozen DEV3 truth mismatch')
 model=json.loads(Path(args.model).read_text())
 if sha(args.model)!=MODEL_SHA or model.get('train_query_count')!=2048 or model.get('train_truth_in_pool')!=1041:
  raise ValueError('Frozen C2 model mismatch')
 cov=json.loads(Path(args.coverage).read_text())
 if any(cov.get(k)!=v for k,v in {'status':'PASS_C2_D_LOCKED_COVERAGE_AUDIT',
 'queries':512,'truth_in_catalog':297,'catalog_sha256':CAT_SHA,'requests_sha256':REQUEST_SHA,
 'truth_sha256':TRUTH_SHA,'frozen_model_sha256':MODEL_SHA,'decision':'ALLOW_ONCE_SCORING'}.items()):
  raise ValueError('Original catalog coverage gate record differs')
 prov=json.loads(Path(args.feature_provenance).read_text())
 if any(prov.get(k)!=v for k,v in {'status':'PASS_LABEL_BLIND_C2D_DEV3_FEATURE_EXPORT',
 'queries':512,'candidate_rows':131072,'requests_sha256':REQUEST_SHA,'catalog_sha256':CAT_SHA,
 'model_sha256':MODEL_SHA,'reference_truth_accessed':False,'protected_test_accessed':False,
 'model_fit_performed':False}.items()):raise ValueError('Feature export provenance mismatch')
 if sha(args.hits)!=prov['hits_sha256'] or sha(args.features)!=prov['features_sha256']:
  raise ValueError('Frozen hits/features SHA mismatch')
 truth={}
 for r in rows(args.truth,TRUTH_HEADER):
  q=r['query_id']
  if q in truth or r['partition']!='DEV':raise ValueError('Bad prospective DEV truth')
  truth[q]=r['truth_candidate_key']
 geometry=defaultdict(list);original={};candidate_idx={}
 for r in rows(args.hits,HIT_HEADER):
  q=r['query_id'];idx=int(r['rank']);key=r['key']
  if idx!=len(geometry[q])+1 or idx>POOL or (q,key) in original:
   raise ValueError('Unsorted/duplicated v0.77 ranks')
  geometry[q].append(key);original[(q,key)]=idx
  candidate_idx[(q,key)]=int(r['candidate_index'])
 c2=defaultdict(list)
 for r in rows(args.ranked,RANK_HEADER):
  q=r['query_id'];i=int(r['new_rank']);key=r['candidate_key'];orig=int(r['original_rank'])
  if i!=len(c2[q])+1 or i>POOL or original.get((q,key))!=orig:
   raise ValueError('C2 rank has missing/changed candidate or original rank')
  c2[q].append(key)
 result=metrics(truth,geometry,c2)
 if result['truth_in_candidate_pool']>cov['truth_in_catalog']:
  raise ValueError('Pool recall exceeds catalog coverage')
 result.update({'status':'PASS_C2_D_PROSPECTIVE_EVALUATION',
 'protocol':'C2_D_fixed_chemistry_vs_original_geometry_paired_DEV3_20261024',
 'queries':Q,'catalog_coverage_count':COV,'catalog_coverage_fraction':COV/Q,
 'pool_recall_given_catalog':result['truth_in_candidate_pool']/COV,
 'model_sha256':MODEL_SHA,'catalog_sha256':CAT_SHA,'requests_sha256':REQUEST_SHA,
 'truth_sha256':TRUTH_SHA,'hits_sha256':prov['hits_sha256'],
 'features_sha256':prov['features_sha256'],'ranked_sha256':sha(args.ranked),
 'trained_or_tuned_on_dev3':False,'test_accessed':False,
 'scientific_limits':['Published PSM-derived DEV reference, not independent synthetic identity confirmation',
 'Same public validation partition as previous DEV cohorts, albeit disjoint sequences',
 'Historical ReDeeM foundation TRAIN overlap remains unverified',
 'Only one frozen C2 model and predeclared acceptance criterion; no DEV3 retuning permitted']})
 out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 print('REDEEM_C2_D_EVALUATION=PASS decision='+result['acceptance'])
 print('REDEEM_C2_D_CONDITIONAL_TOP1=%s/%s vs %s/%s' %(result['c2_top1_count'],result['truth_in_candidate_pool'],result['legacy_top1_count'],result['truth_in_candidate_pool']))
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ('metadata','truth','model','coverage','hits','features','feature_provenance','ranked','output'):
  p.add_argument('--'+x.replace('_','-'),dest=x,required=True)
 evaluate(p.parse_args())
if __name__=='__main__':main()
