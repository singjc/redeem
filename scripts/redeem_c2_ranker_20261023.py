#!/usr/bin/env python3
"""TRAIN-only deterministic pairwise chemistry ranker. No DEV/TEST label access.

Geometry score is the unchanged control score. The trained model adds a small
regularized correction using TRAIN-only preprocessing statistics. Experimental only.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,tempfile
from collections import defaultdict
from pathlib import Path

FEATURES=['precursor_mass_error_ppm','best_isotope_residual_da',
          'b_cleavage_fraction','y_cleavage_fraction',
          'explained_sqrt_intensity_fraction','median_backbone_mass_error_ppm']
TRANSFORMS=['log1p','log1p100','identity','identity','identity','log1p']
POOL=256
CONFIG={'epochs':100,'learning_rate':0.04,'pairwise_l2':0.15,'negatives_per_query':24,
        'max_abs_correction_weight':1.0,'seed':20261023}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()

def read_tsv(path):
 with Path(path).open(newline='') as f:
  reader=csv.DictReader(f,delimiter='\t')
  if not reader.fieldnames:raise ValueError('Missing TSV header')
  for row in reader:
   if None in row or any(x is None for x in row.values()):raise ValueError('Malformed TSV row')
   yield row

def truth_map(path,queries,expected_sha=None):
 if expected_sha and sha(path)!=expected_sha:raise ValueError('TRAIN truth SHA mismatch')
 result={};sequences=set()
 for row in read_tsv(path):
  if set(row)!=set(['query_id','truth_candidate_key','partition','evidence_source']):raise ValueError('Unexpected TRAIN truth columns')
  if row['partition']!='TRAIN' or '/train@' not in row['evidence_source']:raise ValueError('Only TRAIN labels allowed')
  key=row['truth_candidate_key'];seq=key.rsplit('|z',1)[0]
  if row['query_id'] in result or seq in sequences:raise ValueError('Duplicate TRAIN query/sequence')
  result[row['query_id']]=key;sequences.add(seq)
 if len(result)!=queries:raise ValueError('Unexpected TRAIN truth query count')
 return result

def numeric(x,name):
 try:v=float(x)
 except (TypeError,ValueError):raise ValueError(f'Bad feature {name}')
 if not math.isfinite(v):raise ValueError(f'Non-finite feature {name}')
 return v

def feature_matrix(path,expected_queries,expected_sha=None,pool=POOL):
 import numpy as np
 if expected_sha and sha(path)!=expected_sha:raise ValueError('Feature SHA mismatch')
 cols=[];raw=[];byquery=defaultdict(list);seen=set();last={}
 for row in read_tsv(path):
  if any(f not in row for f in FEATURES) or not all(k in row for k in ('query_id','key','original_rank','geometry_score')):raise ValueError('Missing chemistry feature fields')
  q=row['query_id'];rank=int(row['original_rank'])
  if rank!=last.get(q,0)+1 or rank>pool:raise ValueError('Incomplete/out-of-order candidate ranks')
  last[q]=rank
  if (q,row['key']) in seen:raise ValueError('Duplicate candidate in pool')
  seen.add((q,row['key']))
  geom=numeric(row['geometry_score'],'geometry_score')
  values=[]
  for name,trans in zip(FEATURES,TRANSFORMS):
   x=row[name]
   # Missing fragment residual: use capped no-match residual, a TRAIN-fixed sentinel.
   v=1000.0 if x=='' and name=='median_backbone_mass_error_ppm' else numeric(x,name)
   if v<0:raise ValueError('Negative physical evidence feature')
   if trans=='log1p':v=math.log1p(min(v,1e6))
   elif trans=='log1p100':v=math.log1p(min(v*100,1e6))
   values.append(v)
  raw.append(values);cols.append((q,row['key'],rank,geom));byquery[q].append(len(cols)-1)
 if len(byquery)!=expected_queries or len(cols)!=expected_queries*pool or any(r!=pool for r in last.values()):raise ValueError('Feature output must contain complete 256-candidate pools')
 return cols,np.asarray(raw,dtype=np.float64),dict(byquery)

def pairs_for_fit(cols,byquery,truth):
 pos=[];neg=[];covered=0
 for q,idxs in byquery.items():
  found=[i for i in idxs if cols[i][1]==truth[q]]
  if not found:continue
  if len(found)!=1:raise ValueError('Non-unique true candidate')
  covered+=1;p=found[0]
  # Hard negatives = first 24 original geometry-ranked candidates, excluding truth.
  candidates=[i for i in idxs if i!=p][:CONFIG['negatives_per_query']]
  for n in candidates:pos.append(p);neg.append(n)
 if covered<128:raise ValueError(f'Insufficient TRAIN supervision: {covered} pool-covered queries')
 return pos,neg,covered

def train(args):
 import numpy as np
 train_dir=Path(args.train_dir);out=Path(args.output_dir)
 if out.exists():raise ValueError('Model output exists; refuse overwrite')
 meta=json.loads((train_dir/'metadata.json').read_text())
 if meta.get('status')!='PASS_TRAIN_ONLY_SOURCE_PREPARATION' or meta.get('source_partition')!='train' or meta.get('protected_test_accessed') is not False:raise ValueError('Not a permitted TRAIN cohort')
 if meta.get('historical_redeem_train_overlap')!='NOT_CHECKED':raise ValueError('Unexpected historical TRAIN overlap declaration')
 truth=truth_map(train_dir/'truth.tsv',meta['selected_queries'],meta['truth_sha256'])
 prov=json.loads(Path(args.feature_provenance).read_text())
 if prov.get('status')!='PASS_TRAIN_ONLY_CHEMISTRY_FEATURE_EXPORT' or prov.get('reference_truth_accessed') is not False or prov.get('requests_sha256')!=meta['requests_sha256']:raise ValueError('Invalid chemistry export provenance')
 cols,raw,groups=feature_matrix(args.features,meta['selected_queries'],prov['features_sha256'])
 if set(groups)!=set(truth):raise ValueError('TRAIN features/truth query mismatch')
 pos,neg,covered=pairs_for_fit(cols,groups,truth)
 mu=raw.mean(axis=0);st=raw.std(axis=0);st=np.maximum(st,0.05)
 X=np.clip((raw-mu)/st,-6,6)
 p=np.asarray(pos,dtype=np.int64);n=np.asarray(neg,dtype=np.int64)
 d=X[p]-X[n];base=np.asarray([cols[i][3] for i in p])-np.asarray([cols[i][3] for i in n])
 w=np.zeros(len(FEATURES));m=np.zeros_like(w);v=np.zeros_like(w);losses=[]
 for epoch in range(1,CONFIG['epochs']+1):
  margin=np.clip(base+d@w,-50,50);sig=1/(1+np.exp(margin))
  loss=float(np.logaddexp(0,-margin).mean()+CONFIG['pairwise_l2']/2*np.dot(w,w))
  grad=-(d.T@sig)/len(d)+CONFIG['pairwise_l2']*w
  m=.9*m+.1*grad;v=.999*v+.001*grad**2
  mh=m/(1-.9**epoch);vh=v/(1-.999**epoch)
  w-=CONFIG['learning_rate']*mh/(np.sqrt(vh)+1e-8)
  np.clip(w,-CONFIG['max_abs_correction_weight'],CONFIG['max_abs_correction_weight'],out=w)
  if epoch in (1,25,50,100):losses.append({'epoch':epoch,'training_pairwise_loss':loss})
 before=after=0
 for q,idxs in groups.items():
  correct=truth[q]
  orig=max(idxs,key=lambda i:(cols[i][3],-cols[i][2]))
  if cols[orig][1]==correct:before+=1
  best=max(idxs,key=lambda i:(cols[i][3]+float(X[i]@w),-cols[i][2]))
  if cols[best][1]==correct:after+=1
 result={'schema':'redeem.c2.chemistry_pairwise_ranker.v1','status':'TRAIN_ONLY_FROZEN_COEFFICIENTS_NOT_DEV_EVALUATED',
  'feature_names':FEATURES,'transformations':TRANSFORMS,'missing_fragment_error_ppm_sentinel':1000,
  'standardization_mean':mu.tolist(),'standardization_std_min_005':st.tolist(),
  'coefficients':w.tolist(),'base_score':'unchanged_geometry_score','config':CONFIG,
  'train_query_count':len(groups),'train_truth_in_pool':covered,'train_truth_absent_from_pool':len(groups)-covered,
  'train_label_source_sha256':meta['truth_sha256'],'train_requests_sha256':meta['requests_sha256'],
  'train_hits_sha256':prov['hits_sha256'],'train_features_sha256':prov['features_sha256'],
  'training_top1_geometry_count':before,'training_top1_chemistry_count':after,
  'loss_trace':losses,'validation_labels_accessed':False,'test_accessed':False,
  'foundation_historical_train_overlap':'NOT_CHECKED',
  'limitations':['TRAIN-only fit; training top-1 is not external generalization performance',
   'PSM labels are not orthogonal sequence confirmation','C1 ion matches can reuse observed peaks',
   'Unmatched candidates may include chemically indistinguishable alternatives',
   'Training data is not independent of published synthesis catalog.']}
 if not out.parent.is_dir():raise ValueError('Model output parent missing')
 temp=Path(tempfile.mkdtemp(prefix='.c2-model-',dir=out.parent))
 try:
  (temp/'model.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
  (temp/'model.sha256').write_text(sha(temp/'model.json')+'  model.json\n')
  temp.rename(out)
 except BaseException:
  (temp/'FAILED_DO_NOT_USE').write_text('Incomplete training.\n');raise
 print(f'REDEEM_C2_RANKER_TRAIN=PASS train_queries={len(groups)} truth_in_pool={covered} training_top1={after}')
 print('REDEEM_C2_MODEL_SHA256='+sha(out/'model.json'))
 return result

def apply(args):
 import numpy as np
 out=Path(args.output)
 if out.exists():raise ValueError('Output exists; refuse overwrite')
 model_path=Path(args.model);model=json.loads(model_path.read_text())
 if model.get('status')!='TRAIN_ONLY_FROZEN_COEFFICIENTS_NOT_DEV_EVALUATED' or model.get('feature_names')!=FEATURES or model.get('config')!=CONFIG:raise ValueError('Unexpected model artifact')
 cols,raw,groups=feature_matrix(args.features,args.queries,expected_sha=args.features_sha)
 x=np.clip((raw-np.array(model['standardization_mean']))/np.array(model['standardization_std_min_005']),-6,6)
 scores=np.array([x[3] for x in cols])+x@np.array(model['coefficients'])
 with out.open('x',newline='') as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query_id','new_rank','candidate_key','original_rank','original_geometry','frozen_c2_score'])
  for q in sorted(groups):
   ranked=sorted(groups[q],key=lambda i:(-scores[i],cols[i][2]))
   for rank,i in enumerate(ranked,1):w.writerow([q,rank,cols[i][1],cols[i][2],format(cols[i][3],'.10g'),format(scores[i],'.10g')])
 print('REDEEM_C2_RANKER_APPLY=PASS queries='+str(len(groups))+' model_sha256='+sha(model_path))

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
 t=s.add_parser('fit');t.add_argument('--train-dir',required=True);t.add_argument('--features',required=True);t.add_argument('--feature-provenance',required=True);t.add_argument('--output-dir',required=True)
 a=s.add_parser('apply');a.add_argument('--features',required=True);a.add_argument('--features-sha',required=True);a.add_argument('--queries',type=int,required=True);a.add_argument('--model',required=True);a.add_argument('--output',required=True)
 args=p.parse_args()
 if args.action=='fit':train(args)
 else:apply(args)
if __name__=='__main__':main()
