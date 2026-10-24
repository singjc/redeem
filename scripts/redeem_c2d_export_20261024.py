#!/usr/bin/env python3
"""Export C2 chemistry features from frozen v0.77 DEV3 hits; never open truth."""
import argparse,csv,hashlib,json,os,tempfile
from pathlib import Path
import redeem_c2_train_features_20261023 as chemistry

REQUEST_SHA='7c9c4e2df5cd7b496fc44b8aca5fa1bd6cfdf7b7245e894569e265240af1c194'
CATALOG_SHA='a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f'
MODEL_SHA='47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca'
REV='ffde6e4a2925f94dc28c87e33917db19a28f5869'
C2_SOURCE_SHA='288568b3132d362a225ce23203565232074adb1197f0c0c7830344f845508acb'


def verify_meta(path):
 m=json.loads(Path(path).read_text())
 for k,v in {'status':'PASS_C2_D_PROSPECTIVE_DEV3_PREPARATION','source_partition':'validation',
             'source_revision':REV,'queries':512,'requests_sha256':REQUEST_SHA,
             'frozen_model_sha256':MODEL_SHA,'protected_test_accessed':False,
             'catalog_or_ranker_metrics_used_for_selection':False,
             'both_previous_dev_sequence_exclusion':True,'train_sequence_exclusion':True}.items():
  if m.get(k)!=v:raise ValueError(f'Unexpected immutable DEV3 metadata {k}')
 return m


def export(requests,hits,metadata,output):
 req=Path(requests);hit=Path(hits);out=Path(output)
 if out.exists() or not out.parent.is_dir():raise ValueError('Output exists or parent missing')
 verify_meta(metadata)
 if chemistry.sha(Path(chemistry.__file__))!=C2_SOURCE_SHA:raise ValueError('C2 feature implementation was changed')
 if chemistry.sha(req)!=REQUEST_SHA:raise ValueError('DEV3 requests changed')
 if not hit.is_file():raise ValueError('DEV3 hits missing')
 obs=chemistry.load_requests(req,512)
 last={};seen=set();rows=[]
 for row in chemistry.read_rows(hit,chemistry.HIT_HEADER):
  q=row['query_id'];rank=int(row['rank']);z=int(row['charge'])
  if q not in obs or z!=obs[q]['charge'] or rank!=last.get(q,0)+1 or rank>256:
   raise ValueError('Unexpected hit query/charge/rank')
  last[q]=rank
  if (q,row['key']) in seen:raise ValueError('Duplicate peptide in candidate pool')
  seen.add((q,row['key']));rows.append(row)
 if set(last)!=set(obs) or len(rows)!=512*256 or any(r!=256 for r in last.values()):
  raise ValueError('Truncated or incomplete 256-candidate pools')
 tmp=Path(tempfile.mkdtemp(prefix='.c2d-features-',dir=out.parent))
 try:
  tsv=tmp/'candidate_chemistry_features.tsv'
  with tsv.open('w',newline='') as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(chemistry.FEATURE_HEADER)
   for row in rows:
    q=row['query_id'];z=int(row['charge']);v=chemistry.features(obs[q],row)
    r={'query_id':q,'original_rank':int(row['rank']),
       'candidate_index':int(row['candidate_index']),'key':row['key'],'charge':z,**v}
    w.writerow([chemistry.fmt(r[k]) for k in chemistry.FEATURE_HEADER])
  report={'status':'PASS_LABEL_BLIND_C2D_DEV3_FEATURE_EXPORT','queries':512,'candidate_rows':len(rows),
          'candidate_rows_per_query':256,'requests_sha256':REQUEST_SHA,'hits_sha256':chemistry.sha(hit),
          'features_sha256':chemistry.sha(tsv),'model_sha256':MODEL_SHA,
          'catalog_sha256':CATALOG_SHA,'input_partition':'validation',
          'source_scoring_job_id':str(os.environ.get('SLURM_JOB_ID','manual')),
          'reference_truth_accessed':False,'protected_test_accessed':False,
          'model_fit_performed':False,'candidate_rank_changed':False,
          'implementation_c2_sha256':C2_SOURCE_SHA,
          'limitations':['C1 can reuse the same observed peak for multiple theoretical ions',
                         'Spectral labels are PSM-derived, not orthogonal synthesis confirmations',
                         'Original ReDeeM foundation TRAIN overlap has not been verified']}
  (tmp/'feature_provenance.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
  tmp.rename(out)
 except Exception:
  (tmp/'FAILED_DO_NOT_USE').write_text('Feature export failed; retain for diagnostics\n')
  raise
 print('REDEEM_C2D_FEATURES=PASS queries=512 rows='+str(len(rows)))
 print('REDEEM_C2D_FEATURE_SHA256='+report['features_sha256'])
 return report


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for flag in ('requests','hits','metadata','output'):p.add_argument('--'+flag,required=True)
 a=p.parse_args();export(a.requests,a.hits,a.metadata,a.output)
if __name__=='__main__':main()
