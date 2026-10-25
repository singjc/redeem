#!/usr/bin/env python3
"""Label-blind frozen chemistry features for preregistered TEST cohort."""
import argparse,csv,json,os,tempfile
from pathlib import Path
import redeem_c2e_lock_20261025 as lock
import redeem_c2_train_features_20261023 as chemistry
CHEM_SHA='288568b3132d362a225ce23203565232074adb1197f0c0c7830344f845508acb'

def export(requests,hits,metadata,protocol,out):
 meta=lock.test_metadata(metadata,protocol)
 req,hit,out=Path(requests),Path(hits),Path(out)
 if out.exists() or not out.parent.is_dir():raise ValueError('Refuse existing output')
 if lock.sha(req)!=meta['requests_sha256'] or lock.sha(Path(chemistry.__file__))!=CHEM_SHA:
  raise ValueError('Frozen chemistry code or TEST request hash mismatch')
 obs=chemistry.load_requests(req,512)
 last={};seen=set();rows=[]
 for row in chemistry.read_rows(hit,chemistry.HIT_HEADER):
  q=row['query_id'];rank=int(row['rank']);z=int(row['charge'])
  if q not in obs or z!=obs[q]['charge'] or rank!=last.get(q,0)+1 or rank>256:
   raise ValueError('Out-of-order hit/charge mismatch')
  last[q]=rank
  if (q,row['key']) in seen:raise ValueError('Duplicate candidate')
  seen.add((q,row['key']));rows.append(row)
 if set(obs)!=set(last) or len(rows)!=512*256 or any(v!=256 for v in last.values()):
  raise ValueError('Truncated/incomplete full candidate pool')
 tmp=Path(tempfile.mkdtemp(prefix='.c2e-export-',dir=out.parent))
 try:
  tsv=tmp/'candidate_chemistry_features.tsv'
  with tsv.open('w',newline='') as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(chemistry.FEATURE_HEADER)
   for row in rows:
    q=row['query_id'];r={'query_id':q,'original_rank':int(row['rank']),
     'candidate_index':int(row['candidate_index']),'key':row['key'],
     'charge':int(row['charge']),**chemistry.features(obs[q],row)}
    w.writerow([chemistry.fmt(r[k]) for k in chemistry.FEATURE_HEADER])
  report={'status':'PASS_LABEL_BLIND_C2E_TEST_FEATURE_EXPORT','protocol_sha256':lock.PROTOCOL_SHA,
   'queries':512,'candidate_rows':131072,'candidate_rows_per_query':256,
   'requests_sha256':meta['requests_sha256'],'hits_sha256':lock.sha(hit),
   'features_sha256':lock.sha(tsv),'catalog_sha256':lock.CAT_SHA,
   'model_sha256':lock.MODEL_SHA,'source_partition':'test',
   'source_scoring_job_id':str(os.environ.get('SLURM_JOB_ID','manual')),
   'reference_truth_accessed':False,'model_fit_performed':False,
   'rank_changed':False,'chemistry_implementation_sha256':CHEM_SHA}
  (tmp/'feature_provenance.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
  tmp.rename(out)
 except Exception:
  (tmp/'FAILED_DO_NOT_USE').write_text('Partial export. Preserve it.\n');raise
 print('REDEEM_C2E_FEATURE_EXPORT=PASS rows=131072 label_free=YES')
 return report

def main():
 p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('requests','hits','metadata','protocol','output')]
 a=p.parse_args();export(a.requests,a.hits,a.metadata,a.protocol,a.output)
if __name__=='__main__':main()
