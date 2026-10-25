#!/usr/bin/env python3
"""C2 TRAIN-only label-blind feature export for arbitrary pinned cohort size.

Reads only the pinned measured peak requests and unmodified v0.77 hit ranks.
Never reads reference identities, PTM labels, training/test data, or refits scores.
The existing geometric rank is retained as a *feature*, not replaced.
"""
from __future__ import annotations
import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

# Monoisotopic amino-acid residue masses (Da), without water.
AA = {
 'A':71.037113805,'R':156.101111050,'N':114.042927470,'D':115.026943065,
 'C':103.009184505,'E':129.042593135,'Q':128.058577540,'G':57.021463735,
 'H':137.058911875,'I':113.084064015,'L':113.084064015,'K':128.094963050,
 'M':131.040484645,'F':147.068413945,'P':97.052763875,'S':87.032028435,
 'T':101.047678505,'W':186.079312980,'Y':163.063328575,'V':99.068413945,
}
ALLOWED = set(AA) - {'C'}
H2O = 18.010564684
NH3 = 17.026549101
PROTON = 1.00727646677
ISOTOPE = 1.0033548378
EXPECTED_POOL = 256
REQ_HEADER = ['query_id','observed_precursor_mz','observed_charge','top_k','peak_mz','intensity']
HIT_HEADER = ['query_id','rank','candidate_index','key','charge','theoretical_neutral_mass','absolute_neutral_mass_error','geometry_score']
FEATURE_HEADER = [
 'query_id','original_rank','candidate_index','key','charge','geometry_score',
 'precursor_mass_error_da','precursor_mass_error_ppm','best_isotope_offset',
 'best_isotope_residual_da','observed_peak_count','theoretical_backbone_ion_count',
 'matched_backbone_ions','matched_backbone_fraction','b_cleavage_fraction',
 'y_cleavage_fraction','complementary_pair_fraction','longest_b_ladder',
 'longest_y_ladder','matched_neutral_loss_ions','neutral_loss_ion_count',
 'explained_sqrt_intensity_fraction','median_backbone_mass_error_ppm',
]


def sha(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
 return h.hexdigest()


def fail(message):
 raise ValueError(message)


def read_rows(path, header):
 with path.open(newline='') as f:
  reader=csv.DictReader(f,delimiter='\t')
  if reader.fieldnames != header: fail(f'Unexpected TSV header: {path}: {reader.fieldnames}')
  for row in reader:
   if any(v is None for v in row.values()): fail(f'Truncated TSV row: {path}')
   yield row


def positive_finite(x):
 value=float(x)
 if not math.isfinite(value) or value<=0: fail('Expected positive finite number')
 return value


def load_requests(path,expected_queries):
 observations={}
 for row in read_rows(path,REQ_HEADER):
  q=row['query_id']
  if not q: fail('Empty query ID')
  z=int(row['observed_charge']); top=int(row['top_k']); mz=positive_finite(row['observed_precursor_mz'])
  if z not in (2,3) or top!=256: fail('Unexpected precursor charge/top_k')
  peak=(positive_finite(row['peak_mz']),positive_finite(row['intensity']))
  if q not in observations: observations[q]={'charge':z,'mz':mz,'peaks':[]}
  record=observations[q]
  if record['charge']!=z or record['mz']!=mz: fail('Inconsistent repeated request header')
  record['peaks'].append(peak)
 if len(observations)!=expected_queries: fail(f'Expected {expected_queries} spectra, observed {len(observations)}')
 for record in observations.values():
  if not 10<=len(record['peaks'])<=256: fail('Unexpected retained peak count')
  record['peaks'].sort(key=lambda p:(p[0],p[1]))
  record['mass_list']=[x[0] for x in record['peaks']]
  record['sqrt_sum']=sum(math.sqrt(v) for _,v in record['peaks'])
 return observations


def parse_candidate(key, charge):
 if not key.endswith(f'|z{charge}') or key.count('|')!=1: fail(f'Unsupported modified candidate key: {key}')
 seq=key[:-len(f'|z{charge}')]
 if not 7<=len(seq)<=64 or not set(seq)<=ALLOWED: fail(f'Unsupported residue/length: {key}')
 return seq


def theoretical(seq, precursor_charge):
 """Return b/y ion tuples (family, cleavage index, charge, mz, loss eligibility)."""
 prefix=[];total=0.0
 for residue in seq:
  total+=AA[residue]
  prefix.append(total)
 result=[]
 for cut in range(1,len(seq)):
  left=seq[:cut];right=seq[cut:]
  # b_n: residue sum, y_(L-n): residue sum + H2O.
  for family,frag,neutral in [('b',left,prefix[cut-1]),('y',right,total-prefix[cut-1]+H2O)]:
   for ion_z in (1,2):
    if ion_z>precursor_charge:continue
    mz=(neutral + ion_z*PROTON)/ion_z
    losses=[]
    if set(frag)&set('STED'):losses.append(('H2O',H2O))
    if set(frag)&set('NQKR'):losses.append(('NH3',NH3))
    result.append((family,cut,ion_z,mz,tuple(losses)))
 return result


def match(obs, ion_mz):
 """Choose nearest peak within max(0.02 Da, 20 ppm), unique index return."""
 masses=obs['mass_list']; tol=max(0.02,20e-6*ion_mz)
 # Search all peaks in window; deterministic nearest, then higher intensity.
 lo=bisect.bisect_left(masses,ion_mz-tol);hi=bisect.bisect_right(masses,ion_mz+tol)
 if lo>=hi:return None
 return min(range(lo,hi),key=lambda i:(abs(masses[i]-ion_mz),-obs['peaks'][i][1],i))


def longest_run(cuts):
 best=run=0;previous=None
 for cut in sorted(cuts):
  run=run+1 if previous is not None and cut==previous+1 else 1
  best=max(best,run);previous=cut
 return best


def features(obs, hit):
 key=hit['key'];z=int(hit['charge']);seq=parse_candidate(key,z)
 geom=float(hit['geometry_score']);delta=float(hit['absolute_neutral_mass_error'])
 neutral=positive_finite(hit['theoretical_neutral_mass'])
 if not math.isfinite(geom) or not math.isfinite(delta) or delta<0:fail('Invalid hit score or error')
 observed_neutral=z*(obs['mz']-PROTON)
 if abs(abs(observed_neutral-neutral)-delta)>1e-4:fail('Hit neutral mass inconsistent with precursor')
 ions=theoretical(seq,z); matched=set();bcuts=set();ycuts=set();residuals=[]
 loss_matches=loss_total=0;backbone_count=0;backbone_match=0
 for family,cut,ion_z,mz,losses in ions:
  backbone_count+=1
  m=match(obs,mz)
  if m is not None:
   backbone_match+=1;matched.add(m)
   (bcuts if family=='b' else ycuts).add(cut)
   residuals.append(abs(obs['mass_list'][m]-mz)/mz*1e6)
  for _,shift in losses:
   loss_total+=1
   loss_mz=mz-shift/ion_z
   if loss_mz>0 and match(obs,loss_mz) is not None:loss_matches+=1
 ncuts=len(seq)-1
 # Fraction of cleavage positions supported at both sequence ends.
 complementary=len(bcuts & ycuts)
 offset=min((-1,0,1,2),key=lambda n:(abs(observed_neutral-neutral-n*ISOTOPE),abs(n),n))
 isotope_residual=abs(observed_neutral-neutral-offset*ISOTOPE)
 sqrt_explained=sum(math.sqrt(obs['peaks'][i][1]) for i in matched)/obs['sqrt_sum']
 result={
  'geometry_score':geom,'precursor_mass_error_da':delta,
  'precursor_mass_error_ppm':delta/neutral*1e6,
  'best_isotope_offset':offset,'best_isotope_residual_da':isotope_residual,
  'observed_peak_count':len(obs['peaks']),'theoretical_backbone_ion_count':backbone_count,
  'matched_backbone_ions':backbone_match,'matched_backbone_fraction':backbone_match/backbone_count,
  'b_cleavage_fraction':len(bcuts)/ncuts,'y_cleavage_fraction':len(ycuts)/ncuts,
  'complementary_pair_fraction':complementary/ncuts,
  'longest_b_ladder':longest_run(bcuts),'longest_y_ladder':longest_run(ycuts),
  'matched_neutral_loss_ions':loss_matches,'neutral_loss_ion_count':loss_total,
  'explained_sqrt_intensity_fraction':sqrt_explained,
  'median_backbone_mass_error_ppm':statistics.median(residuals) if residuals else None,
 }
 return result


def fmt(x):
 if x is None:return ''
 if isinstance(x,float):return format(x,'.10g')
 return str(x)


def export(args):
 req=Path(args.requests);hits=Path(args.hits);out=Path(args.output_dir)
 if out.exists():fail('Output exists; do not overwrite')
 meta=json.loads(Path(args.train_meta).read_text())
 if meta.get('source_partition')!='train' or meta.get('status')!='PASS_TRAIN_ONLY_SOURCE_PREPARATION' or meta.get('protected_test_accessed') is not False:fail('Not verified TRAIN input')
 if meta.get('selected_queries')!=args.queries or meta.get('requests_sha256')!=args.requests_sha:fail('TRAIN metadata mismatch')
 for path,expected in [(req,args.requests_sha),(hits,args.hits_sha)]:
  if not path.is_file() or sha(path)!=expected:fail(f'Pinned SHA-256 mismatch or missing file: {path}')
 requests=load_requests(req,args.queries)
 # Validate each query's full rank sequence BEFORE writing output.
 rows=[];last_rank={};seen=set()
 for row in read_rows(hits,HIT_HEADER):
  q=row['query_id']
  if q not in requests:fail('Unexpected query in hits')
  rank=int(row['rank']);idx=int(row['candidate_index']);z=int(row['charge'])
  if rank!=last_rank.get(q,0)+1 or idx<0 or z!=requests[q]['charge']:
   fail('Wrong candidate order/rank/charge')
  last_rank[q]=rank
  if rank>EXPECTED_POOL:fail('More than expected candidates for query')
  key=(q,row['key'])
  if key in seen:fail('Duplicate candidate within query')
  seen.add(key)
  rows.append(row)
 if set(last_rank)!=set(requests) or any(r!=EXPECTED_POOL for r in last_rank.values()):
  fail('Missing/truncated candidate pools; expected exactly 256 per query in real run')
 if len(rows)!=args.queries*EXPECTED_POOL:fail('Unexpected candidate row count')
 if not out.parent.is_dir():fail('Output parent does not exist')
 tmp=Path(tempfile.mkdtemp(prefix='.redeem-chem-',dir=out.parent))
 try:
  outfile=tmp/'candidate_chemistry_features.tsv';count=0
  with outfile.open('w',newline='') as f:
   writer=csv.writer(f,delimiter='\t',lineterminator='\n')
   writer.writerow(FEATURE_HEADER)
   for row in rows:
    q=row['query_id'];z=int(row['charge']);v=features(requests[q],row)
    record={'query_id':q,'original_rank':int(row['rank']),
            'candidate_index':int(row['candidate_index']), 'key':row['key'],'charge':z,**v}
    writer.writerow([fmt(record[k]) for k in FEATURE_HEADER]);count+=1
  report={
   'status':'PASS_TRAIN_ONLY_CHEMISTRY_FEATURE_EXPORT',
   'protocol':'redeem_c2_20261023_train_chemistry_features_v1',
   'source_scoring_job_id':str(os.environ.get('SLURM_JOB_ID','manual')),'queries':len(requests), 'candidate_rows':count,
   'candidate_rows_per_query':EXPECTED_POOL,'requests_sha256':sha(req),'hits_sha256':sha(hits),
   'features_sha256':sha(outfile),'training_performed':False,'scoring_fit_performed':False,
   'reference_truth_accessed':False,'protected_test_accessed':False,
   'feature_names':FEATURE_HEADER[6:],
   'scope':'unmodified_cysteine_free_z2_z3',
   'limitations':[
     'Fragment matching is descriptive; multiple theoretical ions can reuse the same observed peak.',
     'Neutral-loss eligibility is a simple residue rule, not a predicted fragmentation probability.',
     'Isotope-offset choice is an explanatory feature, not precursor validation or a new mass calibration.',
     'Original frozen geometry scores and ranks remain unchanged; no new identification accuracy is claimed.',
     'Input MS/MS spectra have PSM-derived evidence; historical foundation TRAIN overlap is not evaluated.',
     'Only TRAIN labels may be used in the separate fit stage; never use exposed DEV512.'
   ],
  }
  (tmp/'feature_provenance.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  tmp.rename(out)
 except BaseException:
  # Preserve an output from a failed run for forensic inspection, but do not mark it complete.
  (tmp/'FAILED_DO_NOT_USE').write_text('Feature export incomplete. Inspect original inputs and logs.\n')
  raise
 print('REDEEM_CHEMISTRY_FEATURES=PASS rows=%s queries=%s' % (count,len(requests)))
 print('REDEEM_CHEMISTRY_OUTPUT='+str(out))


def cli():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--requests',required=True);p.add_argument('--hits',required=True)
 p.add_argument('--train-meta',required=True);p.add_argument('--queries',type=int,required=True)
 p.add_argument('--requests-sha',required=True);p.add_argument('--hits-sha',required=True)
 p.add_argument('--output-dir',required=True)
 a=p.parse_args()
 if not 128<=a.queries<=8192 or len(a.requests_sha)!=64 or len(a.hits_sha)!=64:
  fail('Bad cohort size or missing pinned hashes')
 export(a)

if __name__=='__main__':cli()
