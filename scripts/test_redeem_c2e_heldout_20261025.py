#!/usr/bin/env python3
"""Synthetic only. No real held-out data or network access."""
import json,random,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
import redeem_c2e_lock_20261025 as lock
import redeem_c2e_prepare_20261025 as prep
import redeem_c2e_evaluate_20261025 as ev
import redeem_c2d_evaluate_20261024 as original

ROOT=Path(__file__).parent

def sequence(n):
 aa='ADEFGHIKLMNPQRSTVWY';z=''
 for _ in range(7):z=aa[n%len(aa)]+z;n//=len(aa)
 return z

def spectra(num=520):
 for n in range(num):
  q=sequence(n)
  yield {'sequence':q,'modified_sequence':q,'precursor_charge':2,'precursor_mz':500.0,
   'experiment_name':'TUM_first_pool','evidence_index':n+1,'scan_number':n+100,
   'mz_array':[float(i+50) for i in range(12)],'intensity_array':[float(i+1) for i in range(12)]}

class Contract(unittest.TestCase):
 def test_protocol_lock_and_mutation(self):
  f=ROOT/'LOCKED_PROTOCOL_20261025.json';p=lock.protocol(f)
  self.assertEqual(p['data_partition'],'test')
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'protocol.json';path.write_bytes(f.read_bytes()+b' ')
   with self.assertRaisesRegex(ValueError,'PREREGISTRATION_CHANGED'):lock.protocol(path)

 def test_selection_exact512_and_order_independent(self):
  rows=list(spectra(520))
  selected,a=prep.select(rows,set(),set())
  selected2,b=prep.select(list(reversed(rows)),set(),set())
  self.assertEqual([r['query_id'] for r in selected],[r['query_id'] for r in selected2])
  self.assertEqual(len(selected),512)
  self.assertEqual(len({r['seq'] for r in selected}),512)
  self.assertTrue(all(r['query_id'].startswith('pttest-') for r in selected))

 def test_prior_sequence_excluded_and_charge(self):
  rows=list(spectra(513));excluded={rows[0]['sequence']}
  selected,_=prep.select(rows,excluded,set())
  self.assertEqual(len(selected),512)
  self.assertNotIn(rows[0]['sequence'],{r['seq'] for r in selected})
  other=dict(rows[4]);other['precursor_charge']=4
  self.assertIsNone(prep.eligible(other,set(),set()))

 def test_one_time_output_and_partition_without_real_test(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);out=root/'testcohort'
   fake_model={'training_top1_geometry_count':888,'training_top1_chemistry_count':931}
   class Args:
    output=str(out);protocol=str(ROOT/'LOCKED_PROTOCOL_20261025.json')
    model_dir='stub';dev1='stub';dev2='stub';dev3='stub';train='stub';parquet='stub'
   with patch.object(prep,'freeze',return_value=fake_model),patch.object(prep,'embargo',return_value=(set(),set())):
    meta=prep.prepare(Args(),stream_override=spectra(520))
    self.assertEqual(meta['status'],'PASS_C2E_HELDOUT_TEST_PREPARATION')
    self.assertEqual(meta['seed'],20261025)
    self.assertEqual(meta['source_partition'],'test')
    self.assertEqual(meta['protocol_sha256'],lock.PROTOCOL_SHA)
    self.assertTrue(all('TEST'==v['partition'] for v in original.rows(out/'truth.tsv',prep.TRUTH)))
    self.assertEqual(lock.test_metadata(out/'metadata.json',Args.protocol)['truth_sha256'],lock.sha(out/'truth.tsv'))
    with self.assertRaisesRegex(ValueError,'Output exists'):
     prep.prepare(Args(),stream_override=spectra(520))

 def test_paired_bootstrap_determinism(self):
  d=[0]*261+[1]*15+[-1]*5
  self.assertEqual(original.paired_ci(d,seed=20261025),original.paired_ci(d,seed=20261025))
  self.assertEqual(len(original.paired_ci(d,seed=20261025)),2)

 def test_pool_identity_check(self):
  truth={'a':'T','b':'T'};geom={'a':['T','X'],'b':['X','T']};c2={'a':['X','T'],'b':['T','X']}
  m=original.metrics(truth,geom,c2,query_count=2,pool=2)
  self.assertEqual(m['paired_corrected'],1);self.assertEqual(m['paired_regressed'],1)
  c2['a']=['X','Y']
  with self.assertRaisesRegex(ValueError,'different candidate pools'):
   original.metrics(truth,geom,c2,query_count=2,pool=2)

 def test_metadata_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'meta.json';m={'status':'PASS_C2E_HELDOUT_TEST_PREPARATION','source_partition':'test',
    'source_revision':lock.REV,'queries':512,'seed':20261025,
    'protocol_sha256':lock.PROTOCOL_SHA,'frozen_model_sha256':lock.MODEL_SHA,
    'disjoint_from_dev1_dev2_dev3_and_c2_train':True,'selection_uses_scores_or_catalog':False}
   path.write_text(json.dumps(m));lock.test_metadata(path,ROOT/'LOCKED_PROTOCOL_20261025.json')
   m['source_partition']='validation';path.write_text(json.dumps(m))
   with self.assertRaisesRegex(ValueError,'cohort metadata'):lock.test_metadata(path,ROOT/'LOCKED_PROTOCOL_20261025.json')

if __name__=='__main__':unittest.main()
