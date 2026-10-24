import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import redeem_c2d_evaluate_20261024 as E
import redeem_c2d_export_20261024 as F

class Contract(unittest.TestCase):
 def test_paired_identical(self):
  truth={'a':'A','b':'B'}
  geom={'a':['A','C','D','E'],'b':['B','A','C','D']}
  c2={'a':['A','D','C','E'],'b':['B','C','D','A']}
  m=E.metrics(truth,geom,c2,query_count=2,pool=4)
  self.assertEqual(m['legacy_top1_count'],2)
  self.assertEqual(m['c2_top1_count'],2)
  self.assertEqual(m['acceptance'],'FAIL_FIXED_C2_DEV3_GATE_STOP_NO_RETUNING')
 def test_positive_pairing(self):
  truth={str(i):'A' for i in range(6)}
  geom={str(i):['B','A','C'] for i in range(6)}
  c2={str(i):['A','B','C'] for i in range(6)}
  m=E.metrics(truth,geom,c2,query_count=6,pool=3)
  self.assertEqual(m['paired_corrected'],6)
  self.assertEqual(m['paired_regressed'],0)
  self.assertEqual(m['acceptance'],'PASS_FIXED_C2_DEV3_GATE')
 def test_rejected_changed_candidate_universe(self):
  with self.assertRaisesRegex(ValueError,'Incomplete/different'):
   E.metrics({'a':'A'},{'a':['A','B']},{'a':['A','Z']},query_count=1,pool=2)
 def test_rejected_missing_spectrum(self):
  with self.assertRaisesRegex(ValueError,'Missing or extra'):
   E.metrics({'a':'A'},{'a':['A']},{'a':['A'],'b':['B']},query_count=1,pool=1)
 def test_paired_ci_is_deterministic(self):
  one=E.paired_ci([1,0,-1,1,0,-1,1],n=100)
  self.assertEqual(one,E.paired_ci([1,0,-1,1,0,-1,1],n=100))
 def test_dev_metadata_fail_closed(self):
  m={'status':'PASS_C2_D_PROSPECTIVE_DEV3_PREPARATION','source_partition':'validation',
    'source_revision':F.REV,'queries':512,'requests_sha256':F.REQUEST_SHA,
    'frozen_model_sha256':F.MODEL_SHA,'protected_test_accessed':False,
    'catalog_or_ranker_metrics_used_for_selection':False,
    'both_previous_dev_sequence_exclusion':True,'train_sequence_exclusion':True}
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'meta.json';p.write_text(json.dumps(m))
   self.assertEqual(F.verify_meta(p),m)
   m['source_partition']='train';p.write_text(json.dumps(m))
   with self.assertRaisesRegex(ValueError,'source_partition'):F.verify_meta(p)
if __name__=='__main__':unittest.main()
