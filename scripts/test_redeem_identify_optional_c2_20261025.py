"""Synthetic-only optional C2 inference tests. Never use measured DEV or TEST."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import csv
import hashlib
import itertools
import json
import math
import sys
import unittest

import redeem_identify_optional_c2_20261025 as adapter
import redeem_c2_train_features_20261023 as chem
import redeem_c2_ranker_20261023 as trained


def fixture(base: Path, n=256):
    base.mkdir(parents=True, exist_ok=True)
    req = base / 'requests.tsv'; catalog = base / 'catalog.yaml'; native = base / 'frozen_hits.tsv'
    catalog.write_text('synthetic candidate catalog, no evaluated peptides\n')
    precursor = 600.0; charge = 2
    with req.open('w', newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(chem.REQ_HEADER)
        for i in range(12):w.writerow(['synthetic-q',precursor,charge,256,125.0+i*65,100.0+i])
    sequences = [''.join(t) for t in itertools.islice(itertools.product('AGKP',repeat=7),n)]
    with native.open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(chem.HIT_HEADER)
        for rank,seq in enumerate(sequences,1):
            neutral = sum(chem.AA[c] for c in seq) + chem.H2O
            delta = abs(charge*(precursor-chem.PROTON)-neutral)
            geom = 0.05 - rank*0.0001
            w.writerow(['synthetic-q',rank,rank-1,f'{seq}|z2',charge,
                        format(neutral,'.12f'),format(delta,'.12f'),format(geom,'.12f')])
    model={
        'schema':'redeem.c2.chemistry_pairwise_ranker.v1',
        'status':'TRAIN_ONLY_FROZEN_COEFFICIENTS_NOT_DEV_EVALUATED',
        'feature_names':trained.FEATURES,'transformations':trained.TRANSFORMS,
        'config':trained.CONFIG,'base_score':'unchanged_geometry_score',
        'missing_fragment_error_ppm_sentinel':1000,
        'standardization_mean':[0.0]*6,
        'standardization_std_min_005':[1.0]*6,
        'coefficients':[0.1,0.0,0.5,0.0,0.5,0.0],
        'validation_labels_accessed':False,'test_accessed':False
    }
    m=base/'model.json';m.write_text(json.dumps(model,sort_keys=True)+'\n')
    fake=base/'fake_identifier.py'
    fake.write_text("""import sys,shutil\na=sys.argv[1:]\nif a==['--version']: print('v0.77.0-practical-identifier-stream')\nelif a==['--self-test']:print('v0770_practical_identifier_stream_self_test=PASS')\nelif a[0]=='identify-stream':\n shutil.copyfile(%r,a[a.index('--output-hits')+1]);open(a[a.index('--output-summary')+1],'w').write('key\\tvalue\\nstatus\\tPASS\\n')\nelse:raise SystemExit(2)\n"""%str(native))
    return req,catalog,native,m,fake


def args(base,req,cat,model,ranker='c2-experimental'):
    return SimpleNamespace(ranker=ranker,catalog=cat,spectra=req,output_hits=base/'out_hits.tsv',
                           output_summary=base/'out_summary.tsv',model=model if ranker!='geometry' else None,
                           queries=1,output_native_hits=base/'original.tsv' if ranker!='geometry' else None,
                           output_scores=base/'scores.tsv' if ranker!='geometry' else None,
                           identifier_exe=base/'fake_identifier.py',sif=None,bind_root=None,
                           container_runtime='singularity')


class OptionalC2Test(TestCase):
    @classmethod
    def setUpClass(cls):
        adapter.load_bundle(Path(adapter.__file__).parent)

    def test_geometry_preserves_native_hit_bytes(self):
        with TemporaryDirectory() as folder:
            b=Path(folder);r,c,h,m,fake=fixture(b)
            a=args(b,r,c,m,'geometry')
            with patch.object(adapter,'identifier_command',return_value=[sys.executable,str(fake)]):
                result=adapter.run(a)
            self.assertEqual(result['ranker'],'geometry')
            self.assertEqual(a.output_hits.read_bytes(),h.read_bytes())
            self.assertEqual(a.output_summary.read_text(),'key\tvalue\nstatus\tPASS\n')

    def test_c2_preserves_pool_and_writes_explicit_provenance(self):
        with TemporaryDirectory() as folder:
            b=Path(folder);r,c,h,m,fake=fixture(b)
            a=args(b,r,c,m)
            with patch.object(adapter,'identifier_command',return_value=[sys.executable,str(fake)]):
                result=adapter.run(a,expected_model=adapter.sha(m),expected_catalog=adapter.sha(c))
            self.assertEqual(result['status'],adapter.STATUS)
            self.assertEqual(a.output_native_hits.read_bytes(),h.read_bytes())
            self.assertIn('production_approved\tNO',a.output_summary.read_text())
            self.assertIn('model_sha256\t'+adapter.sha(m),a.output_summary.read_text())
            ranked=list(chem.read_rows(a.output_hits,chem.HIT_HEADER))
            original=list(chem.read_rows(h,chem.HIT_HEADER))
            self.assertEqual(len(ranked),256)
            self.assertEqual({(x['key'],x['candidate_index']) for x in ranked},
                             {(x['key'],x['candidate_index']) for x in original})
            self.assertEqual([int(x['rank']) for x in ranked],list(range(1,257)))

    def test_numpy_reference_apply_has_identical_ranking(self):
        with TemporaryDirectory() as folder:
            b=Path(folder);r,c,h,m,fake=fixture(b)
            obs,hits=adapter.load_native(r,h,1)
            feature_file=b/'features.tsv';adapter.generate_features(obs,hits,feature_file)
            pure=b/'pure.tsv';adapter.apply_frozen_c2(feature_file,pure,json.loads(m.read_text()),1)
            numpy=b/'numpy.tsv'
            trained.apply(SimpleNamespace(model=m,features=feature_file,
                            features_sha=adapter.sha(feature_file),queries=1,output=numpy))
            self.assertEqual([(x['candidate_key'],x['original_rank']) for x in
                              adapter.read_rankings(pure)],
                             [(x['candidate_key'],x['original_rank']) for x in
                              adapter.read_rankings(numpy)])

    def test_wrong_model_and_catalog_fail_closed(self):
        with TemporaryDirectory() as folder:
            b=Path(folder);r,c,h,m,fake=fixture(b);a=args(b,r,c,m)
            with self.assertRaisesRegex(ValueError,'catalog SHA'):
                adapter.run(a,expected_catalog='0'*64)
            with self.assertRaisesRegex(ValueError,'model missing or SHA'):
                adapter.run(a,expected_catalog=adapter.sha(c),expected_model='0'*64)
            self.assertFalse(a.output_hits.exists())

    def test_incomplete_pool_and_unsupported_modified_key(self):
        with TemporaryDirectory() as folder:
            b=Path(folder);r,c,h,m,fake=fixture(b)
            rows=list(chem.read_rows(h,chem.HIT_HEADER))
            def replace(rows):
                with h.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=chem.HIT_HEADER,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
            replace(rows[:-1])
            with self.assertRaisesRegex(ValueError,'complete candidates'):
                adapter.load_native(r,h,1)
            rows[0]['key']='AGKPAAC|R2:UniMod:4|z2'
            replace(rows)
            with self.assertRaisesRegex(ValueError,'Unsupported modified candidate'):
                obs,hits=adapter.load_native(r,h,1)
                adapter.generate_features(obs,hits,b/'bad.tsv')

    def test_c2_rejects_unpinned_direct_binary_and_existing_outputs(self):
        with TemporaryDirectory() as folder:
            b=Path(folder);r,c,h,m,fake=fixture(b);a=args(b,r,c,m)
            with self.assertRaisesRegex(ValueError,'parity-qualified SIF'):
                adapter.identifier_command(a)
            a.output_hits.write_text('existing output')
            with self.assertRaisesRegex(ValueError,'Output already exists'):
                adapter.run(a,expected_catalog=adapter.sha(c),expected_model=adapter.sha(m))
            self.assertEqual(a.output_hits.read_text(),'existing output')


# Read-only helper for rank comparison; no TEST labels or runtime state.
def read_rankings(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file, delimiter='\t'))

adapter.read_rankings = read_rankings

if __name__=='__main__':unittest.main()
