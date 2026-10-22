#!/usr/bin/env python3
"""Regression tests for label-blind human FASTA reference catalog builder."""
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MOD_PATH=Path(__file__).with_name('redeem_reference_catalog_20261022.py')
spec=importlib.util.spec_from_file_location('reference_builder',MOD_PATH)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class ReferenceCatalogTests(unittest.TestCase):
    def test_trypsin_proline_and_missed(self):
        self.assertEqual(list(mod.digest('PEPTIDER')), ['PEPTIDER'])
        self.assertEqual(list(mod.digest('PEPTIDKAEPTIDER', missed=0)), ['PEPTIDK', 'AEPTIDER'])
        self.assertEqual(list(mod.digest('PEPTIDKAEPTIDER', missed=1)), ['PEPTIDK', 'PEPTIDKAEPTIDER', 'AEPTIDER'])
        self.assertEqual(list(mod.digest('PEPTIDKPQR', missed=0)), ['PEPTIDKPQR'])

    def test_build_is_independent_and_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'ref.fasta').write_text('>sp|A|HUMAN\nMPEPTIDERKPEPTIDK\n')
            out=p/'reference'
            result=mod.build(p/'ref.fasta',out)
            self.assertFalse(result['candidate_source_uses_truth'])
            self.assertFalse(result['candidate_source_uses_requests'])
            self.assertEqual(result['candidate_count'],2*result['unique_peptide_sequences'])
            keys,n=mod.catalog_keys_and_count(out/'catalog.yaml')
            self.assertEqual(n,result['candidate_count'])
            self.assertIn('MPEPTIDER|z2',keys)
            self.assertIn('MPEPTIDER|z3',keys)
            with self.assertRaisesRegex(ValueError,'exists'):
                mod.build(p/'ref.fasta',out)

    def test_evaluation_after_creation_and_partition_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'ref.fasta').write_text('>human1\nMPEPTIDERKPEPTIDK\n')
            cat=p/'reference';mod.build(p/'ref.fasta',cat)
            coh=p/'cohort';coh.mkdir()
            with (coh/'truth.tsv').open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['query_id','truth_candidate_key','partition','evidence_source'])
                for i in range(512):w.writerow([f'q{i}','MPEPTIDER|z2' if i==0 else f'NOTFOUNDPEPTIDE{i}|z2','DEV','fixture'])
            (coh/'requests.tsv').write_text('synthetic pinned bytes\n')
            (coh/'metadata.json').write_text(json.dumps({'source_partition':'validation','selected_queries':512,'source_revision':mod.DATASET_REVISION,'protected_partition_accessed':False}))
            with patch.object(mod,'TRUTH_SHA',mod.sha(coh/'truth.tsv')):
                with patch.object(mod,'REQUESTS_SHA',mod.sha(coh/'requests.tsv')):
                    result=mod.audit(cat,coh,p/'coverage.json')
                    self.assertEqual(result['reference_in_catalog'],1)
                    self.assertEqual(result['next_step'],'STOP_NO_SCORING_RUN_BELOW_50_PERCENT')
                    with self.assertRaises(ValueError):mod.audit(cat,coh,p/'coverage.json')
                    meta=json.loads((coh/'metadata.json').read_text());meta['source_partition']='test'
                    (coh/'metadata.json').write_text(json.dumps(meta))
                    with self.assertRaisesRegex(ValueError,'partition'):
                        mod.audit(cat,coh,p/'blocked.json')

    def test_budget_no_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'ref.fasta').write_text('>X\nMPEPTIDERKPEPTIDK\n')
            with self.assertRaisesRegex(ValueError,'budget'):
                mod.build(p/'ref.fasta',p/'no',max_candidates=1)
            self.assertFalse((p/'no').exists())

if __name__=='__main__':unittest.main()
