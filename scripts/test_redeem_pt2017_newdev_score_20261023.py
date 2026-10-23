#!/usr/bin/env python3
"""Contract tests for PT2017 source-scoped external DEV512 scoring report."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE=Path(__file__).with_name('redeem_pt2017_newdev_score_20261023.py')
spec=importlib.util.spec_from_file_location('redeem_pt2017_newdev_score',SOURCE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sample():
    return dict(queries=512,truth_in_catalog=263,truth_in_frozen_256_pool=200,
                legacy_top1=145/512,diagnostic_top1=147/512,
                legacy_top10=180/512,legacy_mrr=.31,diagnostic_mrr=.312,
                changed_top1=18,diagnostic_corrected=5,diagnostic_regressed=3)


class ScoreContracts(unittest.TestCase):
    def test_conditional_denominators(self):
        r=mod.summary(sample())
        self.assertEqual(r['nearest256_recall_unconditional'],200/512)
        self.assertEqual(r['nearest256_recall_given_catalog'],200/263)
        self.assertEqual(r['legacy_top1_given_truth_in_pool'],145/200)
        self.assertEqual(r['diagnostic_top1_given_truth_in_pool'],147/200)
        self.assertIs(r['independent_test_accessed'],False)
        self.assertIs(r['candidate_score_fit_performed'],False)
    def test_no_pool_is_reported_not_invented(self):
        d=sample();d.update(truth_in_frozen_256_pool=0,legacy_top1=0.0,diagnostic_top1=0.0,diagnostic_corrected=0,diagnostic_regressed=0)
        r=mod.summary(d)
        self.assertIsNone(r['legacy_top1_given_truth_in_pool'])
    def test_bogus_coverage_rejected(self):
        with self.assertRaises(ValueError):
            mod.summary({**sample(),'truth_in_frozen_256_pool':264})
    def test_noninteger_correct_counts_rejected(self):
        with self.assertRaises(ValueError):
            mod.summary({**sample(),'legacy_top1':.32211})
    def test_correction_arithmetic_rejected(self):
        with self.assertRaises(ValueError):
            mod.summary({**sample(),'diagnostic_regressed':2})
    def test_sha_guard_rejects_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data';p.write_bytes(b'hello')
            ok=mod.digest(p)
            mod.checked(p,ok,'fixture')
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):mod.checked(p,ok,'fixture')


if __name__=='__main__':unittest.main()
