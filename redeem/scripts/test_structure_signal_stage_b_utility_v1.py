#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import structure_signal_stage_b_utility_v1 as utility


class StageBUtilityTests(unittest.TestCase):
    def test_group_folds_keep_peptidoform_together(self) -> None:
        groups = ["PEP-A", "PEP-A", "PEP-B", "PEP-C", "PEP-D", "PEP-E"]
        assignments = utility.group_folds(groups, 3)
        self.assertEqual(assignments["PEP-A"], assignments["PEP-A"])
        self.assertEqual(set(assignments), set(groups))

    def test_structure_label_columns_are_rejected(self) -> None:
        rows = [{"identity_key": "PEP|z2", "target_ccs": "300", "radius_of_gyration_mean": "4"}]
        with self.assertRaisesRegex(ValueError, "label columns"):
            utility.choose_structure_columns(list(rows[0]), rows)

    def test_v038_prediction_aggregation_uses_identity_median(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pred.tsv"
            fields = [
                "record_index",
                "source_id",
                "identity_key",
                "sequence",
                "peptidoform",
                "charge",
                "precursor_mz",
                "target_ccs",
                "predicted_ccs",
            ]
            rows = [
                ["1", "a", "PEP|z2", "PEP", "PEP", "2", "400", "300", "290"],
                ["2", "b", "PEP|z2", "PEP", "PEP", "2", "400", "320", "305"],
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(fields)
                writer.writerows(rows)
            labels, count, raw_mae = utility.aggregate_v038_predictions(path)
            self.assertEqual(count, 2)
            self.assertAlmostEqual(raw_mae, 12.5)
            self.assertAlmostEqual(labels["PEP|z2"].target_ccs_median, 310.0)
            self.assertAlmostEqual(labels["PEP|z2"].predicted_ccs_median, 297.5)
            self.assertAlmostEqual(labels["PEP|z2"].residual_ccs_median, 12.5)
            self.assertEqual(labels["PEP|z2"].source_count, 2)

    def test_manifest_fingerprint_changes_with_identity(self) -> None:
        rows = [
            {
                "stratum_key": "z2|03_13-18|unmodified",
                "selection_hash_sha256": "a" * 64,
                "identity_key": "PEPA|z2",
            },
            {
                "stratum_key": "z3|03_13-18|unmodified",
                "selection_hash_sha256": "b" * 64,
                "identity_key": "PEPB|z3",
            },
        ]
        first = utility.manifest_fingerprint(rows)
        rows[1]["identity_key"] = "PEPC|z3"
        second = utility.manifest_fingerprint(rows)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
