#!/usr/bin/env python3
"""Targeted tests for the deterministic Stage B manifest selector."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("structure_signal_stage_b_manifest_v1.py")
SPEC = importlib.util.spec_from_file_location("stage_b_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stage_b = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage_b
SPEC.loader.exec_module(stage_b)


class StageBManifestTests(unittest.TestCase):
    def test_classification_contract(self) -> None:
        base = {"modified": "YES"}
        self.assertEqual(
            stage_b.classify_ptm(
                {
                    **base,
                    "modification_labels": "UniMod:4@Residue:C:x",
                }
            ),
            "carbamidomethyl",
        )
        self.assertEqual(
            stage_b.classify_ptm(
                {
                    **base,
                    "modification_labels": "UniMod:4@Residue:C:x;UniMod:35@Residue:M:x",
                }
            ),
            "common_ptm_combination",
        )
        self.assertEqual(
            stage_b.classify_ptm(
                {
                    **base,
                    "modification_labels": "UniMod:4@Residue:C:x;UniMod:999@Residue:K:x",
                }
            ),
            "supported_mixed_other",
        )
        self.assertEqual(
            stage_b.classify_ptm({"modified": "NO", "modification_labels": ""}),
            "unmodified",
        )

    def test_quota_sum_and_rare_stratum(self) -> None:
        counts = {
            stage_b.Stratum(2, "02_8-12", "unmodified"): 1000,
            stage_b.Stratum(3, "03_13-18", "oxidation"): 200,
            stage_b.Stratum(6, "06_36+", "acetyl"): 1,
        }
        quotas = stage_b.allocate_quotas(counts, target_size=100, minimum_per_stratum=1)
        self.assertEqual(sum(quotas.values()), 100)
        self.assertEqual(quotas[stage_b.Stratum(6, "06_36+", "acetyl")], 1)

    def test_end_to_end_is_deterministic_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = root / "identity_inventory.tsv"
            stage_b.write_synthetic_inventory(inventory)
            out_a = root / "a"
            out_b = root / "b"
            a = stage_b.run_manifest(inventory, out_a, target_size=40, minimum_per_stratum=1)
            b = stage_b.run_manifest(inventory, out_b, target_size=40, minimum_per_stratum=1)
            self.assertEqual(a["manifest_fingerprint"], b["manifest_fingerprint"])
            self.assertEqual(a["selected_identities"], "40")

            # Input row order must not affect the selected cohort.
            fieldnames, synthetic = stage_b.synthetic_rows()
            reversed_inventory = root / "identity_inventory_reversed.tsv"
            stage_b.write_tsv(reversed_inventory, fieldnames, reversed(synthetic))
            out_c = root / "c"
            c = stage_b.run_manifest(
                reversed_inventory, out_c, target_size=40, minimum_per_stratum=1
            )
            self.assertEqual(a["manifest_fingerprint"], c["manifest_fingerprint"])

            with (out_a / "stage_b_manifest.tsv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertTrue(any(row["charge"] == "6" for row in rows))
            self.assertTrue(
                all(row["existing_attachment_template_candidate"] == "YES" for row in rows)
            )
            self.assertTrue(all(not row["failure_reasons"] for row in rows))

    def test_forbidden_label_column_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tsv"
            fieldnames, rows = stage_b.synthetic_rows()
            fieldnames = [*fieldnames, "target_ccs"]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerow({**rows[0], "target_ccs": "123.4"})
            with self.assertRaisesRegex(ValueError, "CCS/mobility"):
                stage_b.first_pass(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
