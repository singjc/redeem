#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import structure_signal_stage_b_full_v1 as full
import structure_signal_stage_b_probe_v1 as probe


class StageBFullTests(unittest.TestCase):
    def make_manifest(self, root: Path) -> tuple[Path, str]:
        path = root / "manifest.tsv"
        fingerprint = full.tiny_manifest(path)
        return path, fingerprint

    def test_round_robin_shards_are_disjoint_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self.make_manifest(Path(tmp))
            _, rows, _ = probe.load_manifest(manifest)
            assigned: list[str] = []
            for shard_index in range(3):
                assigned.extend(
                    row["identity_key"]
                    for row in full.shard_rows(rows, shard_index, 3)
                )
            self.assertEqual(len(assigned), len(set(assigned)))
            self.assertEqual(
                sorted(assigned), sorted(row["identity_key"] for row in rows)
            )

    def test_etkdg_only_generation_is_reproducible(self) -> None:
        sequence = "PEPTIDERK"
        row = {
            "identity_key": sequence + "|z3",
            "peptidoform": sequence,
            "sequence": sequence,
            "theoretical_neutral_mass_da": f"{probe.rdMolDescriptors.CalcExactMolWt(probe.Chem.MolFromSequence(sequence)):.8f}",
        }
        built = probe.build_neutral_molecule(row, 0.002)
        candidates = probe.candidate_charge_sites(built.mol, set())
        sites = probe.microstate_sites(candidates, 3, 0)
        charged = probe.apply_positive_charge(built.mol, sites, 3)
        a, ids_a, mode_a, _ = full.embed_etkdg_only(charged, sequence, 0, 1)
        b, ids_b, mode_b, _ = full.embed_etkdg_only(charged, sequence, 0, 1)
        self.assertTrue(ids_a)
        self.assertTrue(ids_b)
        self.assertEqual(mode_a, mode_b)
        self.assertLessEqual(probe.max_coordinate_difference(a, b, 0), 1e-7)

    def test_finalize_requires_every_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, fingerprint = self.make_manifest(root)
            shards = root / "shards"
            shards.mkdir()
            full.run_shard(
                manifest,
                fingerprint,
                shards / "shard_000_of_002",
                0,
                2,
                7,
                1,
                1,
                0.002,
            )
            with self.assertRaisesRegex(ValueError, "missing shard output"):
                full.run_finalize(
                    manifest,
                    fingerprint,
                    shards,
                    root / "final",
                    2,
                    7,
                )

    def test_manifest_with_ccs_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _ = self.make_manifest(root)
            fields, rows = full.read_tsv(manifest)
            bad = root / "bad.tsv"
            full.write_tsv(bad, fields + ["measured_ccs"], rows)
            with self.assertRaisesRegex(ValueError, "CCS/mobility"):
                probe.load_manifest(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
