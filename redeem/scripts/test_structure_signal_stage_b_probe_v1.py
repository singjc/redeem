#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("structure_signal_stage_b_probe_v1.py")
spec = importlib.util.spec_from_file_location("stage_b_probe", SCRIPT)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


class StageBProbeTests(unittest.TestCase):
    def test_common_ptm_mass_semantics(self) -> None:
        cases = [
            ("ACAAA|R1:UniMod:4", "ACAAA", 57.021463716),
            ("AMAAA|R1:UniMod:35", "AMAAA", 15.994914620),
            ("ANAAA|R1:UniMod:7", "ANAAA", 0.984015588),
            ("ASAAA|R1:UniMod:21", "ASAAA", 79.966330522),
            ("AKAAA|R1:UniMod:1", "AKAAA", 42.010564684),
            ("AAAAA|N-term:UniMod:1", "AAAAA", 42.010564684),
        ]
        for peptidoform, sequence, delta in cases:
            base = probe.Chem.MolFromSequence(sequence)
            expected = probe.rdMolDescriptors.CalcExactMolWt(base) + delta
            row = {
                "identity_key": f"{peptidoform}|z2",
                "peptidoform": peptidoform,
                "sequence": sequence,
                "theoretical_neutral_mass_da": f"{expected:.8f}",
            }
            built = probe.build_neutral_molecule(row, 0.002)
            self.assertLess(abs(built.mass_error_da), 1e-6)

    def test_unsupported_modification_fails_closed(self) -> None:
        sequence = "AKAAA"
        base = probe.Chem.MolFromSequence(sequence)
        row = {
            "identity_key": "AKAAA|R1:UniMod:999999|z2",
            "peptidoform": "AKAAA|R1:UniMod:999999",
            "sequence": sequence,
            "theoretical_neutral_mass_da": f"{probe.rdMolDescriptors.CalcExactMolWt(base):.8f}",
        }
        with self.assertRaisesRegex(ValueError, "does not implement"):
            probe.build_neutral_molecule(row, 0.002)

    def test_charge_microstates_have_exact_formal_charge(self) -> None:
        mol = probe.Chem.MolFromSequence("PEPTIDERK")
        sites = probe.candidate_charge_sites(mol, set())
        primary = probe.microstate_sites(sites, 3, 0)
        alternate = probe.microstate_sites(sites, 3, 1)
        self.assertNotEqual([site.atom_idx for site in primary], [site.atom_idx for site in alternate])
        for selected in (primary, alternate):
            charged = probe.apply_positive_charge(mol, selected, 3)
            self.assertEqual(sum(atom.GetFormalCharge() for atom in charged.GetAtoms()), 3)

    def test_selection_is_input_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.tsv"
            probe.synthetic_manifest(manifest)
            _, rows, _ = probe.load_manifest(manifest)
            a = probe.select_probe_rows(rows, 21, 2, 2)
            b = probe.select_probe_rows(list(reversed(rows)), 21, 2, 2)
            self.assertEqual(
                sorted(entry.row["identity_key"] for entry in a),
                sorted(entry.row["identity_key"] for entry in b),
            )

    def test_manifest_with_ccs_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.tsv"
            probe.synthetic_manifest(manifest)
            lines = manifest.read_text().splitlines()
            lines[0] += "\tmeasured_ccs"
            lines[1:] = [line + "\t123.4" for line in lines[1:]]
            manifest.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "CCS/mobility"):
                probe.load_manifest(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
