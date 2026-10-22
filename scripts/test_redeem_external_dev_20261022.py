"""Offline contract checks; tests use synthetic source records only."""
import importlib.util
import json
import pathlib
import tempfile
import unittest

P = pathlib.Path(__file__).parent / 'redeem_external_dev_benchmark_20261022.py'
spec = importlib.util.spec_from_file_location('external', P)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def rec(seq, index, charge=2, modified=None, n=20):
    return {'experiment_name':'run1','evidence_index':index,'scan_number':index+300,
            'sequence':seq,'modified_sequence':seq if modified is None else modified,
            'precursor_mz':550.0,'precursor_charge':charge,
            'mz_array':list(range(100,100+n)),'intensity_array':list(range(1,n+1))}


class ExternalContract(unittest.TestCase):
    def test_provenance_id_only_from_spectrum_identity(self):
        a=rec('PEPTIDE',2); b=rec('PEPTIDK',2)
        self.assertEqual(e.make_uid(a),e.make_uid(b))
        self.assertNotIn('PEPTIDE',e.make_uid(a))

    def test_modified_exclusion(self):
        self.assertEqual(e.parse_spectrum(rec('PEPTIDE',1,modified='PEPTID[+16]E'))[1], 'modified_unqualified')

    def test_ambiguous_cysteine_exclusion(self):
        self.assertEqual(e.parse_spectrum(rec('PEPTICER',1))[1], 'cysteine_modification_uncertain')

    def test_charge_scope(self):
        self.assertEqual(e.parse_spectrum(rec('PEPTIDE',1,charge=4))[1], 'charge_outside_declared_z2_z3_scope')

    def test_broken_peak_arrays_rejected(self):
        r=rec('PEPTIDE',1);r['intensity_array'].pop()
        self.assertEqual(e.parse_spectrum(r)[1], 'peak_array_mismatch')

    def test_deterministic_selection_independent_order(self):
        rows=[rec(s,i) for i,s in enumerate(['PEPTIDE','PEPTIDK','PEPTIDR','PEPTIDA','PEPTIDN','PEPTIDQ'])]
        x,_=e.select_records(rows,3,123)
        y,_=e.select_records(list(reversed(rows)),3,123)
        self.assertEqual([r['query_id'] for r in x],[r['query_id'] for r in y])

    def test_unique_peptide_charge(self):
        rows=[rec('PEPTIDE',1),rec('PEPTIDE',2),rec('PEPTIDE',3,charge=3),rec('PEPTIDK',4)]
        chosen,_=e.select_records(rows,3,123)
        self.assertEqual(3,len({(r['sequence'],r['charge']) for r in chosen}))

    def test_input_truncation_refused(self):
        with self.assertRaises(ValueError):
            e.select_records([rec('PEPTIDE',1)],2,123)

    def test_preparation_existing_path_refused(self):
        with tempfile.TemporaryDirectory() as d:
            src=pathlib.Path(d)/'source.jsonl'
            src.write_text(json.dumps(rec('PEPTIDE',1))+'\n')
            args=type('A',(),{'source':'jsonl','input_jsonl':str(src),'output_dir':str(pathlib.Path(d)/'prepared'),
                              'limit':1,'seed':10,'scan_cap':0})()
            e.prepare(args)
            with self.assertRaises(ValueError):e.prepare(args)

if __name__=='__main__':unittest.main(verbosity=2)
