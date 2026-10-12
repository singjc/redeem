#!/usr/bin/env python3
"""Paired numerical check for original v0.52 export adapter vs native backend.

Usage: python3 verify_redeem_v0520_native_parity.py LEGACY_REFERENCE_PREDICTIONS.tsv NATIVE_PREDICTIONS.tsv
Both files must have corresponding `.annotated_fragments.tsv` files.
"""
import csv
import math
import sys
from collections import Counter
from pathlib import Path

ABS_TOLERANCE = 1.0e-4
REL_TOLERANCE = 2.0e-6


def load(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty TSV: {path}")
    return rows


def check_float(expected, actual, label):
    if not math.isclose(float(expected), float(actual), rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE):
        raise ValueError(f"{label}: reference={expected} native={actual}")


def compare(ref_file, native_file):
    ref = load(ref_file)
    native = load(native_file)
    if len(ref) != 793 or len(native) != len(ref):
        raise ValueError(f"RT/precursor counts differ or not frozen 793: {len(ref)}, {len(native)}")
    for i, (a, b) in enumerate(zip(ref, native)):
        for field in ('record_index', 'corpus_record_index', 'sequence', 'charge', 'ms2_annotation_scope'):
            if a.get(field) != b.get(field):
                raise ValueError(f"precursor row {i}: {field} {a.get(field)!r} != {b.get(field)!r}")
        check_float(a['predicted_rt'], b['predicted_rt'], f"RT row {i}")
    ref_ms2 = load(Path(str(ref_file) + '.annotated_fragments.tsv'))
    native_ms2 = load(Path(str(native_file) + '.annotated_fragments.tsv'))
    if len(ref_ms2) != len(native_ms2):
        raise ValueError(f"fragment counts differ: {len(ref_ms2)}, {len(native_ms2)}")
    precursor_count = len({row['record_index'] for row in ref_ms2})
    if precursor_count != 404:
        raise ValueError(f"unexpected frozen MS2 precursor count: {precursor_count}")
    for i, (a, b) in enumerate(zip(ref_ms2, native_ms2)):
        for field in ('record_index', 'sequence', 'cleavage_index', 'channel',
                      'charged_frag_type', 'fragment_number', 'product_mz'):
            if a[field] != b[field]:
                raise ValueError(f"MS2 row {i}: {field} {a[field]!r} != {b[field]!r}")
        check_float(a['target_intensity'], b['target_intensity'], f"MS2 target row {i}")
        check_float(a['foundation_intensity'], b['foundation_intensity'], f"MS2 prediction row {i}")
    print('v0520_native_common_validation_parity=PASS')
    print(f'rt_records={len(ref)}')
    print(f'ms2_precursors={precursor_count}')
    print(f'annotated_fragment_rows={len(ref_ms2)}')
    print(f'absolute_tolerance={ABS_TOLERANCE}')


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / 'a.tsv'
        b = Path(td) / 'b.tsv'
        a.write_text('record_index\tcorpus_record_index\tsequence\tcharge\tms2_annotation_scope\tpredicted_rt\n0\t1\tPEPTIDE\t2\tnone\t1.0\n')
        b.write_text(a.read_text().replace('1.0', '1.0000001'))
        assert load(a)[0]['sequence'] == 'PEPTIDE'
        check_float('1.0', '1.0000001', 'self-test')
        try:
            check_float('1.0', '1.1', 'self-test')
        except ValueError:
            pass
        else:
            raise AssertionError('mismatch was not detected')
    print('v0520_native_comparator_self_test=PASS')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--self-test':
        self_test()
    elif len(sys.argv) == 3:
        compare(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        raise SystemExit('usage: verify_redeem_v0520_native_parity.py --self-test | OLD.tsv NEW.tsv')
