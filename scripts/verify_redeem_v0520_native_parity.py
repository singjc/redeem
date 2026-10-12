#!/usr/bin/env python3
"""Paired numerical check for original v0.52 export adapter vs native backend.

Usage: python3 verify_redeem_v0520_native_parity.py LEGACY_REFERENCE_PREDICTIONS.tsv NATIVE_PREDICTIONS.tsv
Both files must have corresponding `.annotated_fragments.tsv` files.
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ABS_TOLERANCE = 1.0e-4
REL_TOLERANCE = 2.0e-6
FRAGMENT_IDENTITY = (
    'record_index', 'sequence', 'cleavage_index', 'channel',
    'charged_frag_type', 'fragment_number', 'product_mz',
)



def load(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty TSV: {path}")
    return rows


def check_float(expected, actual, label):
    if not math.isclose(float(expected), float(actual), rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE):
        raise ValueError(f"{label}: reference={expected} native={actual}")


def group_fragments(rows):
    # The historical exporter orders selected MS2 precursors by a seeded hash;
    # the native backend writes corpus-index order. Compare identities, not rows.
    grouped = defaultdict(list)
    for row in rows:
        identity = tuple(row[field] for field in FRAGMENT_IDENTITY)
        grouped[identity].append((
            float(row['target_intensity']), float(row['foundation_intensity'])
        ))
    return grouped


def compare(ref_file, native_file, expected_rt=793, expected_ms2=404):
    ref = load(ref_file)
    native = load(native_file)
    if len(ref) != expected_rt or len(native) != len(ref):
        raise ValueError(f"RT/precursor counts differ or not frozen {expected_rt}: {len(ref)}, {len(native)}")
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
    if precursor_count != expected_ms2:
        raise ValueError(f"unexpected frozen MS2 precursor count: {precursor_count}")
    ref_groups = group_fragments(ref_ms2)
    native_groups = group_fragments(native_ms2)
    if ref_groups.keys() != native_groups.keys():
        missing = ref_groups.keys() - native_groups.keys()
        extra = native_groups.keys() - ref_groups.keys()
        raise ValueError(f"MS2 fragment identities differ: missing={len(missing)} "
                         f"extra={len(extra)} sample_missing={next(iter(missing), None)} "
                         f"sample_extra={next(iter(extra), None)}")
    for identity in sorted(ref_groups):
        expected = sorted(ref_groups[identity])
        actual = sorted(native_groups[identity])
        if len(expected) != len(actual):
            raise ValueError(f"MS2 fragment multiplicity differs for {identity}: "
                             f"reference={len(expected)} native={len(actual)}")
        for i, ((target_a, pred_a), (target_b, pred_b)) in enumerate(zip(expected, actual)):
            check_float(target_a, target_b, f"MS2 target {identity} duplicate {i}")
            check_float(pred_a, pred_b, f"MS2 prediction {identity} duplicate {i}")
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
        precursor_header = ('record_index', 'corpus_record_index', 'sequence',
                            'charge', 'ms2_annotation_scope', 'predicted_rt')
        precursors = [
            ('0', '13428', 'PEPTIDEK', '2', 'annotated_fragments_only', '10.0'),
            ('1', '30699', 'OTHERTIDEK', '3', 'annotated_fragments_only', '20.0'),
        ]
        fragment_header = (*FRAGMENT_IDENTITY, 'target_intensity', 'foundation_intensity')
        fragments = [
            ('30699', 'OTHERTIDEK', '0', '0', 'b_z1', '1', '100.00000000', '0.4', '0.5'),
            ('13428', 'PEPTIDEK', '1', '2', 'y_z1', '6', '200.00000000', '0.2', '0.3'),
            ('30699', 'OTHERTIDEK', '2', '0', 'b_z1', '3', '300.00000000', '0.6', '0.7'),
            # Repeated annotated coordinate must compare as a multiset, not by order.
            ('30699', 'OTHERTIDEK', '0', '0', 'b_z1', '1', '100.00000000', '0.3', '0.35'),
        ]
        def write_tsv(path, header, rows):
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(header)
                writer.writerows(rows)
        write_tsv(a, precursor_header, precursors)
        write_tsv(b, precursor_header, precursors)
        fa = Path(str(a) + '.annotated_fragments.tsv')
        fb = Path(str(b) + '.annotated_fragments.tsv')
        write_tsv(fa, fragment_header, fragments)
        write_tsv(fb, fragment_header, list(reversed(fragments)))
        compare(a, b, expected_rt=2, expected_ms2=2)

        # Fail closed on an altered fragment prediction, missing row, or identity.
        for corrupted in (
            [*fragments[:2], (*fragments[2][:-1], '0.9'), *fragments[3:]],
            list(reversed(fragments))[:-1],
            [*fragments[:2], ('99999', *fragments[2][1:]), *fragments[3:]],
        ):
            write_tsv(fb, fragment_header, corrupted)
            try:
                compare(a, b, expected_rt=2, expected_ms2=2)
            except ValueError:
                pass
            else:
                raise AssertionError('corrupt MS2 rows were accepted')
    print('v0520_native_comparator_self_test=PASS')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--self-test':
        self_test()
    elif len(sys.argv) == 3:
        compare(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        raise SystemExit('usage: verify_redeem_v0520_native_parity.py --self-test | OLD.tsv NEW.tsv')
