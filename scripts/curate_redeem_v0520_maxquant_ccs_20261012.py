#!/usr/bin/env python3
"""Extract real, strictly unmodified CCS observations from MaxQuant evidence.txt.

No synthetic observations and no predicted labels. Requires source-level run IDs,
measured CCS, explicit Å² attestation, and identification confidence metadata.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

OUTPUT_FIELDS = (
    'measurement_id', 'study_id', 'run_id', 'sequence', 'peptidoform_state',
    'charge', 'precursor_mz', 'nce', 'observed_rt', 'rt_coordinate',
    'observed_ccs', 'ccs_unit', 'observed_native_mobility', 'mobility_coordinate',
)
SOURCE_FIELDS = ('Raw file', 'Sequence', 'Modified sequence', 'Modifications',
                 'Charge', 'CCS', 'PEP', 'Reverse', 'Potential contaminant')
SEQ_PATTERN = re.compile(r'^_([ACDEFGHIKLMNPQRSTVWY]{7,35})_$')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def finite_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def curate(source, output, study, max_rows, min_rows, ccs_unit, provenance, sample_unlabelled):
    if not study.strip() or study.upper() in ('UNKNOWN', 'EXAMPLE', 'PXD...'):
        raise ValueError('real study accession/identifier is required')
    if ccs_unit != 'angstrom2':
        raise ValueError('explicit --confirm-ccs-unit angstrom2 required')
    if not sample_unlabelled:
        raise ValueError('confirm no non-cysteine fixed PTMs or isotopic labels in the source sample')
    if max_rows < 1 or max_rows > 2048 or not (1 <= min_rows <= max_rows):
        raise ValueError('require 1 <= min_rows <= max_rows <= 2048')
    if output.exists() or Path(str(output) + '.provenance.json').exists():
        raise ValueError('output already exists; refusing overwrite')

    rejected = Counter()
    best = {}
    total = 0
    with open(source, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('missing or duplicated source TSV header')
        missing = set(SOURCE_FIELDS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f'missing required MaxQuant evidence columns: {sorted(missing)}')
        for index, r in enumerate(reader, start=2):
            total += 1
            if None in r or any(v is None for v in r.values()):
                rejected['malformed_row'] += 1
                continue
            if r['Reverse'].strip() == '+' or r['Potential contaminant'].strip() == '+':
                rejected['decoy_or_contaminant'] += 1
                continue
            seq_match = SEQ_PATTERN.fullmatch(r['Modified sequence'].strip())
            if not seq_match or r['Modifications'].strip().lower() != 'unmodified':
                rejected['modified_or_not_explicitly_unmodified'] += 1
                continue
            seq = seq_match.group(1)
            if seq != r['Sequence'].strip() or 'C' in seq:
                # MaxQuant can hide fixed carbamidomethyl-C in the modification string.
                rejected['sequence_mismatch_or_implicit_fixed_cysteine_ptm'] += 1
                continue
            if r['Charge'].strip() not in [str(i) for i in range(1, 9)]:
                rejected['charge_unsupported'] += 1
                continue
            run = r['Raw file'].strip()
            if not run or any(c in run for c in ('\t', '\n', '\r')):
                rejected['run_id_missing'] += 1
                continue
            pep = finite_float(r['PEP'])
            if pep is None or pep < 0 or pep > 0.01:
                rejected['pep_missing_or_gt_0.01'] += 1
                continue
            ccs = finite_float(r['CCS'])
            if ccs is None or not (100 < ccs < 2000):
                rejected['ccs_missing_or_out_of_range'] += 1
                continue
            key = (run, seq, r['Charge'].strip())
            candidate = (pep, index, ccs)
            if key not in best or candidate < best[key]:
                if key in best:
                    rejected['duplicate_precursor'] += 1
                best[key] = candidate
            else:
                rejected['duplicate_precursor'] += 1

    picked = sorted(best.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2])))[:max_rows]
    if len(picked) < min_rows:
        raise ValueError(f'only {len(picked)} eligible observed CCS precursors, require {min_rows}; '
                         'inspect provenance/source columns or set --min-rows deliberately')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=OUTPUT_FIELDS, lineterminator='\n')
        writer.writeheader()
        for i, ((run, seq, charge), (_, src_line, ccs)) in enumerate(picked):
            writer.writerow({
                'measurement_id': f'{study}:evidence_line_{src_line}',
                'study_id': study, 'run_id': run, 'sequence': seq,
                'peptidoform_state': 'unmodified', 'charge': charge,
                'precursor_mz': '', 'nce': '', 'observed_rt': '', 'rt_coordinate': '',
                'observed_ccs': format(ccs, '.10g'), 'ccs_unit': 'angstrom2',
                'observed_native_mobility': '', 'mobility_coordinate': '',
            })
    meta = {
        'status': 'CURATED_NOT_INDEPENDENCE_VERIFIED',
        'source': str(source.resolve()), 'source_sha256': sha256(source),
        'source_study_id': study, 'source_url_or_doi': provenance,
        'measurement_sha256': sha256(output), 'measured_ccs_unit_attested': 'angstrom2',
        'measurement_rows': len(picked), 'source_rows': total,
        'source_runs': sorted({k[0] for k, _ in picked}),
        'exclusions': dict(rejected),
        'training_identity_overlap': 'NOT_CHECKED',
        'training_study_overlap': 'NOT_CHECKED',
        'selection': 'lowest PEP <=0.01 per raw-file/sequence/charge; alphabetical first N keys',
        'non_ccs_target_status': 'NOT_SCORED',
        'precursor_mz_and_nce': 'not supplied; inferred/defaulted by the existing panel backend',
        'fixed_cysteine_modification': 'all C-containing peptides conservatively excluded',
        'other_fixed_modifications_and_isotope_labels': 'user_attested_absent',
    }
    meta_path = Path(str(output) + '.provenance.json')
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    print('maxquant_ccs_curation=PASS')
    print(f'measured_precursors={len(picked)} source_rows={total}')
    print(f'measurements_sha256={meta["measurement_sha256"]}')
    print('training_overlap=NOT_CHECKED')
    print(f'output={output}')
    print(f'provenance={meta_path}')


def self_test():
    with TemporaryDirectory() as d:
        d = Path(d)
        source, dest = d / 'evidence.txt', d / 'measured.tsv'
        data = [
            ['run001', 'PEPTIDEK', '_PEPTIDEK_', 'Unmodified', '2', '329.0', '0.001', '', ''],
            ['run001', 'PEPTIDEK', '_PEPTIDEK_', 'Unmodified', '2', '340.0', '0.008', '', ''],
            ['run002', 'ACDEFGHIK', '_ACDEFGHIK_', 'Unmodified', '3', '435', '0.001', '', ''],
            ['run001', 'MPEPTIDEK', '_M(ox)PEPTIDEK_', 'Oxidation (M)', '2', '402', '0.001', '', ''],
            ['run002', 'PEPTIDES', '_PEPTIDES_', 'Unmodified', '2', '380', '0.001', '+', ''],
            ['run002', 'MPEPTIDEK', '_MPEPTIDEK_', 'Unmodified', '2', '412.2', '0.02', '', ''],
            ['run002', 'PEPTIDER', '_PEPTIDER_', 'Unmodified', '2', '390.7', '0.001', '', ''],
        ]
        with open(source, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(SOURCE_FIELDS)
            writer.writerows(data)
        curate(source, dest, 'PXD_TEST', 4, 2, 'angstrom2', 'source-test', True)
        with open(dest, newline='') as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        assert len(rows) == 2 and rows[0]['observed_ccs'] == '329'
        assert all('C' not in r['sequence'] and r['nce'] == '' and r['observed_rt'] == '' for r in rows)
        assert {r['run_id'] for r in rows} == {'run001', 'run002'}
        assert json.loads(Path(str(dest) + '.provenance.json').read_text())['training_identity_overlap'] == 'NOT_CHECKED'
        try:
            curate(source, dest, 'PXD_TEST', 4, 2, 'angstrom2', 'source-test', True)
        except ValueError as e:
            assert 'already exists' in str(e)
        else:
            raise AssertionError('overwrite accepted')
    print('maxquant_ccs_curator_self_test=PASS')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--evidence', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--study-id')
    ap.add_argument('--source-url-or-doi', default='NOT_PROVIDED')
    ap.add_argument('--confirm-ccs-unit', choices=('angstrom2',))
    ap.add_argument('--confirm-no-other-fixed-ptms-or-labels', action='store_true')
    ap.add_argument('--max-rows', type=int, default=256)
    ap.add_argument('--min-rows', type=int, default=100)
    args = ap.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if not args.evidence or not args.output or not args.study_id:
                raise ValueError('--evidence, --output and --study-id required')
            curate(args.evidence, args.output, args.study_id, args.max_rows,
                   args.min_rows, args.confirm_ccs_unit, args.source_url_or_doi,
                   args.confirm_no_other_fixed_ptms_or_labels)
    except (ValueError, OSError, KeyError) as exc:
        ap.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
