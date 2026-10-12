#!/usr/bin/env python3
"""Label-blind external-study evaluation for the qualified v0.52 peptide-panel backend.

No model imports or new training. Only the four supported panel columns are
passed to inference; independently measured properties stay in the source TSV.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

FIELDS = ('measurement_id', 'study_id', 'run_id', 'sequence', 'peptidoform_state', 'charge',
          'precursor_mz', 'nce', 'observed_rt', 'rt_coordinate',
          'observed_ccs', 'ccs_unit', 'observed_native_mobility', 'mobility_coordinate')
PANEL_FIELDS = ('sequence', 'charge', 'precursor_mz', 'nce')
PREDICT_FIELDS = ('input_index', 'sequence', 'charge', 'precursor_mz', 'nce',
                  'predicted_rt', 'predicted_native_mobility', 'predicted_ccs',
                  'ccs_factor', 'ms2_scope')
IONS = ('b_z1', 'b_z2', 'y_z1', 'y_z2')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path, required):
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'{path}: missing or duplicate header')
        absent = set(required) - set(reader.fieldnames)
        if absent:
            raise ValueError(f'{path}: missing columns {sorted(absent)}')
        result = list(reader)
        if not result or any(None in r or any(v is None for v in r.values()) for r in result):
            raise ValueError(f'{path}: no records or malformed TSV')
        return result


def number(value, field, positive=False):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'non-numeric {field}: {value!r}') from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0):
        raise ValueError(f'invalid {field}: {value!r}')
    return parsed


def optional_number(value, field, positive=False):
    return None if value == '' else number(value, field, positive)


def validate_observations(obs):
    if len(obs) > 2048:
        raise ValueError('v0520_peptide_panel supports at most 2048 precursors per invocation')
    seen = set()
    for i, r in enumerate(obs):
        ident = r['measurement_id']
        if not ident or '\t' in ident or ident in seen:
            raise ValueError(f'row {i}: missing or duplicate measurement_id')
        seen.add(ident)
        if not r['study_id'] or not r['run_id']:
            raise ValueError(f'row {i}: source study_id/run_id required')
        if r['peptidoform_state'] != 'unmodified':
            raise ValueError(f'row {i}: peptidoform_state must explicitly be unmodified')
        seq = r['sequence']
        if not 7 <= len(seq) <= 35 or any(aa not in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):
            raise ValueError(f'row {i}: only unmodified uppercase canonical peptides (7..35)')
        if r['charge'] not in tuple(str(c) for c in range(1, 9)):
            raise ValueError(f'row {i}: charge must be 1..8')
        mz = optional_number(r['precursor_mz'], 'precursor_mz', True)
        nce = optional_number(r['nce'], 'nce', True)
        if nce is not None and nce > 100:
            raise ValueError(f'row {i}: nce must be <=100')
        rt = optional_number(r['observed_rt'], 'observed_rt')
        ccs = optional_number(r['observed_ccs'], 'observed_ccs', True)
        mobility = optional_number(r['observed_native_mobility'], 'observed_native_mobility', True)
        if rt is not None and not r['rt_coordinate']:
            raise ValueError(f'row {i}: specify rt_coordinate (model_target, raw_seconds, raw_minutes, iRT, unknown)')
        if r['rt_coordinate'] and r['rt_coordinate'] not in ('model_target', 'raw_seconds', 'raw_minutes', 'iRT', 'unknown'):
            raise ValueError(f'row {i}: unsupported rt_coordinate')
        if ccs is not None and r['ccs_unit'] != 'angstrom2':
            raise ValueError(f'row {i}: CCS must be explicitly in angstrom2')
        if mobility is not None and r['mobility_coordinate'] not in ('model_native', 'inverse_k0', 'unknown'):
            raise ValueError(f'row {i}: unsupported mobility_coordinate')
        # 'inverse_k0' is NOT assumed equivalent to the checkpoint mobility coordinate.
        if mz is not None and mz <= 0:
            raise ValueError('unreachable m/z check')
    return obs


def write_tsv(path, fields, entries):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(entries)


def prepare(source, panel, manifest):
    obs = validate_observations(rows(source, FIELDS))
    if panel.exists() or manifest.exists():
        raise ValueError('refusing to overwrite existing inference panel or manifest')
    panel.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(panel, PANEL_FIELDS, ({f: r[f] for f in PANEL_FIELDS} for r in obs))
    manifest.write_text(json.dumps({
        'schema': 'redeem_v0520_external_study_v1',
        'measurements_sha256': digest(source),
        'inference_panel_sha256': digest(panel),
        'rows': len(obs),
        'study_ids': sorted({r['study_id'] for r in obs}),
        'training_identity_overlap': 'NOT_CHECKED',
        'target_labels_in_inference_panel': False,
    }, indent=2, sort_keys=True) + '\n')
    print('external_study_prepare=PASS')
    print(f'precursors={len(obs)}')
    print(f'panel_sha256={digest(panel)}')
    print('targets_in_panel=NO')


def error_metrics(pairs):
    if not pairs:
        return {'n': 0, 'mae': None, 'rmse': None, 'bias': None}
    ds = [p - t for t, p in pairs]
    return {'n': len(ds), 'mae': sum(map(abs, ds)) / len(ds),
            'rmse': math.sqrt(sum(d*d for d in ds) / len(ds)),
            'bias': sum(ds) / len(ds)}


def fragment_metrics(observations, predicted_path, observed_path):
    if observed_path is None:
        return {'measured_fragment_table': 'NOT_SUPPLIED', 'complete_spectra': 0,
                'incomplete_spectra': 0, 'mean_cosine': None}, {}
    preds = rows(predicted_path, ('input_index', 'sequence', 'charge', 'ion_type',
                                 'fragment_number', 'predicted_intensity'))
    measured = rows(observed_path, ('measurement_id', 'ion_type', 'fragment_number', 'observed_intensity'))
    by_pred = defaultdict(dict)
    by_obs = defaultdict(dict)
    ids = {r['measurement_id']: i for i, r in enumerate(observations)}
    for r in preds:
        i = int(r['input_index'])
        if i < 0 or i >= len(observations):
            raise ValueError(f'prediction fragments invalid input_index {i}')
        if r['sequence'] != observations[i]['sequence'] or r['charge'] != observations[i]['charge']:
            raise ValueError(f'fragment precursor identity mismatch at {i}')
        key = (r['ion_type'], r['fragment_number'])
        if key[0] not in IONS or not key[1].isdigit():
            raise ValueError(f'unsupported predicted fragment channel {key}')
        if key in by_pred[i]:
            raise ValueError(f'duplicate predicted fragment {i} {key}')
        val = number(r['predicted_intensity'], 'predicted_intensity')
        if val < 0:
            raise ValueError('predicted fragment intensities must be >=0')
        by_pred[i][key] = val
    for i, r in enumerate(observations):
        sequence = r['sequence']
        channels = (0, 2) if r['charge'] == '1' else (0, 1, 2, 3)
        expected = {
            (IONS[c], str(cleavage + 1 if c < 2 else len(sequence) - cleavage - 1))
            for cleavage in range(len(sequence) - 1) for c in channels
        }
        if set(by_pred[i]) != expected:
            raise ValueError(f'predicted fragment grid incomplete or invalid at index {i}')
    for r in measured:
        if r['measurement_id'] not in ids:
            raise ValueError(f'observed fragment unknown measurement_id {r["measurement_id"]}')
        i = ids[r['measurement_id']]
        key = (r['ion_type'], r['fragment_number'])
        if key in by_obs[i]:
            raise ValueError(f'duplicate measured fragment {i} {key}')
        value = number(r['observed_intensity'], 'observed_intensity')
        if value < 0:
            raise ValueError('observed fragment intensities must be >=0')
        by_obs[i][key] = value
    cos = {}
    incomplete = 0
    for i in range(len(observations)):
        pk = by_pred[i]
        mk = by_obs[i]
        if not pk or set(pk) != set(mk):
            incomplete += 1
            continue  # cannot treat absent annotations as measured zeros
        dot = sum(pk[k] * mk[k] for k in pk)
        aa = math.sqrt(sum(v*v for v in pk.values()))
        bb = math.sqrt(sum(v*v for v in mk.values()))
        if aa > 0 and bb > 0:
            cos[i] = dot / (aa*bb)
        else:
            incomplete += 1
    return {'measured_fragment_table': str(observed_path), 'complete_spectra': len(cos),
            'incomplete_spectra': incomplete,
            'mean_cosine': statistics.mean(cos.values()) if cos else None}, cos


def score(source, panel, manifest, predictions, output, predicted_fragments=None, measured_fragments=None):
    if output.exists():
        raise ValueError('refusing to overwrite existing evaluation report')
    meta = json.loads(manifest.read_text())
    if meta.get('schema') != 'redeem_v0520_external_study_v1' or meta.get('target_labels_in_inference_panel') is not False:
        raise ValueError('unknown manifest or labels were passed into model')
    if meta['measurements_sha256'] != digest(source) or meta['inference_panel_sha256'] != digest(panel):
        raise ValueError('measurement or inference panel SHA256 changed since preparation')
    obs = validate_observations(rows(source, FIELDS))
    given = rows(panel, PANEL_FIELDS)
    pred = rows(predictions, PREDICT_FIELDS)
    if len(obs) != len(given) or len(obs) != len(pred) or len(obs) != meta['rows']:
        raise ValueError('measurement/panel/prediction row count mismatch')
    pairs = defaultdict(list)
    per_run = defaultdict(lambda: defaultdict(list))
    unused = defaultdict(int)
    for i, (r, p, q) in enumerate(zip(obs, given, pred)):
        if any(r[f] != p[f] for f in PANEL_FIELDS):
            raise ValueError(f'inference panel changed input order/content at index {i}')
        if q['input_index'] != str(i) or q['sequence'] != r['sequence'] or q['charge'] != r['charge']:
            raise ValueError(f'prediction precursor mismatch at input_index {i}')
        if q['ms2_scope'] != 'theoretical_b_y_1_2_only':
            raise ValueError(f'unknown MS2 prediction scope at input_index {i}')
        if r['nce'] != '':
            if abs(number(q['nce'], 'predicted nce') - number(r['nce'], 'input nce')) > 0.0001:
                raise ValueError(f'NCE mismatch at index {i}')
        elif q['nce'] != '':
            raise ValueError(f'predicted NCE should be empty at index {i}')
        if r['precursor_mz'] != '':
            mz1 = number(q['precursor_mz'], 'predicted precursor_mz', True)
            mz0 = number(r['precursor_mz'], 'input precursor_mz', True)
            if abs(mz1 - mz0) > 1e-4 + 1e-6*mz0:
                raise ValueError(f'precursor m/z mismatch at index {i}')
        pvals = {k: number(q['predicted_' + k], 'predicted_' + k)
                 for k in ('rt', 'ccs', 'native_mobility')}
        if pvals['ccs'] <= 0 or pvals['native_mobility'] <= 0:
            raise ValueError(f'nonpositive native prediction at index {i}')
        group = (r['study_id'], r['run_id'])
        target_settings = (
            ('rt', 'observed_rt', 'rt_coordinate', 'model_target'),
            ('ccs', 'observed_ccs', 'ccs_unit', 'angstrom2'),
            ('native_mobility', 'observed_native_mobility', 'mobility_coordinate', 'model_native'),
        )
        for metric, field, unit_field, compatible in target_settings:
            if r[field] == '':
                unused[f'{metric}_missing'] += 1
            elif r[unit_field] != compatible:
                unused[f'{metric}_unit_not_comparable'] += 1
            else:
                target = number(r[field], field, metric != 'rt')
                pairs[metric].append((target, pvals[metric]))
                per_run[group][metric].append((target, pvals[metric]))
    frag = {'measured_fragment_table': 'NOT_SUPPLIED', 'complete_spectra': 0,
            'incomplete_spectra': 0, 'mean_cosine': None}
    if measured_fragments is not None:
        if predicted_fragments is None:
            raise ValueError('--predicted-fragments required with --measured-fragments')
        frag, _ = fragment_metrics(obs, predicted_fragments, measured_fragments)
    if not any(pairs.values()) and not frag['complete_spectra']:
        raise ValueError('no comparable measured targets; refusing an empty accuracy PASS')
    report = {
        'status': 'PASS',
        'meaning': 'evaluation completed; not an accuracy threshold or training-generalization claim',
        'schema': 'redeem_v0520_external_study_v1',
        'rows': len(obs),
        'studies': sorted({r['study_id'] for r in obs}),
        'training_identity_overlap': 'NOT_CHECKED',
        'sha256': {'measurements': digest(source), 'panel': digest(panel), 'predictions': digest(predictions),
                   'predicted_fragments': digest(predicted_fragments) if predicted_fragments else None,
                   'measured_fragments': digest(measured_fragments) if measured_fragments else None},
        'overall': {k: error_metrics(pairs[k]) for k in ('rt', 'ccs', 'native_mobility')},
        'per_run': [{'study_id': k[0], 'run_id': k[1],
                     **{m: error_metrics(v[m]) for m in ('rt', 'ccs', 'native_mobility')}}
                    for k, v in sorted(per_run.items())],
        'not_scored': dict(sorted(unused.items())),
        'ms2': frag,
        'caveats': ['RT is only scored for explicitly supplied model_target coordinates.',
                    'CCS is only scored when angstrom2 is declared.',
                    'Mobility is only scored for explicitly compatible model_native coordinates.',
                    'Fragment cosine requires exhaustive four-channel b/y grid with observed zeros supplied.',
                    'External-study provenance does not establish no peptide-identity overlap with training.'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('v0520_external_study_evaluation=PASS')
    print(f'precursors={len(obs)}')
    for key in ('rt', 'ccs', 'native_mobility'):
        m = report['overall'][key]
        print(f'{key}_n={m["n"]} {key}_mae={m["mae"]} {key}_rmse={m["rmse"]}')
    print(f'ms2_complete_spectra={frag["complete_spectra"]} ms2_cosine={frag["mean_cosine"]}')
    print(f'metrics_json={output}')


def self_test():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        src = root / 'measurements.tsv'
        header = list(FIELDS)
        ex = [dict(zip(FIELDS, ('m1', 'study1', 'run1', 'PEPTIDEK', 'unmodified', '2', '', '27',
                                '20.0', 'model_target', '300', 'angstrom2',
                                '0.8', 'model_native'))),
              dict(zip(FIELDS, ('m2', 'study1', 'run1', 'ACDEFGHIK', 'unmodified', '3', '', '30',
                                '1800', 'raw_seconds', '400', 'angstrom2',
                                '0.9', 'inverse_k0')))]
        write_tsv(src, header, ex)
        panel = root/'panel.tsv'
        manifest = root/'panel.manifest.json'
        prepare(src, panel, manifest)
        pred = root/'pred.tsv'
        write_tsv(pred, PREDICT_FIELDS, [
            dict(zip(PREDICT_FIELDS, (str(i), r['sequence'], r['charge'], str(400+i),
                                       r['nce'], str(21+i), str(0.81+i*.1),
                                       str(301+i*100), '400', 'theoretical_b_y_1_2_only')))
            for i, r in enumerate(ex)])
        out = root/'report.json'
        score(src, panel, manifest, pred, out)
        metrics = json.loads(out.read_text())
        assert metrics['overall']['rt']['n'] == 1
        assert metrics['overall']['rt']['mae'] == 1.0
        assert metrics['overall']['ccs']['n'] == 2
        assert metrics['overall']['ccs']['mae'] == 1.0
        assert metrics['overall']['native_mobility']['n'] == 1
        assert metrics['not_scored']['rt_unit_not_comparable'] == 1
        pred_frags = root / 'pred_frag.tsv'
        obs_frags = root / 'obs_frag.tsv'
        fragments = []
        measured = []
        for i, row in enumerate(ex):
            seq = row['sequence']
            for cleavage in range(len(seq) - 1):
                for channel, ion in enumerate(IONS):
                    number_ion = cleavage + 1 if channel < 2 else len(seq) - cleavage - 1
                    fragments.append(dict(input_index=str(i), sequence=seq,
                                          charge=row['charge'], ion_type=ion,
                                          fragment_number=str(number_ion),
                                          predicted_intensity='1'))
                    measured.append(dict(measurement_id=row['measurement_id'], ion_type=ion,
                                         fragment_number=str(number_ion), observed_intensity='1'))
        write_tsv(pred_frags, ('input_index', 'sequence', 'charge', 'ion_type',
                               'fragment_number', 'predicted_intensity'), fragments)
        write_tsv(obs_frags, ('measurement_id', 'ion_type', 'fragment_number',
                              'observed_intensity'), measured)
        score(src, panel, manifest, pred, root/'with_fragments.json', pred_frags, obs_frags)
        fm = json.loads((root/'with_fragments.json').read_text())
        assert fm['ms2']['complete_spectra'] == 2
        assert abs(fm['ms2']['mean_cosine'] - 1) < 1e-12
        write_tsv(obs_frags, ('measurement_id', 'ion_type', 'fragment_number',
                              'observed_intensity'), measured[1:])
        score(src, panel, manifest, pred, root/'missing_fragment.json', pred_frags, obs_frags)
        fm = json.loads((root/'missing_fragment.json').read_text())
        assert fm['ms2']['complete_spectra'] == 1 and fm['ms2']['incomplete_spectra'] == 1
        ex[0]['sequence'] = 'CHANGED'
        write_tsv(src, header, ex)
        try:
            score(src, panel, manifest, pred, root/'changed_report.json')
        except ValueError as exc:
            assert 'SHA256 changed' in str(exc)
        else:
            raise AssertionError('modified measured data was accepted')
        print('v0520_external_study_evaluator_self_test=PASS')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--measurements', type=Path, required=True)
    prep.add_argument('--panel', type=Path, required=True)
    prep.add_argument('--manifest', type=Path, required=True)
    evaluation = sub.add_parser('score')
    evaluation.add_argument('--measurements', type=Path, required=True)
    evaluation.add_argument('--panel', type=Path, required=True)
    evaluation.add_argument('--manifest', type=Path, required=True)
    evaluation.add_argument('--predictions', type=Path, required=True)
    evaluation.add_argument('--predicted-fragments', type=Path)
    evaluation.add_argument('--measured-fragments', type=Path)
    evaluation.add_argument('--report', type=Path, required=True)
    sub.add_parser('self-test')
    args = p.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args.measurements, args.panel, args.manifest)
        elif args.command == 'score':
            score(args.measurements, args.panel, args.manifest, args.predictions,
                  args.report, args.predicted_fragments, args.measured_fragments)
        else:
            self_test()
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
