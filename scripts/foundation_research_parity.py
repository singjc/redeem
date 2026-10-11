#!/usr/bin/env python3
"""Read-only v0.52 checkpoint inventory and paired Validation RT/MS2 comparison.

Uses original corpus record indices and the historical v0.52 exporter. No TEST access,
no model changes, no third-party Python dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def metadata_fields(path: Path) -> dict[str, str]:
    # Historical checkpoint identity fields are flat YAML scalars; no YAML dependency.
    fields = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'([a-z][a-z0-9_]*):\s*(.*?)\s*', line)
        if match:
            fields[match.group(1)] = match.group(2).strip('"\'')
    return fields


def inventory(args):
    rows = []
    skip = {'.git', 'target', 'node_modules', '.venv', 'cache', 'tmp'}
    for root_arg in args.roots:
        root = Path(root_arg).resolve(strict=True)
        for base, dirs, names in os.walk(root, followlinks=False):
            depth = len(Path(base).relative_to(root).parts)
            dirs[:] = sorted(d for d in dirs if d not in skip) if depth < args.max_depth else []
            if 'metadata.yaml' not in names or 'model.safetensors' not in names:
                continue
            directory = Path(base)
            try:
                m = metadata_fields(directory / 'metadata.yaml')
                version = m.get('version', '')
                if version not in ('350', '500', '510', '520'):
                    continue
                weights = directory / 'model.safetensors'
                is_v052 = version == '520'
                valid = (not is_v052 or (
                    m.get('objective') == 'v0520_mobility_aware_pair_representation'
                    and m.get('architecture') == 'deep_pair_mobility_aware_representation_v0520'
                    and m.get('smoke_mode', '').lower() == 'false'
                    and int(m.get('completed_updates', '0')) > 0
                ))
                rows.append(dict(
                    version=version, checkpoint=str(directory),
                    status=('candidate' if valid else 'ineligible'),
                    objective=m.get('objective', ''),
                    architecture=m.get('architecture', ''),
                    completed_epochs=m.get('completed_epochs', ''),
                    completed_updates=m.get('completed_updates', ''),
                    benchmark_fingerprint=m.get('benchmark_manifest_fingerprint', ''),
                    corpus_fingerprint=m.get('corpus_fingerprint', ''),
                    metadata_sha256=sha256(directory / 'metadata.yaml'),
                    model_bytes=weights.stat().st_size,
                    model_sha256=sha256(weights) if args.hash_models else 'NOT_HASHED',
                ))
            except (OSError, ValueError) as exc:
                raise ValueError(f'failed to inspect {directory}: {exc}') from exc
    rows.sort(key=lambda r: (r['version'], r['checkpoint']))
    output = Path(args.output)
    if output.exists():
        raise ValueError(f'output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = ['version', 'status', 'checkpoint', 'objective', 'architecture',
            'completed_epochs', 'completed_updates', 'benchmark_fingerprint',
            'corpus_fingerprint', 'metadata_sha256', 'model_bytes', 'model_sha256']
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'checkpoint_inventory={output}')
    print(f'checkpoint_candidates={sum(r["status"] == "candidate" for r in rows)}')
    print(f'v0520_candidates={sum(r["version"] == "520" and r["status"] == "candidate" for r in rows)}')
    print('train_holdout_consumed=NO')
    print('historical_test_consumed=NO')


def tsv(path: Path):
    with path.open(newline='') as f:
        yield from csv.DictReader((l for l in f if not l.startswith('#')), delimiter='\t')


def num(value, what):
    try:
        x = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'bad {what}: {value!r}') from None
    if not math.isfinite(x):
        raise ValueError(f'nonfinite {what}: {value!r}')
    return x


def metrics(target, prediction):
    if len(target) != len(prediction) or not target:
        raise ValueError('invalid MS2 vector dimensions')
    if any(not math.isfinite(v) for v in target + prediction):
        raise ValueError('nonfinite MS2 prediction')
    mse = sum((a-b)**2 for a,b in zip(target, prediction))/len(target)
    aa = sum(x*x for x in target)
    bb = sum(x*x for x in prediction)
    dot = sum(a*b for a,b in zip(target, prediction))
    cosine = dot / math.sqrt(aa*bb) if aa > 0 and bb > 0 else 0.0
    ta, tb = mean(target), mean(prediction)
    x = [a-ta for a in target]
    y = [b-tb for b in prediction]
    xx, yy = sum(a*a for a in x), sum(b*b for b in y)
    pearson = sum(a*b for a,b in zip(x,y))/math.sqrt(xx*yy) if xx > 0 and yy > 0 else None
    return cosine, mse, pearson


def compare(args):
    historical = metadata_fields(Path(args.legacy_checkpoint) / 'metadata.yaml')
    integrated = metadata_fields(Path(args.integrated_checkpoint) / 'metadata.yaml')
    if (historical.get('version') != '520'
            or historical.get('architecture') != 'deep_pair_mobility_aware_representation_v0520'
            or historical.get('smoke_mode', '').lower() != 'false'):
        raise ValueError('historical checkpoint is not a selected non-smoke v0.52 model')
    if integrated.get('schema') != 'redeem.foundation.model':
        raise ValueError('integrated checkpoint is not a production foundation model')
    for label, historical_field, integrated_field in (
        ('corpus', 'corpus_fingerprint', 'corpus_fingerprint'),
        ('benchmark', 'benchmark_manifest_fingerprint', 'benchmark_fingerprint'),
    ):
        left = historical.get(historical_field)
        right = integrated.get(integrated_field)
        if not left or not right or left != right:
            raise ValueError(f'{label} fingerprints differ or are missing: {left!r} != {right!r}')
    benchmark = list(tsv(Path(args.benchmark)))
    validation = [r for r in benchmark if r['partition'] == 'validation']
    if len(validation) < args.prefix:
        raise ValueError(f'only {len(validation)} Validation records; expected {args.prefix}')
    prefix = validation[:args.prefix]
    prefix_ids = {int(r['record_index']) for r in prefix}
    original_by_ordinal = {i: int(r['record_index']) for i,r in enumerate(prefix)}
    expected_sequences = {int(r['record_index']): r['sequence'] for r in prefix}

    current = {}
    for r in tsv(Path(args.production)):
        ordinal = int(r['record_index'])
        if ordinal not in original_by_ordinal:
            raise ValueError(f'production ordinal outside Validation prefix: {ordinal}')
        global_id = original_by_ordinal[ordinal]
        if global_id in current or r['sequence'] != expected_sequences[global_id]:
            raise ValueError(f'duplicate or sequence mismatch in production row {ordinal}')
        current[global_id] = r
    if len(current) != args.prefix:
        raise ValueError(f'production contains {len(current)} records, expected {args.prefix}')

    rt_values = defaultdict(list)
    for r in tsv(Path(args.legacy_precursors)):
        idx = int(r['record_index'])
        if idx not in prefix_ids or r['partition'] != 'VALIDATION':
            raise ValueError(f'legacy precursor outside Validation prefix: {idx}')
        if r['sequence'] != expected_sequences[idx]:
            raise ValueError(f'legacy sequence mismatch: {idx}')
        if r['selected_rt'] != 'YES':
            continue
        prod = current[idx]
        if not prod['predicted_rt']:
            continue
        target = num(r['target_rt'], 'target RT')
        old = num(r['foundation_rt'], 'legacy RT')
        new = num(prod['predicted_rt'], 'production RT')
        rt_values['legacy'].append(abs(target-old))
        rt_values['production'].append(abs(target-new))

    annotated = defaultdict(dict)
    for r in tsv(Path(args.legacy_fragments)):
        idx = int(r['record_index'])
        if idx not in prefix_ids:
            raise ValueError(f'legacy fragment outside Validation prefix: {idx}')
        key = (int(r['cleavage_index']), int(r['channel']))
        if key in annotated[idx]:
            raise ValueError(f'duplicate annotated fragment {idx}/{key}')
        annotated[idx][key] = (num(r['target_intensity'], 'target intensity'),
                                num(r['foundation_intensity'], 'legacy intensity'))

    per_record = []
    for idx in sorted(annotated):
        raw = current[idx]['predicted_ms2']
        if not raw:
            raise ValueError(f'missing production MS2 for record {idx}')
        spectrum = json.loads(raw)
        target, old, new = [], [], []
        for (cleavage, channel), (intensity, historical) in sorted(annotated[idx].items()):
            try:
                production = num(spectrum[cleavage][channel], 'production MS2')
            except (IndexError, TypeError):
                raise ValueError(f'production MS2 shape mismatch at {idx}/{cleavage}/{channel}') from None
            target.append(intensity)
            old.append(historical)
            new.append(production)
        lo, lmse, lp = metrics(target, old)
        no, nmse, np = metrics(target, new)
        per_record.append(dict(record_index=idx, annotated_fragments=len(target),
            legacy_cosine=lo, production_cosine=no,
            legacy_rmse=math.sqrt(lmse), production_rmse=math.sqrt(nmse),
            legacy_pearson='' if lp is None else lp,
            production_pearson='' if np is None else np))

    if not rt_values['legacy'] or not per_record:
        raise ValueError('zero paired RT records or MS2 spectra')
    outdir = Path(args.output)
    if outdir.exists():
        raise ValueError(f'output already exists: {outdir}')
    outdir.mkdir(parents=True)
    with (outdir / 'paired_ms2_records.tsv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_record[0]), delimiter='\t')
        w.writeheader(); w.writerows(per_record)
    summary = [
        ('rt_mae_common_targets', len(rt_values['legacy']), mean(rt_values['legacy']), mean(rt_values['production'])),
        ('ms2_cosine_same_fragments', len(per_record), mean(r['legacy_cosine'] for r in per_record), mean(r['production_cosine'] for r in per_record)),
        ('ms2_rmse_same_fragments', len(per_record), mean(r['legacy_rmse'] for r in per_record), mean(r['production_rmse'] for r in per_record)),
    ]
    for key in ('legacy_pearson', 'production_pearson'):
        if not any(r[key] != '' for r in per_record):
            raise ValueError('zero Pearson-eligible MS2 spectra')
    valid_p = [r for r in per_record if r['legacy_pearson'] != '' and r['production_pearson'] != '']
    if not valid_p:
        raise ValueError('no shared Pearson-eligible MS2 spectra')
    summary.append(('ms2_pearson_same_fragments', len(valid_p),
                    mean(r['legacy_pearson'] for r in valid_p),
                    mean(r['production_pearson'] for r in valid_p)))
    with (outdir / 'paired_summary.tsv').open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['metric', 'paired_records', 'legacy_v0520', 'integrated', 'integrated_minus_legacy'])
        for name, n, old, new in summary:
            w.writerow([name, n, f'{old:.8f}', f'{new:.8f}', f'{new-old:+.8f}'])
    with (outdir / 'input_hashes.tsv').open('w') as f:
        f.write('input\tsha256\n')
        for path in (args.benchmark,args.legacy_precursors,args.legacy_fragments,args.production):
            f.write(f'{path}\t{sha256(Path(path))}\n')
    print(f'paired_summary={outdir / "paired_summary.tsv"}')
    print(f'paired_ms2_records={len(per_record)}')
    print(f'paired_rt_records={len(rt_values["legacy"])}')
    print('common_validation_parity=PASS')
    print('train_holdout_consumed=NO')
    print('historical_test_consumed=NO')


def self_test():
    assert metrics([1., 0.], [1., 0.])[0] == 1.
    assert metrics([1., 0.], [0., 0.])[0] == 0.
    assert metrics([1., 0.], [0., 1.])[0] == 0.
    assert abs(metrics([0., 1.], [0., 2.])[2] - 1.) < 1.e-12
    assert math.isclose(metrics([0., 1.], [0., 2.])[1], .5)
    print('research_parity_self_test=PASS')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('inventory')
    i.add_argument('--roots', nargs='+', required=True)
    i.add_argument('--output', required=True)
    i.add_argument('--max-depth', type=int, default=9)
    i.add_argument('--hash-models', action='store_true')
    i.set_defaults(run=inventory)
    c = sub.add_parser('compare')
    c.add_argument('--benchmark', required=True)
    c.add_argument('--legacy-checkpoint', required=True)
    c.add_argument('--integrated-checkpoint', required=True)
    c.add_argument('--legacy-precursors', required=True)
    c.add_argument('--legacy-fragments', required=True)
    c.add_argument('--production', required=True)
    c.add_argument('--prefix', type=int, default=2048)
    c.add_argument('--output', required=True)
    c.set_defaults(run=compare)
    t = sub.add_parser('self-test')
    t.set_defaults(run=lambda a: self_test())
    args = p.parse_args()
    if getattr(args, 'max_depth', 1) <= 0 or getattr(args, 'prefix', 1) <= 0:
        p.error('depth/prefix must be positive')
    try:
        args.run(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        p.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
