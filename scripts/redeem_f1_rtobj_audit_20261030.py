#!/usr/bin/env python3
"""Read-only independent post-run audit for the ReDeeM F1 RT-objective A/B (8x512 disjoint).

Derived from redeem_disjoint512_audit_20261020.py (SHA d5f70faa...). Changes: source commit
pin, SIF SHA from env F1_SIF_SHA256, and a required rt_objective per arm (env F1_RT_OBJECTIVE).
This is an experiment gate, not the trainer or a replacement for checkpoint reload.
Requires already-produced output files; never touches checkpoints or any cohort.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

SOURCE_SHA = 'a832c3a40225149a9ce84779f796977768be9e70'
SIF_SHA = os.environ.get('F1_SIF_SHA256', 'unset-sif-sha')
RT_OBJECTIVE = os.environ.get('F1_RT_OBJECTIVE', 'IntrinsicAndObserved')
RUN_SHA = '7214b9cdd3836ec028655d5d6b8bb088280558626b24c9afc63ec06d582b4201'
EPOCHS = 8
CYCLES = 512
PROP_BATCH = 8
VAL_RECORDS = 1024
EXPECTED_RT = 504
EXPECTED_CCS = 504
EXPECTED_MS2 = 1024
EXPECTED_POINTS = 5986
INV = 256
GREEDY = 64
RETRIEVAL = 512
MIN_UNIQUE = 60000

class AuditError(Exception):
    pass

def check(ok: bool, message: str) -> None:
    if not ok:
        raise AuditError(message)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def kv_yaml(text: str, key: str) -> str:
    matches = re.findall(rf'(?m)^\s*{re.escape(key)}\s*:\s*([^#\r\n]+)', text)
    check(len(matches) == 1, f'training.yaml expected exactly one {key} (found {len(matches)})')
    return matches[0].strip().strip('"\'')

def read_tsv(path: Path, name: str) -> list[dict[str, str]]:
    check(path.is_file() and path.stat().st_size > 0, f'missing/empty {name}: {path}')
    with path.open(newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        check(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), f'{name}: duplicate or absent header')
        rows = list(reader)
    check(len(rows) == EPOCHS, f'{name}: expected {EPOCHS} rows; got {len(rows)}')
    check(all(row.get('epoch') == str(i) for i, row in enumerate(rows, 1)), f'{name}: epoch sequence is not 1..{EPOCHS}')
    return rows

def integer(row: dict[str, str], field: str, where: str) -> int:
    try:
        val = int(row[field])
    except (KeyError, ValueError, TypeError):
        raise AuditError(f'{where}: missing/invalid integer {field}')
    return val

def number(row: dict[str, str], field: str, where: str) -> float:
    try:
        val = float(row[field])
    except (KeyError, ValueError, TypeError):
        raise AuditError(f'{where}: missing/invalid numeric {field}')
    check(math.isfinite(val), f'{where}: non-finite {field}')
    return val

def epoch_metadata(metadata: Path) -> int:
    check(metadata.is_file(), f'checkpoint metadata absent: {metadata}')
    text = metadata.read_text()
    value = kv_yaml(text, 'completed_epochs')
    try:
        epoch = int(value)
    except ValueError:
        raise AuditError(f'checkpoint metadata completed_epochs non-integer: {metadata}')
    check(1 <= epoch <= EPOCHS, f'checkpoint epoch out of range: {metadata}: {epoch}')
    return epoch

def audit(model: Path, provenance: Path, job_stdout: Path | None, run_yaml: Path | None) -> dict:
    check(model.is_dir(), f'model dir absent: {model}')
    settings = (model / 'training.yaml')
    check(settings.is_file(), f'resolved training.yaml missing: {settings}')
    cfg = settings.read_text()
    required = {
        'batch_size': '8', 'strategy': 'research_curriculum',
        'epochs': '8', 'max_train_batches_per_epoch': '512',
        'property_batch_size': '8', 'seed': '20261006',
        'max_validation_batches': '128', 'inverse_evaluation_records': '256',
        'generation_evaluation_records': '64',
    }
    for key, expected in required.items():
        got = kv_yaml(cfg, key)
        check(got == expected, f'training.yaml {key}: expected {expected}, observed {got}')
    check(kv_yaml(cfg, 'early_stopping_patience') in {'8', '9', '10'}, 'early stopping could truncate run')
    # Optimizer / physics inputs remain the pinned defaults; launcher must preserve them.
    check(kv_yaml(cfg, 'warmup_steps') == '10', 'warmup_steps changed')
    check(abs(float(kv_yaml(cfg, 'learning_rate')) - 0.00002) < 1e-12, 'learning_rate changed')
    check(abs(float(kv_yaml(cfg, 'min_learning_rate_ratio')) - 0.1) < 1e-9, 'min_learning_rate_ratio changed')
    for field, expected in [('max_gradient_norm', 1.0), ('pair_auxiliary_weight', 0.1),
                             ('chemistry_summary_weight', 0.1), ('conformation_proxy_weight', 0.1),
                             ('inverse_weight', 1.0), ('inverse_ptm_mass_weight', 0.1),
                             ('cross_modal_alignment_weight', 0.1)]:
        check(abs(float(kv_yaml(cfg, field)) - expected) < 1e-9, f'{field} changed')
    check(kv_yaml(cfg, 'mobility_consensus_supervision').lower() == 'true', 'mobility consensus disabled')
    check(RT_OBJECTIVE in {'IntrinsicAndObserved', 'Harmonized'}, f'invalid arm rt_objective {RT_OBJECTIVE}')
    check(kv_yaml(cfg, 'rt_objective') == RT_OBJECTIVE, f'rt_objective: expected {RT_OBJECTIVE}')
    check(kv_yaml(cfg, 'peptide_backbone') == 'residue_pair_task_tokens', 'peptide_backbone changed')
    check(kv_yaml(cfg, 'auxiliary_supervision_heads').lower() == 'true', 'auxiliary heads changed')
    check(kv_yaml(cfg, 'ms2_residual_blend').lower() == 'true', 'specialists changed')

    exposure = read_tsv(model / 'training_exposure.tsv', 'training_exposure.tsv')
    prev = 0
    expected_records = None
    exposed = []
    for ep, row in enumerate(exposure, 1):
        where = f'training exposure epoch {ep}'
        for field, expected in [('cycles', CYCLES), ('property_updates', CYCLES * 2),
                                ('property_batch_size', PROP_BATCH), ('property_records', CYCLES * 2 * PROP_BATCH)]:
            check(integer(row, field, where) == expected, f'{where}: {field} expected {expected}, got {row.get(field)}')
        fresh = integer(row, 'new_property_records', where)
        cumulative = integer(row, 'cumulative_unique_property_records', where)
        total = integer(row, 'train_records', where)
        if expected_records is None:
            expected_records = total
        check(total == expected_records and total == 3651363, f'{where}: TRAIN cohort changed ({total})')
        check(0 < fresh <= 8192, f'{where}: new property indices out of bounds ({fresh})')
        check(cumulative == prev + fresh and cumulative <= min(ep * 8192, total), f'{where}: impossible cumulative unique {cumulative}')
        for label in ('rt_labelled', 'ccs_labelled', 'ms2_labelled'):
            check(0 <= integer(row, label, where) <= 8192, f'{where}: {label} out of range')
        inv_train = integer(row, 'inverse_spectrum_records', where)
        check(0 <= inv_train <= 4096, f'{where}: inverse_spectrum_records invalid')
        exposed.append(cumulative)
        prev = cumulative
    check(prev >= MIN_UNIQUE, f'property exposure below preregistered floor {MIN_UNIQUE}: {prev}; SCIENTIFIC ASSESSMENT INCOMPLETE')

    validation = read_tsv(model / 'validation_metrics.tsv', 'validation_metrics.tsv')
    metrics = []
    for ep, row in enumerate(validation, 1):
        where = f'validation epoch {ep}'
        for field, expected in [ ('records', VAL_RECORDS), ('rt_count', EXPECTED_RT),
                                 ('ccs_count', EXPECTED_CCS), ('ms2_spectrum_count', EXPECTED_MS2),
                                 ('ms2_point_count', EXPECTED_POINTS), ('inverse_sequence_count', INV),
                                 ('retrieval_queries', RETRIEVAL), ('generation_attempts', GREEDY),
                                 ('inverse_ptm_site_count', 44) ]:
            check(integer(row, field, where) == expected, f'{where}: {field} changed to {row.get(field)}, expected {expected}')
        for key in row:
            if key != 'epoch':
                number(row, key, where)
        for key in ('rt_mae_native', 'ccs_mae', 'ms2_cosine_similarity', 'ms2_spectral_angle',
                    'inverse_mean_nll', 'inverse_ptm_mass_rmse_da', 'retrieval_top1_accuracy',
                    'retrieval_mrr', 'generation_success_rate', 'generation_sequence_exact_rate',
                    'generation_sequence_il_equivalent_rate', 'generation_mean_length'):
            number(row, key, where)
        for key in ('generation_success_rate','generation_sequence_exact_rate',
                    'generation_sequence_il_equivalent_rate','retrieval_top1_accuracy'):
            check(0.0 <= float(row[key]) <= 1.0, f'{where}: invalid rate {key}')
        metrics.append({k: float(row[k]) for k in ('rt_mae_native', 'ccs_mae', 'ms2_cosine_similarity',
                                                  'inverse_mean_nll', 'generation_sequence_il_equivalent_rate')})

    history = read_tsv(model / 'training_history.tsv', 'training_history.tsv')
    for ep, row in enumerate(history, 1):
        for key in row:
            if key != 'epoch':
                number(row, key, f'training history epoch {ep}')
        for field in ('train_total', 'train_forward', 'train_inverse', 'validation_total',
                      'validation_forward', 'validation_inverse'):
            number(row, field, f'training history epoch {ep}')

    check(provenance.is_file(), f'provenance missing: {provenance}')
    lines = {}
    for line in provenance.read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            lines[key] = value
    for key, expected in (('source_sha', SOURCE_SHA), ('sif_sha256', SIF_SHA), ('run_yaml_sha256', RUN_SHA),
                          ('epochs', '8'), ('train_cycles_per_epoch', '512'),
                          ('historical_holdout_selected', 'NO'), ('historical_test_selected', 'NO')):
        check(lines.get(key) == expected, f'provenance {key}: expected {expected}, observed {lines.get(key)!r}')
    # Require a new proof that disjoint batching was resolved, rather than just the external settings.
    check(kv_yaml(cfg, 'property_batch_size') == '8' and all(integer(r, 'property_batch_size', 'exposure') == 8 for r in exposure),
          'disjoint property batch contract not proven')
    if run_yaml is not None:
        check(run_yaml.is_file() and sha256(run_yaml) == RUN_SHA, 'prepared input YAML SHA mismatch')
    if job_stdout is not None:
        check(job_stdout.is_file(), f'job stdout missing: {job_stdout}')
        output = job_stdout.read_text()
        for selector in ('best_validation', 'best_forward', 'best_inverse'):
            check(re.search(rf'LEARNING_CURVE_CHECKPOINT_RELOAD=PASS selector={selector}\b', output) is not None,
                  f'missing checkpoint reload: {selector}')
    epochs = {
        'best_validation': epoch_metadata(model / 'metadata.yaml'),
        'best_forward': epoch_metadata(model / 'best_forward' / 'metadata.yaml'),
        'best_inverse': epoch_metadata(model / 'best_inverse' / 'metadata.yaml'),
    }
    for name, folder in [('best_validation', model), ('best_forward', model/'best_forward'), ('best_inverse', model/'best_inverse')]:
        check((folder/'model.safetensors').is_file() and (folder/'model.safetensors').stat().st_size > 0,
              f'checkpoint weights missing for {name}')
    return {'result': 'PASS', 'protocol': 'research_curriculum_8x512_disjoint8',
            'train_unique_indices': prev, 'train_records': expected_records,
            'training_fraction': prev / expected_records,
            'selected_checkpoint_epochs': epochs,
            'validation': {str(ep): metrics[ep-1] for ep in (1, 2, 4, 8)}}

def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        model = base / 'model'
        model.mkdir()
        cfg = '\n'.join(f'{k}: {v}' for k, v in {
            'batch_size':'8', 'strategy':'research_curriculum', 'epochs':'8',
            'max_train_batches_per_epoch':'512', 'property_batch_size':'8', 'seed':'20261006',
            'max_validation_batches':'128', 'inverse_evaluation_records':'256',
            'generation_evaluation_records':'64', 'early_stopping_patience':'8',
            'warmup_steps':'10', 'learning_rate':'0.00002', 'min_learning_rate_ratio':'0.1',
            'max_gradient_norm':'1.0','mobility_consensus_supervision':'true',
            'pair_auxiliary_weight':'0.1','chemistry_summary_weight':'0.1',
            'conformation_proxy_weight':'0.1','inverse_weight':'1.0',
            'inverse_ptm_mass_weight':'0.1','cross_modal_alignment_weight':'0.1',
            'rt_objective':RT_OBJECTIVE,'peptide_backbone':'residue_pair_task_tokens',
            'auxiliary_supervision_heads':'true','ms2_residual_blend':'true',
        }.items()) + '\n'
        (model/'training.yaml').write_text(cfg)
        headers = ['epoch','cycles','property_updates','property_batch_size','property_records','new_property_records','cumulative_unique_property_records','train_records','rt_labelled','ms2_labelled','ccs_labelled','inverse_spectrum_records']
        def exposure_rows(fresh=8170):
            return [dict(epoch=str(i), cycles='512', property_updates='1024',property_batch_size='8',
                         property_records='8192',new_property_records=str(fresh),
                         cumulative_unique_property_records=str(i*fresh),train_records='3651363',
                         rt_labelled='7000',ms2_labelled='7000',ccs_labelled='2000',inverse_spectrum_records='4096') for i in range(1,9)]
        def save_tsv(path, heads, rows):
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f, fieldnames=heads, delimiter='\t')
                w.writeheader();w.writerows(rows)
        save_tsv(model/'training_exposure.tsv', headers, exposure_rows())
        mfields=['epoch','records','rt_count','ccs_count','ms2_spectrum_count','ms2_point_count',
                 'inverse_sequence_count','retrieval_queries','generation_attempts','inverse_ptm_site_count',
                 'rt_mae_native','ccs_mae','ms2_cosine_similarity','ms2_spectral_angle',
                 'inverse_mean_nll','inverse_ptm_mass_rmse_da','retrieval_top1_accuracy',
                 'retrieval_mrr','generation_success_rate','generation_sequence_exact_rate',
                 'generation_sequence_il_equivalent_rate','generation_mean_length']
        rows = [dict(epoch=str(i), records='1024',rt_count='504',ccs_count='504',
                     ms2_spectrum_count='1024',ms2_point_count='5986',inverse_sequence_count='256',
                     retrieval_queries='512',generation_attempts='64',inverse_ptm_site_count='44',
                     rt_mae_native='12',ccs_mae='26',ms2_cosine_similarity='0.7',
                     ms2_spectral_angle='0.56', inverse_mean_nll='2.5',
                     inverse_ptm_mass_rmse_da='100',retrieval_top1_accuracy='0.2',retrieval_mrr='0.4',
                     generation_success_rate='1',generation_sequence_exact_rate='0',
                     generation_sequence_il_equivalent_rate='0',generation_mean_length='9') for i in range(1,9)]
        save_tsv(model/'validation_metrics.tsv', mfields, rows)
        hfields=['epoch','train_total','train_forward','train_inverse','validation_total','validation_forward','validation_inverse']
        save_tsv(model/'training_history.tsv',hfields,[dict(epoch=str(i),train_total='1',train_forward='0.4',train_inverse='0.5',
                 validation_total='2',validation_forward='0.8',validation_inverse='1') for i in range(1,9)])
        prov=base/'provenance.txt'
        prov.write_text('\n'.join(f'{key}={val}' for key,val in {
            'source_sha':SOURCE_SHA,'sif_sha256':SIF_SHA,'run_yaml_sha256':RUN_SHA,
            'epochs':'8','train_cycles_per_epoch':'512','historical_holdout_selected':'NO',
            'historical_test_selected':'NO'}.items()))
        for ck, ep in [(model, 8),(model/'best_forward',8),(model/'best_inverse',6)]:
            ck.mkdir(exist_ok=True)
            (ck/'metadata.yaml').write_text(f'completed_epochs: {ep}\n')
            (ck/'model.safetensors').write_bytes(b'valid-synthetic-test-only')
        job=base/'job.out'
        job.write_text('\n'.join(f'LEARNING_CURVE_CHECKPOINT_RELOAD=PASS selector={s} records=8' for s in
                                 ('best_validation','best_forward','best_inverse')))
        assert audit(model, prov, job, None)['train_unique_indices'] == 65360
        (model/'training.yaml').write_text(cfg.replace('property_batch_size: 8','property_batch_size: null'))
        try: audit(model,prov,job,None)
        except AuditError as e: assert 'property_batch_size' in str(e)
        else: raise AssertionError('failed to reject null property batch')
        (model/'training.yaml').write_text(cfg)
        save_tsv(model/'training_exposure.tsv',headers,exposure_rows(fresh=7000))
        try: audit(model,prov,job,None)
        except AuditError as e: assert 'below preregistered floor' in str(e)
        else: raise AssertionError('failed to reject low TRAIN exposure')
        save_tsv(model/'training_exposure.tsv',headers,exposure_rows())
        rows[7]['rt_count']='505'
        save_tsv(model/'validation_metrics.tsv',mfields,rows)
        try: audit(model,prov,job,None)
        except AuditError as e: assert 'rt_count' in str(e)
        else: raise AssertionError('failed to reject validation denominator mutation')
    print('DISJOINT512_AUDIT_SELFTEST=PASS valid, null-property, low-exposure, denominator-mutation')

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--provenance',type=Path)
    ap.add_argument('--job-stdout',type=Path)
    ap.add_argument('--source-yaml',type=Path)
    ap.add_argument('--json-report',type=Path,help='optional write report; refuses overwrite')
    a = ap.parse_args()
    if a.self_test:
        self_test(); return 0
    if a.model_dir is None or a.provenance is None:
        ap.error('--model-dir and --provenance required without --self-test')
    try:
        result = audit(a.model_dir,a.provenance,a.job_stdout,a.source_yaml)
        if a.json_report:
            check(not a.json_report.exists(), f'report already exists: {a.json_report}')
            a.json_report.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(f'DISJOINT512_AUDIT=PASS unique={result["train_unique_indices"]} train_fraction={result["training_fraction"]:.6f}')
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0
    except (AuditError, OSError, ValueError) as exc:
        print(f'DISJOINT512_AUDIT=FAIL reason={exc}',file=sys.stderr)
        return 2
if __name__ == '__main__':
    sys.exit(main())
