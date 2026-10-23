#!/usr/bin/env python3
"""Read-only validation of source-defined PT2017 candidate inputs; frozen v0.77 scoring.

No candidate construction or tuning. No TEST/HOLDOUT access. Evaluates only preselected DEV512.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path('/nfs/research/juan/DIA/singj/foundation_model')
TARGET_REL = 'experiments/redeem_pt2017_synthesis_newdev512_20261023'
PIN = {
    'catalog': 'a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f',
    'requests': '1fdf7df85071129ac316fff3bb1af62cba00f73adc8cf14bb30117227332dad5',
    'truth': '7fef846b19e56de0b93e4d22caf77fd04ff06b08c52a6db794f72fcc0ae49635',
    'workbook': '6d5a1134bee97644cc3197f4a02e1eb9448cbcd1674f616a46bc385106ffbf17',
    'old_truth': '614d00425c1bae2ce20f1363af8bfb03c251d7fd7638ce4eeb9fe1e915823cd5',
    'old_requests': '6ccc96815bd8c016abd91a8106dd584368b1d17ff8e121c0c0b561142fcdba3b',
    'frozen_evaluator': 'e740ac93b30b47266b9b5cd827c6a48b752c1b53e90d1cf8d65e6ac0630eb078',
    'sif': 'd46ed2bcc39ed50dec081762788ac60f4f3551658d44b555ce0aacf33504bf22',
    'parity_hits': 'b913d1cab42dddad77e11fd345c2d23e39b9dc56ed1fe6abc2c274a855bf8ba0',
}
REV = 'ffde6e4a2925f94dc28c87e33917db19a28f5869'
SIF_REL = 'containers/redeem-foundation_v0770-streaming-identifier_f7b8246c7730.sif'
RUNTIME_REL = 'experiments/redeem_external_dev_20261022/pt_validation_512_seed20261022/runtime_recovery_20261022/runtime.json'
BINARY = 'foundation_identify_stream_v0770'
SCHEMA = 'redeem.foundation.practical_identifier.v0752'


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def need(test: bool, message: str):
    if not test:
        raise ValueError(message)


def checked(path: Path, sha: str, desc: str) -> None:
    need(path.is_file() and path.stat().st_size > 0, f'{desc} missing/empty: {path}')
    need(digest(path) == sha, f'{desc} SHA-256 differs: {path}')


def rows(path: Path, expected: list[str]):
    with path.open(newline='') as fp:
        r = csv.DictReader(fp, delimiter='\t')
        need(r.fieldnames == expected, f'incorrect TSV schema in {path}: {r.fieldnames}')
        yield from r


def verify(catalog: Path, provenance: Path, coverage: Path, cohort: Path) -> dict:
    checked(catalog, PIN['catalog'], 'PT2017 source catalog')
    checked(cohort/'requests.tsv', PIN['requests'], 'new DEV512 requests')
    checked(cohort/'truth.tsv', PIN['truth'], 'new DEV512 PSM reference ledger')
    for f in [provenance, coverage, cohort/'metadata.json', cohort/'evidence.tsv']:
        need(f.is_file() and f.stat().st_size > 0, f'input missing: {f}')
    prov = json.loads(provenance.read_text())
    cov = json.loads(coverage.read_text())
    meta = json.loads((cohort/'metadata.json').read_text())
    need(prov.get('status') == 'PASS_SOURCE_DEFINED_CATALOG', 'invalid catalog builder status')
    need(prov.get('source_file_sha256') == PIN['workbook'] and prov.get('catalog_sha256') == PIN['catalog'], 'catalog source identity mismatch')
    need(prov.get('sheet_selection_policy') == 'pt2017-synthesis-v1' and prov.get('source_sheet_count') == 3, 'invalid synthesis-sheet scope')
    need(prov.get('excluded_sheets') == ['Quality Control Peptides'], 'QC sheet exclusion mismatch')
    need(prov.get('candidate_count') == 486810 and prov.get('unique_peptide_sequences') == 243405, 'catalog population mismatch')
    need(prov.get('evaluation_labels_used') is False and prov.get('measured_spectra_used') is False and prov.get('evaluation_requests_used') is False, 'catalog not label blind')
    need(meta.get('status') == 'PASS_DISJOINT_SOURCE_SCOPED_EXTERNAL_DEV', 'DEV cohort status mismatch')
    need(meta.get('source_revision') == REV and meta.get('source_partition') == 'validation', 'wrong dataset/split')
    need(meta.get('selected_queries') == 512 and meta.get('seed') == 20261023, 'DEV selection identity mismatch')
    need(meta.get('protected_partition_accessed') is False and meta.get('candidate_catalog_used_for_selection') is False, 'cohort access or selection contract mismatch')
    need(meta.get('prior_dev_truth_sha256') == PIN['old_truth'] and meta.get('prior_dev_requests_sha256') == PIN['old_requests'], 'previous DEV exclusion provenance mismatch')
    for f, key in [('requests.tsv','requests_sha256'), ('truth.tsv','truth_sha256'), ('evidence.tsv','evidence_sha256')]:
        checked(cohort/f, meta[key], f)
    need(meta['requests_sha256'] == PIN['requests'] and meta['truth_sha256'] == PIN['truth'], 'DEV source hashes changed')
    need(cov.get('status') == 'PASS_SOURCE_DEFINED_NEW_DEV_COVERAGE', 'coverage evidence not approved')
    need(cov.get('queries') == 512 and cov.get('reference_in_catalog') == 263 and cov.get('coverage_threshold_predeclared') == 0.5, 'coverage report discrepancy')
    need(cov.get('catalog_candidates') == 486810 and cov.get('catalog_sha256') == PIN['catalog'], 'coverage used another catalog')
    need(cov.get('cohort_requests_sha256') == PIN['requests'] and cov.get('cohort_truth_sha256') == PIN['truth'], 'coverage used another cohort')
    need(cov.get('coverage_fraction') == 263/512 and cov.get('next') == 'ALLOW_CPU_SCORING_IF_RUNTIME_RESOURCE_BUDGET_OK', 'resource gate not met')
    unique = {}
    for row in rows(cohort/'truth.tsv', ['query_id','truth_candidate_key','partition','evidence_source']):
        qid = row['query_id']
        need(qid and qid not in unique and row['partition'] == 'DEV', 'incorrect/duplicate DEV labels')
        need(row['evidence_source'].endswith('@'+REV), 'unverified query source')
        unique[qid] = row['truth_candidate_key']
    need(len(unique) == 512 and len(set(unique.values())) == 512, 'expected unique 512 peptide-charge identities')
    previous = set()
    seen_queries = set()
    for row in rows(cohort/'requests.tsv',['query_id','observed_precursor_mz','observed_charge','top_k','peak_mz','intensity']):
        qid=row['query_id']
        need(qid in unique and int(row['top_k']) == 256, 'invalid query or truncated pool request')
        need(int(row['observed_charge']) in (2,3), 'charge scope outside z2/z3')
        for field in ('observed_precursor_mz','peak_mz','intensity'):
            need(math.isfinite(float(row[field])), 'nonfinite spectrum measurement')
        seen_queries.add(qid)
    need(seen_queries == set(unique), 'request/truth query IDs differ')
    with catalog.open() as fp:
        need(fp.readline().strip() == 'schema: '+SCHEMA and fp.readline().strip() == 'candidates:', 'catalog schema mismatch')
        catalog_keys = set()
        for line in fp:
            if line.startswith('  - key: '):
                k = line[9:].strip()
                need(k not in catalog_keys, 'duplicate source catalog candidate')
                catalog_keys.add(k)
    need(len(catalog_keys) == 486810, 'catalog row count mismatch')
    need(sum(k in catalog_keys for k in unique.values()) == 263, 'coverage not independently reproduced')
    return {'status':'PASS_VERIFIED_SOURCE_DEFINED_INPUTS','queries':512,'catalog_candidates':486810,
            'truth_in_catalog':263,'catalog_sha256':PIN['catalog'],'requests_sha256':PIN['requests'],
            'truth_sha256':PIN['truth'],'source_revision':REV,
            'label_blind_catalog':True,'source_overlap_train':'NOT_CHECKED'}


def check_runtime(root: Path):
    sif = root / SIF_REL
    checked(sif, PIN['sif'], 'previously parity-qualified v0.77 SIF')
    evpath = root / RUNTIME_REL
    need(evpath.is_file(), 'prior runtime DEV512 parity provenance unavailable')
    ev = json.loads(evpath.read_text())
    need(ev.get('sif_sha256') == PIN['sif'] and ev.get('frozen_dev512_hits_sha256') == PIN['parity_hits'], 'prior SIF parity attestation differs')
    need(ev.get('sif') == str(sif), 'previously qualified SIF path differs')
    ver = subprocess.run(['singularity','exec','--bind',f'{root}:{root}',str(sif),BINARY,'--version'],capture_output=True,text=True,timeout=90,check=False)
    need(ver.returncode == 0 and ver.stdout.strip() == 'v0.77.0-practical-identifier-stream', 'identifier executable version mismatch')
    st = subprocess.run(['singularity','exec','--bind',f'{root}:{root}',str(sif),BINARY,'--self-test'],capture_output=True,text=True,timeout=120,check=False)
    need(st.returncode == 0 and 'v0770_candidate_pool=256' in st.stdout and 'v0770_practical_identifier_stream_self_test=PASS' in st.stdout, 'identifier self-test failed')
    return sif


def summary(ev: dict) -> dict:
    n=ev['queries']
    need(n == 512, 'evaluation denominator mismatch')
    catalog=ev['truth_in_catalog']
    pool=ev['truth_in_frozen_256_pool']
    need(catalog == 263 and 0 <= pool <= catalog, 'candidate coverage invariant failed')
    # Legacy and diagnostic accuracies are fractions over all queries.
    leg=round(ev['legacy_top1']*n)
    diag=round(ev['diagnostic_top1']*n)
    need(abs(ev['legacy_top1']*n-leg) < 1e-6 and abs(ev['diagnostic_top1']*n-diag)<1e-6, 'noninteger correct counts')
    need(0 <= leg <= pool and 0 <= diag <= pool, 'top1 outside pool')
    out={'protocol':'ReDeeM_PT2017_source_catalog_new_disjoint_DEV512_v1',
         'status':'PASS_EXPLORATORY_EXTERNAL_DEV_ONLY','queries':n,
         'source_defined_catalog_candidates':486810,'truth_in_catalog':catalog,
         'catalog_coverage':catalog/n,'truth_in_nearest256_pool':pool,
         'nearest256_recall_unconditional':pool/n,
         'nearest256_recall_given_catalog':pool/catalog if catalog else None,
         'legacy_top1_correct':leg,'legacy_top1_unconditional':leg/n,
         'legacy_top1_given_truth_in_pool':leg/pool if pool else None,
         'diagnostic_top1_correct':diag,'diagnostic_top1_unconditional':diag/n,
         'diagnostic_top1_given_truth_in_pool':diag/pool if pool else None,
         'legacy_mrr_unconditional':ev['legacy_mrr'],
         'legacy_top10_unconditional':ev['legacy_top10'],
         'diagnostic_mrr_unconditional':ev['diagnostic_mrr'],
         'changed_top1_candidates':ev['changed_top1'],
         'diagnostic_corrected':ev['diagnostic_corrected'],
         'diagnostic_regressed':ev['diagnostic_regressed'],
         'unmodified_z2_z3_only':True,
         'frozen_identifier_sha256':PIN['sif'],
         'candidate_library_sha256':PIN['catalog'],
         'validation_requests_sha256':PIN['requests'],
         'validation_psm_reference_sha256':PIN['truth'],
         'scientific_limitation':'PSM-derived validation references are not orthogonally verified synthetic identities; historical ReDeeM TRAIN overlap not checked',
         'independent_test_accessed':False,
         'candidate_score_fit_performed':False,
         'production_change_approved':False}
    need(out['diagnostic_corrected']-out['diagnostic_regressed'] == diag-leg, 'paired correction arithmetic mismatch')
    return out


def run(root: Path, job: str):
    need(job.isdigit(), 'missing numeric Slurm job ID')
    stage = root/TARGET_REL
    cat = stage/'input'/'catalog.yaml'
    provenance = stage/'input'/'catalog_provenance.json'
    coverage = stage/'input'/'coverage.json'
    cohort = stage/'input'/'cohort'
    verified=verify(cat,provenance,coverage,cohort)
    sif=check_runtime(root)
    original=root/'scripts'/'redeem_external_dev_benchmark_20261022.py'
    checked(original,PIN['frozen_evaluator'],'original frozen evaluation script')
    result = stage / ('run_'+job)
    need(not result.exists(), 'existing job result directory; preserve')
    result.mkdir(parents=False)
    hits = result/'identifier_hits.tsv'
    ident_summary = result/'identifier_summary.tsv'
    cmd = ['singularity','exec','--bind',f'{root}:{root}',str(sif),BINARY,'identify-stream',
           '--catalog',str(cat),'--spectra',str(cohort/'requests.tsv'),
           '--output-hits',str(hits),'--output-summary',str(ident_summary)]
    with (result/'identifier.stdout.log').open('x') as output, (result/'identifier.stderr.log').open('x') as errors:
        ret = subprocess.run(cmd,stdout=output,stderr=errors,check=False).returncode
    need(ret == 0 and hits.is_file() and ident_summary.is_file(), f'identifier failed rc={ret}; partial output preserved: {result}')
    spec = importlib.util.spec_from_file_location('redeem_v0770_frozen_experimental_evaluator', original)
    need(spec and spec.loader, 'original evaluator import failed')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Historical evaluator expects source_provenance_status from the earlier DEV adapter.
    # The new cohort records the same pinned HF validation lineage in different fields.
    # Make a read-only-link evaluation view: never alter source metadata or TSV bytes.
    compat = result/'evaluator_view'
    compat.mkdir()
    for name in ('requests.tsv','truth.tsv','evidence.tsv'):
        (compat/name).symlink_to(cohort/name)
    source_meta = json.loads((cohort/'metadata.json').read_text())
    need(source_meta['source_revision'] == REV and source_meta['source_partition'] == 'validation', 'evaluation adapter source not validated')
    need('source_provenance_status' not in source_meta, 'unexpected schema change in original cohort')
    adapted = {**source_meta,
               'source_provenance_status':'huggingface_hub_commit_pinned_validation',
               'compatibility_adapter_only':'legacy evaluator expects source_provenance_status; raw DEV cohort untouched'}
    with (compat/'metadata.json').open('x') as f:
        json.dump(adapted,f,indent=2,sort_keys=True);f.write('\n')
    ev = module.evaluate(argparse.Namespace(prepared_dir=str(compat),catalog=str(cat),hits=str(hits),output_prefix=str(result/'frozen_evaluation')))
    report=summary(ev)
    report['evaluator_compat_metadata_sha256']=digest(compat/'metadata.json')
    report['source_cohort_metadata_sha256']=digest(cohort/'metadata.json')
    report['evaluator_compat_source_files']='read_only_symlinks_to_original_cohort'
    report['hits_sha256']=digest(hits)
    report['runtime_preflight']='PREVIOUS_FROZEN_DEV512_BYTE_PARITY_AND_CURRENT_SIF_PIN'
    report['input_validation']=verified
    with (result/'score_summary.json').open('x') as out:
        json.dump(report,out,indent=2,sort_keys=True);out.write('\n')
    print('REDEEM_PT2017_SCORING=PASS queries=512 catalog=263 pool=%d top1=%d' % (report['truth_in_nearest256_pool'],report['legacy_top1_correct']),flush=True)
    print('REDEEM_PT2017_SCORING_RESULT='+str(result/'score_summary.json'),flush=True)


def self_test():
    sample={'queries':512,'truth_in_catalog':263,'truth_in_frozen_256_pool':200,
            'legacy_top1':145/512,'diagnostic_top1':147/512,'legacy_top10':180/512,
            'legacy_mrr':0.31,'diagnostic_mrr':0.312,'changed_top1':18,
            'diagnostic_corrected':5,'diagnostic_regressed':3}
    got=summary(sample)
    assert got['nearest256_recall_given_catalog'] == 200/263
    assert got['legacy_top1_correct']==145 and got['diagnostic_top1_correct']==147
    assert got['legacy_top1_given_truth_in_pool']==145/200
    try:
        summary({**sample,'diagnostic_regressed':2})
        raise AssertionError('inconsistent delta accepted')
    except ValueError:
        pass
    try:
        summary({**sample,'truth_in_frozen_256_pool':300})
        raise AssertionError('impossible pool count accepted')
    except ValueError:
        pass
    print('REDEEM_PT2017_SCORING_SELFTEST=PASS conditional_denominators,paired_delta,coverage_gate')


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('self-test','verify-inputs','verify-runtime','run'))
    parser.add_argument('--catalog')
    parser.add_argument('--provenance')
    parser.add_argument('--coverage')
    parser.add_argument('--cohort')
    parser.add_argument('--job-id',default=os.environ.get('SLURM_JOB_ID',''))
    args=parser.parse_args()
    try:
        if args.mode=='self-test':self_test()
        elif args.mode=='verify-inputs':
            cat=Path(args.catalog) if args.catalog else ROOT/TARGET_REL/'input'/'catalog.yaml'
            prov=Path(args.provenance) if args.provenance else ROOT/TARGET_REL/'input'/'catalog_provenance.json'
            cov=Path(args.coverage) if args.coverage else ROOT/TARGET_REL/'input'/'coverage.json'
            cohort=Path(args.cohort) if args.cohort else ROOT/TARGET_REL/'input'/'cohort'
            result=verify(cat,prov,cov,cohort)
            print('REDEEM_PT2017_INPUT=PASS queries=512 covered=263 catalog_candidates=486810')
            print(json.dumps(result,indent=2,sort_keys=True))
        elif args.mode=='verify-runtime':
            sif=check_runtime(ROOT)
            print('REDEEM_PT2017_RUNTIME=PASS sha256='+digest(sif))
        else:run(ROOT,args.job_id)
        return 0
    except (ValueError,KeyError,OSError,TypeError,subprocess.TimeoutExpired) as e:
        print('REDEEM_PT2017_SCORING=BLOCKED '+str(e),file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
