#!/usr/bin/env python3
"""ReDeeM external measured-spectrum DEV bridge; no model fitting or protected-data reads.

Requires Python stdlib for synthetic JSONL. Requires datasets and huggingface_hub
for remote pinned validation and PyYAML for frozen catalog evaluation.
"""
import argparse
import csv
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from collections import Counter

DATASET = 'InstaDeepAI/ms_proteometools'
PROTON = 1.00727646677
SIF_SHA = 'c99234a0e71390ad0e2d1bee8411e7a62c8ecba87f752852590d72c730085fc4'
ROOT = Path('/nfs/research/juan/DIA/singj/foundation_model')
CATALOG_REL = 'experiments/practical_identifier_v0761_03_actual_dev_equivalence_final/catalog.yaml'
CATALOG_SHA = 'fffc44ceea4e6e22b19722b92ca07679dc4cfd94284911570c8a7971f1c8f857'
SIF_REL = 'containers/redeem-foundation_published-0f7b9f4c2be9.sif'
REQ_COLS = ['query_id','observed_precursor_mz','observed_charge','top_k','peak_mz','intensity']
HIT_COLS = ['query_id','rank','candidate_index','key','charge','theoretical_neutral_mass','absolute_neutral_mass_error','geometry_score']
ALPHABET = set('ACDEFGHIKLMNPQRSTVWY')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fail(msg):
    raise ValueError(msg)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def make_uid(record):
    identifying = [str(record.get(x, '')) for x in ('experiment_name','evidence_index','scan_number')]
    if any(not x for x in identifying):
        fail('missing experimentally identifying metadata')
    return 'ptdev-' + hashlib.sha256(('\x1f'.join(identifying)).encode()).hexdigest()[:20]


def parse_spectrum(record):
    try:
        seq = str(record['sequence'])
        modified = str(record['modified_sequence'])
        charge = int(record['precursor_charge'])
        precursor = float(record['precursor_mz'])
        mzs = record['mz_array']
        intensities = record['intensity_array']
    except (KeyError, TypeError, ValueError) as exc:
        fail(f'missing or invalid spectrum field: {exc}')
    if modified != seq:
        return None, 'modified_unqualified'
    if 'C' in seq:
        return None, 'cysteine_modification_uncertain'
    if not (7 <= len(seq) <= 64) or any(x not in ALPHABET for x in seq):
        return None, 'sequence_invalid'
    if charge not in (2, 3):
        return None, 'charge_outside_declared_z2_z3_scope'
    if not (math.isfinite(precursor) and 100 < precursor < 5000):
        return None, 'precursor_invalid'
    if len(mzs) != len(intensities):
        return None, 'peak_array_mismatch'
    peaks = []
    for mz, intensity in zip(mzs, intensities):
        try:
            m = float(mz); i = float(intensity)
        except (TypeError, ValueError):
            continue
        if math.isfinite(m) and math.isfinite(i) and m > 0 and i > 0:
            peaks.append((m, i))
    if len(peaks) < 10:
        return None, 'too_few_valid_peaks'
    # Frozen identifier will retain at most 256; applying the same cap in advance
    # reduces transfer volume. Equal-intensity ties are broken by increasing m/z.
    peaks = sorted(sorted(peaks, key=lambda p:(-p[1], p[0]))[:256])
    return {
        'query_id': make_uid(record), 'sequence': seq, 'charge': charge,
        'observed_precursor_mz': precursor, 'peaks': peaks,
        'experiment_name': str(record['experiment_name']),
        'evidence_index': str(record['evidence_index']),
        'scan_number': str(record['scan_number']),
    }, None


def select_records(stream, limit, seed, scan_cap=0):
    if not 1 <= limit <= 10000:
        fail('limit must be between 1 and 10000')
    selected = {}  # (sequence, charge) -> (hash score, query ID, parsed record)
    heap = []  # max-heap by negative priority, lazily reaped on changes
    counts = Counter()
    def worst():
        while heap:
            neg, uid, group = heap[0]
            current = selected.get(group)
            if current is None or current[0] != -neg or current[1] != uid:
                heapq.heappop(heap)
            else:
                return -neg, uid, group
        return None
    for raw in stream:
        if scan_cap and counts['scanned_rows'] >= scan_cap:
            counts['stopped_by_scan_cap'] = 1
            break
        counts['scanned_rows'] += 1
        try:
            candidate, reason = parse_spectrum(raw)
        except (ValueError, KeyError, TypeError):
            counts['malformed_rows'] += 1
            continue
        if reason:
            counts[reason] += 1
            continue
        counts['qualified_rows'] += 1
        group = (candidate['sequence'], candidate['charge'])
        uid = candidate['query_id']
        score = int(hashlib.sha256(f'{seed}|{uid}'.encode()).hexdigest(), 16)
        value = (score, uid, candidate)
        if group in selected:
            if value[:2] >= selected[group][:2]:
                continue
            selected[group] = value
            heapq.heappush(heap, (-score, uid, group))
        elif len(selected) < limit:
            selected[group] = value
            heapq.heappush(heap, (-score, uid, group))
        else:
            low = worst()
            if low is None:
                fail('heap corrupted')
            if (score, uid) >= low[:2]:
                continue
            selected.pop(low[2])
            heapq.heappop(heap)
            selected[group] = value
            heapq.heappush(heap, (-score, uid, group))
    rows = [s[2] for s in sorted(selected.values(), key=lambda v:(v[0], v[1]))]
    if len(rows) != limit:
        fail(f'qualified distinct precursor count {len(rows)} below requested {limit}')
    if len({v['query_id'] for v in rows}) != limit:
        fail('duplicate query identifier in selected result')
    counts['selected_distinct_peptidoform_charge'] = len(rows)
    return rows, dict(counts)


def remote_records(revision):
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        fail(f'HF mode requires datasets and huggingface_hub: {exc}')
    info = HfApi().dataset_info(DATASET, revision=revision)
    pinned = info.sha
    if not pinned or len(pinned) < 35:
        fail('could not resolve immutable upstream dataset revision')
    return load_dataset(DATASET, split='validation', revision=pinned, streaming=True), pinned


def prepare(args):
    out = Path(args.output_dir)
    if out.exists():
        fail('output directory exists: do not overwrite')
    if args.source == 'hf':
        records, revision = remote_records(args.revision)
        source_validation = 'huggingface_hub_commit_pinned_validation'
    else:
        if not args.input_jsonl or not Path(args.input_jsonl).is_file():
            fail('local JSONL source missing')
        def read_local():
            with Path(args.input_jsonl).open() as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        records = read_local()
        revision = 'local-jsonl-sha256:' + sha(args.input_jsonl)
        source_validation = 'LOCAL_UNVERIFIED_FIXTURE_NOT_EXTERNAL_BENCHMARK'
    rows, counts = select_records(records, args.limit, args.seed, args.scan_cap)
    out.mkdir(parents=True)
    req = out / 'requests.tsv'; truth = out / 'truth.tsv'; mapf = out / 'evidence.tsv'
    with req.open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(REQ_COLS)
        for r in rows:
            for mz, intensity in r['peaks']:
                w.writerow([r['query_id'], format(r['observed_precursor_mz'], '.12g'),
                            r['charge'], 256, format(mz,'.12g'), format(intensity,'.12g')])
    with truth.open('w', newline='') as f, mapf.open('w', newline='') as g:
        t = csv.writer(f, delimiter='\t', lineterminator='\n')
        e = csv.writer(g, delimiter='\t', lineterminator='\n')
        t.writerow(['query_id','truth_candidate_key','partition','evidence_source'])
        e.writerow(['query_id','experiment_name','evidence_index','scan_number','sequence','precursor_charge','peak_count'])
        for r in rows:
            t.writerow([r['query_id'], f"{r['sequence']}|z{r['charge']}", 'DEV',
                        f'{DATASET}/validation@{revision}'])
            e.writerow([r['query_id'], r['experiment_name'],r['evidence_index'],r['scan_number'],
                        r['sequence'],r['charge'],len(r['peaks'])])
    summary = {'protocol':'ReDeeM_external_measured_unmodified_DEV_v1',
               'source_dataset':DATASET if args.source == 'hf' else 'local_jsonl',
               'source_revision':revision,'source_partition':'validation',
               'source_provenance_status':source_validation, 'evidence_type':'published_high_confidence_PSMs_from_synthetic_peptides__not_orthogonal_truth',
               'historical_redeem_train_overlap':'NOT_CHECKED',
               'selection':'lowest_seeded_sha256_over_spectrum_identifiers_one_record_per_unmodified_sequence_and_charge',
               'seed':args.seed,'limit':args.limit,'scan_cap':args.scan_cap,
               'source_counts':counts,'selected_queries':len(rows),
               'requests_sha256':sha(req),'truth_sha256':sha(truth),'evidence_sha256':sha(mapf),
               'excluded_ptm_cases':'modified_sequence != sequence or cysteine residue excluded in v1',
               'declared_charge_scope':'z2,z3',
               'protected_partition_accessed':False,'results':'NOT_EVALUATED',
               'modification_localization':'NOT_EVALUATED'}
    dump(out/'metadata.json', summary)
    print(f'REDEEM_EXTERNAL_PREPARE=PASS queries={len(rows)} scanned={counts["scanned_rows"]} source={source_validation}')
    print('REDEEM_EXTERNAL_OUTPUT=' + str(out))
    return summary


def load_catalog(path):
    try:
        import yaml
    except ImportError:
        fail('evaluation requires PyYAML for the frozen catalog')
    data = yaml.safe_load(Path(path).read_text())
    if data.get('schema') != 'redeem.foundation.practical_identifier.v0752':
        fail('unexpected catalog schema')
    cands = data.get('candidates')
    if not isinstance(cands, list) or not cands:
        fail('catalog has no candidates')
    counts = Counter()
    keys = set()
    for row in cands:
        key = row['key']; z=int(row['charge'])
        if key in keys or z <= 0 or not key.endswith(f'|z{z}'):
            fail('catalog duplicate or malformed candidate key')
        keys.add(key); counts[z] += 1
    return cands, keys, counts


def read_tsv(path, cols):
    with Path(path).open(newline='') as f:
        rdr = csv.DictReader(f, delimiter='\t')
        if rdr.fieldnames != cols:
            fail(f'invalid TSV header in {path}: {rdr.fieldnames}')
        yield from rdr


def evaluate(args):
    prepared = Path(args.prepared_dir)
    meta = json.loads((prepared/'metadata.json').read_text())
    if meta.get('source_partition') != 'validation' or meta.get('protected_partition_accessed') is not False:
        fail('partition contract not satisfied')
    for name,field in [('requests.tsv','requests_sha256'),('truth.tsv','truth_sha256'),('evidence.tsv','evidence_sha256')]:
        if sha(prepared/name) != meta[field]:
            fail(f'input changed since preparation: {name}')
    catalog, keys, charge_counts = load_catalog(args.catalog)
    labels = {}
    for r in read_tsv(prepared/'truth.tsv',['query_id','truth_candidate_key','partition','evidence_source']):
        if r['partition'] != 'DEV' or r['query_id'] in labels or not r['evidence_source']:
            fail('invalid or duplicate DEV truth ledger entry')
        labels[r['query_id']] = r['truth_candidate_key']
    if len(labels) != meta['selected_queries']:
        fail('truth ledger length mismatch')
    qmeta = {}
    for r in read_tsv(prepared/'requests.tsv',REQ_COLS):
        qid = r['query_id']
        z = int(r['observed_charge']); mz = float(r['observed_precursor_mz'])
        if int(r['top_k']) != 256 or z<1 or not math.isfinite(mz):
            fail('request metadata invalid')
        if qid in qmeta:
            if qmeta[qid] != (z,mz):
                fail('request metadata changed within spectrum')
        else:
            qmeta[qid] = (z,mz)
    if qmeta.keys() != labels.keys():
        fail('query ID mismatch requests vs truth')
    grouped = {qid:[] for qid in labels}
    for r in read_tsv(args.hits,HIT_COLS):
        qid = r['query_id']
        if qid not in grouped:
            fail('unexpected query in hits')
        rank=int(r['rank']); index=int(r['candidate_index']); charge=int(r['charge'])
        if index >= len(catalog) or index<0 or rank != len(grouped[qid])+1:
            fail('candidate rank/index invalid or hits not grouped')
        if catalog[index]['key'] != r['key'] or charge != qmeta[qid][0] or int(catalog[index]['charge']) != charge:
            fail('candidate catalog key/index/charge mismatch')
        geom=float(r['geometry_score']); delta=float(r['absolute_neutral_mass_error']); neutral=float(r['theoretical_neutral_mass'])
        observed_neutral=charge*(qmeta[qid][1]-PROTON)
        if not all(map(math.isfinite, (geom,delta,neutral))) or delta<0 or abs(abs(neutral-observed_neutral)-delta)>1e-4:
            fail('hit scores or precursor mass inconsistent')
        grouped[qid].append((r['key'],geom,delta,neutral,index))
    metrics = Counter()
    details=[]
    for qid, target in labels.items():
        hits = grouped[qid]
        expected = min(256,charge_counts[qmeta[qid][0]])
        if len(hits) != expected:
            fail(f'truncated full candidate pool {qid}: {len(hits)} != {expected}')
        metrics['queries']+=1
        in_catalog=target in keys
        ranks = [i+1 for i,h in enumerate(hits) if h[0] == target]
        if len(ranks) > 1:fail('duplicate candidate in one pool')
        in_pool=bool(ranks)
        baseline_rank=ranks[0] if ranks else None
        metrics['truth_in_catalog']+=int(in_catalog)
        metrics['truth_in_pool']+=int(in_pool)
        metrics['legacy_top1']+=int(baseline_rank==1)
        metrics['legacy_top10']+=int(baseline_rank is not None and baseline_rank<=10)
        if baseline_rank:metrics['legacy_mrr']+=1/baseline_rank
        # Zero training/tuning; fixed diagnostic penalty under a 20ppm or 0.02Da scale.
        scored=[]
        for h in hits:
            scale=max(0.02, h[3]*20e-6)
            score=h[1]-0.02*math.log1p((h[2]/scale)**2)
            scored.append((score,h[4],h[0]))
        reranked=sorted(scored,key=lambda a:(-a[0],a[1]))
        candidate_rank=next((i+1 for i,s in enumerate(reranked) if s[2]==target),None)
        metrics['diagnostic_top1']+=int(candidate_rank==1)
        metrics['diagnostic_top10']+=int(candidate_rank is not None and candidate_rank<=10)
        if candidate_rank:metrics['diagnostic_mrr']+=1/candidate_rank
        metrics['changed_top1']+=int(hits[0][0] != reranked[0][2])
        metrics['corrected']+=int(baseline_rank!=1 and candidate_rank==1)
        metrics['regressed']+=int(baseline_rank==1 and candidate_rank!=1)
        details.append({'query_id':qid,'truth_in_catalog':int(in_catalog),'truth_in_pool':int(in_pool),
                        'legacy_rank':baseline_rank if baseline_rank else 'NA',
                        'diagnostic_rank':candidate_rank if candidate_rank else 'NA',
                        'legacy_top_key':hits[0][0],'diagnostic_top_key':reranked[0][2],
                        'legacy_top_mass_error_da':hits[0][2]})
    n=metrics['queries']
    results={'protocol':'ReDeeM_exploratory_external_validation_candidate_ranking_v1',
             'status':'PASS_EXTERNAL_DEVELOPMENT_DIAGNOSTIC',
             'source_provenance_status':meta['source_provenance_status'],
             'source_revision':meta['source_revision'],
             'external_test_accessed':False,'redeem_train_overlap':'NOT_CHECKED',
             'truth_type':'published_PSM_annotation_not_independent_sequence_synthesis_confirmation',
             'ptm_localization':'NOT_EVALUATED_unmodified_only',
             'catalog_sha256':sha(args.catalog),'hits_sha256':sha(args.hits),
             'requests_sha256':meta['requests_sha256'],'truth_sha256':meta['truth_sha256'],
             'queries':n,'truth_in_catalog':metrics['truth_in_catalog'],
             'catalog_coverage':metrics['truth_in_catalog']/n,
             'truth_in_frozen_256_pool':metrics['truth_in_pool'],
             'pool_recall_at_256':metrics['truth_in_pool']/n,
             'legacy_top1':metrics['legacy_top1']/n, 'legacy_top10':metrics['legacy_top10']/n,
             'legacy_mrr':metrics['legacy_mrr']/n,
             'diagnostic_top1':metrics['diagnostic_top1']/n,
             'diagnostic_top10':metrics['diagnostic_top10']/n,
             'diagnostic_mrr':metrics['diagnostic_mrr']/n,
             'changed_top1':metrics['changed_top1'],
             'diagnostic_corrected':metrics['corrected'],
             'diagnostic_regressed':metrics['regressed'],
             'diagnostic_mass_score':'geometry - 0.02 * ln(1 + (absolute_neutral_mass_error / max(0.02Da,20ppm*theoretical_neutral_mass))^2)',
             'diagnostic_gate':'NOT_APPROVED_FOR_PRODUCTION_OR_CLAIM_OF_INDEPENDENT_GAIN',
             'limitations':['PSM-derived labels, not direct synthetic pool assignments',
                 'Original ReDeeM TRAIN peptide overlap not checked',
                 'No modification localization tested',
                 'Mass penalty chosen a priori, exploratory, not calibrated for isotope/charge mistakes',
                 'Catalog excludes unknown sequences; no target forcing or catalog extension',
                 'External validation may contain systematic differences from frozen catalog; compare coverage first']}
    output=Path(args.output_prefix)
    if output.with_suffix('.json').exists() or Path(str(output)+'.queries.tsv').exists():
        fail('output already exists; do not overwrite')
    output.parent.mkdir(parents=True,exist_ok=True)
    dump(str(output)+'.json', results)
    with Path(str(output)+'.queries.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(details)
    print('REDEEM_EXTERNAL_EVALUATION=PASS queries=%d catalog=%d pool=%d legacy_top1=%d diagnostic_top1=%d' %
          (n,metrics['truth_in_catalog'],metrics['truth_in_pool'],metrics['legacy_top1'],metrics['diagnostic_top1']))
    print('REDEEM_EXTERNAL_RESULT=' + str(output)+'.json')
    return results


def run_codon(args):
    work=Path(args.workdir)
    if not work.is_dir():fail('work directory missing')
    sif=ROOT/SIF_REL; catalog=ROOT/CATALOG_REL
    if not sif.is_file() or sha(sif) != SIF_SHA:
        fail('published SIF missing or SHA mismatch')
    if not catalog.is_file() or sha(catalog) != CATALOG_SHA:
        fail('frozen catalog missing or SHA mismatch')
    meta=json.loads((work/'metadata.json').read_text())
    if meta['source_provenance_status'] != 'huggingface_hub_commit_pinned_validation' or meta['selected_queries'] != 512 or meta['scan_cap'] != 0:
        fail('real Codon run requires complete, 512-query immutable HF validation')
    if sha(work/'requests.tsv')!=meta['requests_sha256'] or sha(work/'truth.tsv')!=meta['truth_sha256'] or sha(work/'evidence.tsv')!=meta['evidence_sha256']:
        fail('transferred source hashes mismatch')
    job=os.environ.get('SLURM_JOB_ID','manual')
    result=work/f'run_{job}'
    if result.exists():fail('run output directory exists')
    result.mkdir()
    summary=result/'identifier_summary.tsv';hits=result/'identifier_hits.tsv'
    cmd=['singularity','exec','--bind',f'{ROOT}:{ROOT}',str(sif),'foundation_identify_stream_v0770',
         'identify-stream','--catalog',str(catalog),'--spectra',str(work/'requests.tsv'),
         '--output-hits',str(hits),'--output-summary',str(summary)]
    run=subprocess.run(cmd,check=False)
    if run.returncode != 0:
        fail(f'frozen v0770 identifier failed rc={run.returncode}; partial run retained {result}')
    result_data=evaluate(argparse.Namespace(prepared_dir=str(work),catalog=str(catalog),hits=str(hits),output_prefix=str(result/'evaluation')))
    print('REDEEM_EXTERNAL_CODON_JOB=PASS job_id='+job+' result='+str(result/'evaluation.json'))
    return result_data


def self_test():
    def fake(seq, idx, ch=2):
        return {'sequence':seq,'modified_sequence':seq,'precursor_charge':ch,'precursor_mz':500.0,
                'mz_array':[float(i+100) for i in range(20)],'intensity_array':[float(i+1) for i in range(20)],
                'experiment_name':'fixture','evidence_index':idx,'scan_number':100+idx}
    fixture=[fake('PEPTIDE',1),fake('PEPTIDK',2),fake('PEPTIDE',3),fake('PEPTIDR',4)]
    with tempfile.TemporaryDirectory() as d:
        base=Path(d)
        src=base/'source.jsonl'
        src.write_text(''.join(json.dumps(x)+'\n' for x in fixture))
        args=argparse.Namespace(source='jsonl',input_jsonl=str(src), output_dir=str(base/'prepared'),limit=2,seed=11,scan_cap=0)
        prepare(args)
        catalog={'schema':'redeem.foundation.practical_identifier.v0752','candidates':[
            {'key':s+'|z2','sequence':s,'charge':2} for s in ['PEPTIDE','PEPTIDK','PEPTIDR']]}
        import yaml
        cat=base/'catalog.yaml';cat.write_text(yaml.safe_dump(catalog))
        prepared=base/'prepared'
        labels=list(read_tsv(prepared/'truth.tsv',['query_id','truth_candidate_key','partition','evidence_source']))
        hit=base/'hits.tsv'
        with hit.open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(HIT_COLS)
            for lab in labels:
                for rank, (index,cand) in enumerate(enumerate(catalog['candidates']),1):
                    w.writerow([lab['query_id'],rank,index,cand['key'],2,2*(500.0-PROTON),0.0,1/rank])
        ev=evaluate(argparse.Namespace(prepared_dir=str(prepared),catalog=str(cat),hits=str(hit),output_prefix=str(base/'eval')))
        assert ev['queries']==2 and ev['pool_recall_at_256']==1 and ev['catalog_coverage']==1
        # Fail-closed protection: a truncated hit file is never treated as recall evidence.
        bad=base/'bad.tsv'
        with hit.open() as f:lines=f.readlines()
        bad.write_text(''.join([lines[0], lines[1], lines[4]]))
        try:
            evaluate(argparse.Namespace(prepared_dir=str(prepared),catalog=str(cat),hits=str(bad),output_prefix=str(base/'no')))
            raise AssertionError('truncated hits should fail')
        except ValueError:
            pass
        # Strict source split and source hash preservation.
        wrong=json.loads((prepared/'metadata.json').read_text());wrong['source_partition']='test'
        dump(prepared/'metadata.json',wrong)
        try:
            evaluate(argparse.Namespace(prepared_dir=str(prepared),catalog=str(cat),hits=str(hit),output_prefix=str(base/'no')))
            raise AssertionError('protected split should fail')
        except ValueError:
            pass
    print('REDEEM_EXTERNAL_DEV_SELFTEST=PASS unmodified_selection,provenance,full_pool,evaluation,no_test,nonoverwrite')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--source',choices=('hf','jsonl'),default='hf')
    prep.add_argument('--input-jsonl')
    prep.add_argument('--revision',default='main')
    prep.add_argument('--output-dir',required=True)
    prep.add_argument('--limit',type=int,default=512)
    prep.add_argument('--seed',type=int,default=20261022)
    prep.add_argument('--scan-cap',type=int,default=0,help='0 scans full validation')
    ev=sub.add_parser('evaluate')
    ev.add_argument('--prepared-dir',required=True)
    ev.add_argument('--catalog',required=True)
    ev.add_argument('--hits',required=True)
    ev.add_argument('--output-prefix',required=True)
    run=sub.add_parser('run-codon');run.add_argument('--workdir',required=True)
    sub.add_parser('self-test')
    args=ap.parse_args()
    try:
        if args.command=='prepare':prepare(args)
        elif args.command=='evaluate':evaluate(args)
        elif args.command=='run-codon':run_codon(args)
        elif args.command=='self-test':self_test()
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        print('REDEEM_EXTERNAL_DEV=BLOCKED '+str(exc),file=__import__('sys').stderr)
        raise SystemExit(2)

if __name__=='__main__':main()
