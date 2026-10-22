#!/usr/bin/env python3
"""Read-only explanation of exact-key coverage gaps in the frozen external DEV512 cohort.

Never changes, extends, or selects candidates. Does not produce truth labels.
Input FASTA and candidate catalog must match the previously pinned SHA256.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

REVISION = 'ffde6e4a2925f94dc28c87e33917db19a28f5869'
PIN = {
    'fasta': 'dffdff3807ec6eb7e752035b5475aaefa26d1b36e513e3344198d521a2169c42',
    'catalog': '9d67664416c23ccc51f27397512f7965eade9ce4b8156ff4bc60dcbb2a26f2ef',
    'truth': '614d00425c1bae2ce20f1363af8bfb03c251d7fd7638ce4eeb9fe1e915823cd5',
    'requests': '6ccc96815bd8c016abd91a8106dd584368b1d17ff8e121c0c0b561142fcdba3b',
}
TYPES = (
    'IN_FROZEN_REFERENCE_CATALOG',
    'EXACT_PROTEOME_MATCH_OVER_2_MISSED_CLEAVAGES',
    'EXACT_PROTEOME_MATCH_SEMITRYPTIC',
    'EXACT_PROTEOME_MATCH_NONTRYPTIC',
    'ONLY_IL_EQUIVALENT_CANONICAL_SEQUENCE',
    'NOT_PRESENT_IN_REVIEWED_CANONICAL_PROTEOME',
)
KEY_RE = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY]{7,64})\|z([23])$')
TRUTH_FIELDS = ['query_id','truth_candidate_key','partition','evidence_source']


def digest_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sequences(path: Path) -> list[str]:
    proteins=[]; current=[]; seen=False
    for original in path.open(encoding='utf-8'):
        line=original.strip()
        if not line: continue
        if line.startswith('>'):
            if seen: proteins.append(''.join(current))
            current=[];seen=True
        else:
            if not seen: raise ValueError('FASTA content before header')
            current.append(line.upper())
    if seen: proteins.append(''.join(current))
    if not proteins or any(not p for p in proteins):
        raise ValueError('empty protein sequence or FASTA')
    return proteins


def cleavage_positions(seq: str) -> set[int]:
    pos={0,len(seq)}
    for i,aa in enumerate(seq):
        if aa in 'KR' and (i+1==len(seq) or seq[i+1]!='P'):
            pos.add(i+1)
    return pos


def exact_match_category(peptide: str, proteins: list[str], cuts: list[set[int]]) -> str | None:
    # A fully tryptic peptide with 0--2 missed cleavages MUST have been cataloged.
    # If such a peptide is absent, input/build equivalence is broken: fail closed.
    best=0
    for prot,sites in zip(proteins,cuts):
        offset=0
        while True:
            index=prot.find(peptide,offset)
            if index < 0: break
            end=index+len(peptide)
            start_ok=index in sites
            end_ok=end in sites
            if start_ok and end_ok:
                missed=sum(index < s < end for s in sites)
                if missed<=2:
                    raise ValueError('catalog mismatch: fully tryptic eligible exact peptide absent')
                return TYPES[1]
            if start_ok or end_ok:
                best=max(best,2)
            else:
                best=max(best,1)
            offset=index+1
    if best==2:return TYPES[2]
    if best==1:return TYPES[3]
    return None


def il_equivalent(peptide: str, normalized_proteins: list[str]) -> bool:
    q=peptide.replace('I','L')
    return any(q in protein for protein in normalized_proteins)


def read_truth(path: Path) -> list[dict[str,str]]:
    with path.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f,delimiter='\t')
        if r.fieldnames!=TRUTH_FIELDS:
            raise ValueError('DEV truth header mismatch')
        rows=list(r)
    if len(rows)!=512:raise ValueError('expected exactly 512 DEV annotations')
    qids=set();keys=set()
    for row in rows:
        m=KEY_RE.fullmatch(row['truth_candidate_key'])
        if row['partition']!='DEV' or not m or not row['evidence_source'] or not row['query_id']:
            raise ValueError('invalid DEV reference row')
        if row['query_id'] in qids or row['truth_candidate_key'] in keys:
            raise ValueError('duplicate DEV query or peptide-charge reference')
        qids.add(row['query_id']);keys.add(row['truth_candidate_key'])
    return rows


def catalog_intersection(path: Path, targets: set[str]) -> set[str]:
    present=set()
    with path.open(encoding='utf-8') as f:
        if f.readline().strip()!='schema: redeem.foundation.practical_identifier.v0752' or f.readline().strip()!='candidates:':
            raise ValueError('unexpected catalog schema')
        for line in f:
            if line.startswith('  - key: '):
                key=line[9:].strip()
                if key in targets:
                    if key in present:raise ValueError('duplicate matching catalog key')
                    present.add(key)
    return present


def diagnose_rows(truth: list[dict[str,str]], present: set[str], proteins: list[str]) -> list[dict[str,str]]:
    cuts=[cleavage_positions(p) for p in proteins]
    norm=[p.replace('I','L') for p in proteins]
    result=[]
    for row in truth:
        key=row['truth_candidate_key'];match=KEY_RE.fullmatch(key)
        if match is None:raise ValueError('invalid reference key')
        peptide,charge=match.groups()
        if key in present:
            category=TYPES[0]
        else:
            category=exact_match_category(peptide, proteins, cuts)
            if category is None:
                category=TYPES[4] if il_equivalent(peptide,norm) else TYPES[5]
        result.append({'query_id':row['query_id'],'reference_key':key,'charge':charge,
            'sequence_length':str(len(peptide)),'classification':category,
            'exact_catalog_member':'YES' if key in present else 'NO'})
    return result


def run(fasta: Path,catalog_dir: Path,cohort: Path,output_prefix: Path,override:dict[str,str]|None=None):
    if not output_prefix.parent.is_dir():raise ValueError('output parent missing')
    output_json=Path(str(output_prefix)+'.json')
    output_tsv=Path(str(output_prefix)+'.tsv')
    if output_json.exists() or output_tsv.exists():raise ValueError('output already exists; refuse overwrite')
    catalog=catalog_dir/'catalog.yaml'
    paths={'fasta':fasta,'catalog':catalog,'truth':cohort/'truth.tsv','requests':cohort/'requests.tsv'}
    pins=override if override is not None else PIN
    hashes={}
    for name,p in paths.items():
        if not p.is_file() or not p.stat().st_size:raise ValueError(f'missing {name}: {p}')
        hashes[name]=digest_sha(p)
        if hashes[name]!=pins[name]:raise ValueError(f'pin mismatch: {name}')
    prov=json.loads((catalog_dir/'catalog_provenance.json').read_text())
    meta=json.loads((cohort/'metadata.json').read_text())
    cov=json.loads((catalog_dir/'coverage.json').read_text())
    if (prov.get('fasta_sha256')!=hashes['fasta'] or prov.get('catalog_sha256')!=hashes['catalog']
        or prov.get('candidate_source_uses_truth') is not False or prov.get('candidate_source_uses_requests') is not False
        or prov.get('candidate_count')!=3375146 and override is None):
        raise ValueError('catalog provenance does not agree with frozen source')
    if (meta.get('source_partition')!='validation' or meta.get('source_revision')!=REVISION
        or meta.get('selected_queries')!=512 or meta.get('protected_partition_accessed') is not False):
        raise ValueError('not the permitted pinned external validation cohort')
    if (cov.get('status')!='PASS_INDEPENDENT_CATALOG_COVERAGE' or cov.get('reference_in_catalog')!=239 and override is None
        or cov.get('catalog_sha256')!=hashes['catalog'] or cov.get('truth_sha256')!=hashes['truth']
        or cov.get('requests_sha256')!=hashes['requests']
        or cov.get('next_step')!='STOP_NO_SCORING_RUN_BELOW_50_PERCENT'):
        raise ValueError('coverage report/provenance mismatch')
    truth=read_truth(cohort/'truth.tsv')
    present=catalog_intersection(catalog,{r['truth_candidate_key'] for r in truth})
    if len(present)!=cov['reference_in_catalog']:
        raise ValueError('catalog coverage does not reproduce prior report')
    proteome=sequences(fasta)
    if override is None and len(proteome)!=20416:
        raise ValueError('reference FASTA protein count changed')
    rows=diagnose_rows(truth,present,proteome)
    counts=Counter(r['classification'] for r in rows)
    by_charge={z:{'total':sum(r['charge']==z for r in rows),
         'present':sum(r['charge']==z and r['exact_catalog_member']=='YES' for r in rows),
         'missing_by_reason':dict(sorted(Counter(r['classification'] for r in rows if r['charge']==z and r['exact_catalog_member']=='NO').items()))} for z in ('2','3')}
    summary={'status':'PASS_READ_ONLY_REFERENCE_COVERAGE_TAXONOMY',
         'protocol':'redeem_external_pt_validation512_reference_gap_v1',
         'truth_scope':'published_validation_PSM_annotations_not_orthogonal_identity',
         'reference_catalog_policy':'human_canonical_reviewed_trypsin_0to2missed_unmodified_noC_z2z3',
         'total_queries':len(rows),'reference_in_catalog':len(present),
         'missing_references':len(rows)-len(present),
         'missing_by_reason':{k:counts[k] for k in TYPES[1:]},
         'by_charge':by_charge,'proteins_scanned':len(proteome),'verified_sha256':hashes,
         'catalog_generation_uses_labels':False,'analysis_uses_validation_labels':True,
         'candidate_keys_modified':False,'new_identification_run':False,
         'coverage_gate':'STOP_BELOW_PREDECLARED_50_PERCENT',
         'next_decision':'CHOOSE_LABEL_BLIND_LIBRARY_OR_DIGEST_POLICY_ON_NEW_DISJOINT_DEV_BEFORE_RANKING',
         'limits':['Post hoc label-based error categories; do not tune new catalog against this DEV512 and report it as an independent evaluation.',
         'I/L-equivalent sequence presence does not establish exact-key catalog coverage.',
         'A reviewed canonical human reference FASTA can exclude synthetic/alternative/isoform sequences.',
         'Only protein sequence and enzymatic context are classified, not spectra or mass-score failures.',
         'Protected TEST/HOLDOUT partitions not accessed.']}
    # Compute everything and validate totals BEFORE touching the output.
    if sum(counts.values())!=512 or sum(summary['missing_by_reason'].values())!=len(rows)-len(present):
        raise ValueError('classification totals inconsistent')
    output_tsv.write_text('query_id\treference_key\tcharge\tsequence_length\tclassification\texact_catalog_member\n'+
        ''.join('\t'.join(r[k] for k in ('query_id','reference_key','charge','sequence_length','classification','exact_catalog_member'))+'\n' for r in rows))
    output_json.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print('REDEEM_REFERENCE_GAP=PASS references=%d covered=%d missing=%d' % (len(rows),len(present),len(rows)-len(present)))
    print('REDEEM_REFERENCE_GAP_REPORT='+str(output_json))
    return summary


def self_test():
    # Correct cleavage classification and invariants. Independent of original dataset.
    seq='MPEPTIDERKAAAAAARKQQQQQQK'
    assert list(sorted(cleavage_positions(seq)))==[0,9,10,17,18,25]
    prot=[seq]
    assert exact_match_category('PEPTIDER',prot,[cleavage_positions(seq)])==TYPES[2]
    assert exact_match_category('PEPTIDE',prot,[cleavage_positions(seq)])==TYPES[3]
    assert exact_match_category('MPEPTIDERKAAAAAARKQQQQQQK',prot,[cleavage_positions(seq)])==TYPES[1]
    assert il_equivalent('PEPTIDELK',['MPEPTIDEIK'.replace('I','L')])
    assert not il_equivalent('GGGGGGGG',['MPEPTIDEIK'.replace('I','L')])
    try:
        exact_match_category('MPEPTIDER',prot,[cleavage_positions(seq)])
        raise AssertionError('missed failure for absent fully tryptic catalog peptide')
    except ValueError:pass
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);cat=root/'catalog.yaml'
        cat.write_text('schema: redeem.foundation.practical_identifier.v0752\ncandidates:\n  - key: MPEPTIDER|z2\n    sequence: MPEPTIDER\n    charge: 2\n')
        assert catalog_intersection(cat,{'MPEPTIDER|z2','PEPTIDER|z2'})=={'MPEPTIDER|z2'}
        assert diagnose_rows([{'query_id':'q1','truth_candidate_key':'MPEPTIDER|z2'},
          {'query_id':'q2','truth_candidate_key':'PEPTIDER|z2'}],{'MPEPTIDER|z2'},prot)[1]['classification']==TYPES[2]
    # Full 512-query synthetic run tests outputs, checksum gates and overwrite rejection.
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        fasta=root/'ref.fasta';fasta.write_text('>reviewed-human-fixture\n'+seq+'\n')
        catdir=root/'cat';catdir.mkdir()
        (catdir/'catalog.yaml').write_text('schema: redeem.foundation.practical_identifier.v0752\ncandidates:\n  - key: MPEPTIDER|z2\n    sequence: MPEPTIDER\n    charge: 2\n')
        cohort=root/'cohort';cohort.mkdir()
        aa='ACDEFGHIKLMNPQRSTVWY'
        def code(number):
            letters=[]
            for _ in range(6):
                letters.append(aa[number%20]);number//=20
            return 'Q'+''.join(letters)
        with (cohort/'truth.tsv').open('w',newline='') as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n')
            writer.writerow(TRUTH_FIELDS)
            for idx in range(512):
                pep='MPEPTIDER' if idx==0 else 'PEPTIDER' if idx==1 else code(idx)
                writer.writerow((str(idx),f'{pep}|z2','DEV','synthetic_fixture'))
        (cohort/'requests.tsv').write_text('synthetic requests placeholder\n')
        (cohort/'metadata.json').write_text(json.dumps({'source_partition':'validation','source_revision':REVISION,
            'selected_queries':512,'protected_partition_accessed':False}))
        pins={'fasta':digest_sha(fasta),'catalog':digest_sha(catdir/'catalog.yaml'),
              'truth':digest_sha(cohort/'truth.tsv'),'requests':digest_sha(cohort/'requests.tsv')}
        (catdir/'catalog_provenance.json').write_text(json.dumps({
            'fasta_sha256':pins['fasta'],'catalog_sha256':pins['catalog'],
            'candidate_source_uses_truth':False,'candidate_source_uses_requests':False,'candidate_count':1}))
        (catdir/'coverage.json').write_text(json.dumps({'status':'PASS_INDEPENDENT_CATALOG_COVERAGE',
            'reference_in_catalog':1,'catalog_sha256':pins['catalog'],'truth_sha256':pins['truth'],
            'requests_sha256':pins['requests'],'next_step':'STOP_NO_SCORING_RUN_BELOW_50_PERCENT'}))
        prefix=root/'diagnostic'
        summary=run(fasta,catdir,cohort,prefix,override=pins)
        assert summary['total_queries']==512 and summary['reference_in_catalog']==1
        assert summary['missing_by_reason'][TYPES[2]]==1
        assert summary['missing_references']==511
        assert Path(str(prefix)+'.json').exists() and Path(str(prefix)+'.tsv').exists()
        try:
            run(fasta,catdir,cohort,prefix,override=pins)
            raise AssertionError('existing outputs were overwritten')
        except ValueError as e:
            assert 'exists' in str(e)
        (cohort/'requests.tsv').write_text('changed requests\n')
        try:
            run(fasta,catdir,cohort,root/'blocked',override=pins)
            raise AssertionError('modified requests were accepted')
        except ValueError as e:
            assert 'pin mismatch' in str(e)
        assert not (root/'blocked.json').exists()
    print('REDEEM_REFERENCE_GAP_SELFTEST=PASS cleavage_categories,il_equivalence,catalog_reproduction,dev512_end_to_end,sha_guard,nonoverwrite')


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('self-test')
    p=sub.add_parser('run')
    p.add_argument('--fasta',required=True,type=Path)
    p.add_argument('--catalog-dir',required=True,type=Path)
    p.add_argument('--cohort',required=True,type=Path)
    p.add_argument('--out-prefix',required=True,type=Path)
    args=parser.parse_args()
    if args.mode=='self-test':self_test()
    else:run(args.fasta,args.catalog_dir,args.cohort,args.out_prefix)

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,AssertionError) as exc:
        print('REDEEM_REFERENCE_GAP=BLOCKED '+str(exc))
        raise SystemExit(2)
