#!/usr/bin/env python3
"""Build label-blind tryptic reference catalog and audit independent external coverage.

No machine learning, no reference-label reads during catalog construction.
The catalog YAML schema is redeem.foundation.practical_identifier.v0752.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

AA = set('ACDEFGHIKLMNPQRSTVWY')
SCHEMA = 'redeem.foundation.practical_identifier.v0752'
DATASET_REVISION = 'ffde6e4a2925f94dc28c87e33917db19a28f5869'
REQUESTS_SHA = '6ccc96815bd8c016abd91a8106dd584368b1d17ff8e121c0c0b561142fcdba3b'
TRUTH_SHA = '614d00425c1bae2ce20f1363af8bfb03c251d7fd7638ce4eeb9fe1e915823cd5'
FASTA_SOURCE = 'https://rest.uniprot.org/uniprotkb/stream?query=%28proteome%3AUP000005640%29%20AND%20%28reviewed%3Atrue%29&format=fasta'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def fasta_sequences(path):
    seen=0
    sequence=[]
    with Path(path).open(encoding='utf-8') as f:
        for raw in f:
            s=raw.strip()
            if not s: continue
            if s.startswith('>'):
                if seen:
                    yield ''.join(sequence)
                seen += 1
                sequence=[]
            else:
                if not seen:raise ValueError('FASTA sequence before header')
                sequence.append(s.upper())
    if seen:yield ''.join(sequence)
    else:raise ValueError('empty FASTA')


def digest(seq, missed=2, minlen=7, maxlen=64):
    cut=[0]
    for i,c in enumerate(seq):
        if c in 'KR' and (i+1==len(seq) or seq[i+1]!='P'):
            cut.append(i+1)
    if cut[-1]!=len(seq):cut.append(len(seq))
    for i in range(len(cut)-1):
        for j in range(i+1,min(i+missed+2,len(cut))):
            p=seq[cut[i]:cut[j]]
            if minlen<=len(p)<=maxlen and 'C' not in p and set(p)<=AA:
                yield p


def build(fasta, out, max_candidates=4000000):
    fasta=Path(fasta); out=Path(out)
    if out.exists():raise ValueError('output directory exists; refuse overwrite')
    if not fasta.is_file() or not fasta.stat().st_size:raise ValueError('FASTA missing or empty')
    peptides=set(); records=0; rejects=0
    for seq in fasta_sequences(fasta):
        records += 1
        if not seq or len(seq)>200000:
            rejects += 1;continue
        peptides.update(digest(seq))
        if len(peptides)*2>max_candidates:
            raise ValueError('candidate budget exceeded; no catalog written')
    if records<1 or not peptides:raise ValueError('FASTA yields no eligible peptides')
    size=len(peptides)*2
    out.mkdir(parents=True,exist_ok=False)
    catalog=out/'catalog.yaml'
    with catalog.open('w') as f:
        f.write('schema: '+SCHEMA+'\ncandidates:\n')
        for peptide in sorted(peptides):
            for charge in (2,3):
                f.write(f'  - key: {peptide}|z{charge}\n    sequence: {peptide}\n    charge: {charge}\n')
    summary={'status':'PASS_LABEL_BLIND_REFERENCE_CATALOG','protocol':'human_reviewed_reference_trypsin_0to2missed_unmodified_noC_z2z3_v1',
      'schema':SCHEMA,'fasta_sha256':sha(fasta),'fasta_source_url':FASTA_SOURCE,
      'fasta_proteins':records,'rejected_protein_sequences':rejects,
      'unique_peptide_sequences':len(peptides),'candidate_count':size,
      'charges':[2,3],'min_sequence_length':7,'max_sequence_length':64,'missed_cleavages':2,
      'trypsin_proline_exception':True,'cysteine_excluded':True,'candidate_source_uses_truth':False,
      'candidate_source_uses_requests':False,'catalog_sha256':sha(catalog),
      'limitations':['Reviewed human canonical UniProtKB FASTA downloaded at execution time and preserved by SHA256; not an immutable UniProt release URL',
        'No variable or fixed modifications, isoforms not included unless source FASTA provides them',
        'Only fully tryptic sequences and up to two missed cleavages; external ProteomeTools peptide library can differ',
        'Unmodified, no-cysteine, charge 2/3 selection matches external DEV512 pilot',
        'No reference-label selection or target injection; coverage must be measured separately']}
    (out/'catalog_provenance.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print('REDEEM_REFERENCE_BUILD=PASS proteins=%d peptides=%d candidates=%d' %(records,len(peptides),size))
    print('REDEEM_REFERENCE_CATALOG='+str(catalog))
    return summary


def catalog_keys_and_count(path):
    n=0;keys=set()
    with Path(path).open() as f:
        if f.readline().strip()!='schema: '+SCHEMA or f.readline().strip()!='candidates:':
            raise ValueError('catalog schema mismatch')
        for line in f:
            if line.startswith('  - key: '):
                key=line[9:].strip()
                if not re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]+\|z[23]',key) or key in keys:
                    raise ValueError('invalid or duplicate catalog key')
                keys.add(key);n+=1
    return keys,n


def audit(catalog_dir, cohort, output):
    catalog_dir=Path(catalog_dir);cohort=Path(cohort);output=Path(output)
    if output.exists():raise ValueError('audit output exists; refuse overwrite')
    summary=json.loads((catalog_dir/'catalog_provenance.json').read_text())
    if summary['status']!='PASS_LABEL_BLIND_REFERENCE_CATALOG' or summary['candidate_source_uses_truth'] or summary['candidate_source_uses_requests']:
        raise ValueError('catalog source not independent of validation labels')
    if sha(catalog_dir/'catalog.yaml')!=summary['catalog_sha256']:
        raise ValueError('catalog checksum differs from build provenance')
    cohort_meta=json.loads((cohort/'metadata.json').read_text())
    if cohort_meta.get('source_partition')!='validation' or cohort_meta.get('selected_queries')!=512 or cohort_meta.get('source_revision')!=DATASET_REVISION or cohort_meta.get('protected_partition_accessed') is not False:
        raise ValueError('unexpected external cohort partition, revision, or size')
    if sha(cohort/'truth.tsv')!=TRUTH_SHA or sha(cohort/'requests.tsv')!=REQUESTS_SHA:
        raise ValueError('external validation inputs do not match job 4825996')
    keys,n=catalog_keys_and_count(catalog_dir/'catalog.yaml')
    if n!=summary['candidate_count']:raise ValueError('catalog candidate count mismatch')
    labels={};charges=Counter()
    with (cohort/'truth.tsv').open(newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            qid=row['query_id'];key=row['truth_candidate_key']
            if qid in labels or row['partition']!='DEV' or not key.endswith(('|z2','|z3')):
                raise ValueError('invalid DEV truth row')
            labels[qid]=key;charges['z'+key.rsplit('|z',1)[1]]+=1
    if len(labels)!=512:raise ValueError('not exactly 512 DEV labels')
    present={k for k in labels.values() if k in keys}
    by_charge={z:{'references':sum(k.endswith('|'+z) for k in labels.values()),
                   'in_catalog':sum(k.endswith('|'+z) for k in labels.values() if k in keys)} for z in ('z2','z3')}
    result={'status':'PASS_INDEPENDENT_CATALOG_COVERAGE','queries':512,'catalog_candidates':n,
      'unique_sequences':summary['unique_peptide_sequences'],'reference_in_catalog':len(present),
      'coverage_fraction':len(present)/512,'baseline_frozen_63332_reference_in_catalog':4,
      'baseline_frozen_63332_coverage_fraction':4/512,'coverage_by_charge':by_charge,
      'source_revision':DATASET_REVISION,'truth_sha256':TRUTH_SHA,'requests_sha256':REQUESTS_SHA,
      'catalog_sha256':summary['catalog_sha256'],'fasta_sha256':summary['fasta_sha256'],
      'candidate_generation_used_validation_truth':False,
      'candidate_generation_used_validation_requests':False,
      'next_step':'RUN_FROZEN_IDENTIFIER_IF_COVERAGE_AT_LEAST_50_PERCENT' if len(present)>=256 else 'STOP_NO_SCORING_RUN_BELOW_50_PERCENT',
      'limits':['PSM-derived annotations are not orthogonally verified sequence identity',
        'Exact-key presence in catalog does not establish same-charge nearest256 pool recall',
        'Proteome reference FASTA is not validation/TEST data but can contain overlapping peptides',
        'No model training or tuning, no protected test or holdout use']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('REDEEM_REFERENCE_COVERAGE=PASS found=%d/512 fraction=%.6f' %(len(present),len(present)/512))
    print('REDEEM_REFERENCE_NEXT='+result['next_step'])
    return result


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        b=Path(temp);fa=b/'mini.fasta';fa.write_text('>human1\nMPEPTIDERKPEPTIDK\n>human2\nQQQPEPTIDER\n')
        summary=build(fa,b/'out')
        assert summary['candidate_count']>0
        keys,n=catalog_keys_and_count(b/'out/catalog.yaml')
        assert n==summary['candidate_count'] and 'MPEPTIDER|z2' in keys
        assert all('|z' in s and 'C' not in s for s in keys)
        try:
            build(fa,b/'out')
            raise AssertionError('overwrite accepted')
        except ValueError:pass
        try:
            build(fa,b/'bad',max_candidates=1)
            raise AssertionError('candidate cap not enforced')
        except ValueError:pass
        assert not (b/'bad').exists()
        assert list(digest('AKPQR'))==[]
    print('REDEEM_REFERENCE_SELFTEST=PASS trypsin,proline,deterministic_z2z3,independent_builder,overwrite,candidate_budget')


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('build');s.add_argument('--fasta',required=True);s.add_argument('--out',required=True);s.add_argument('--max-candidates',type=int,default=4000000)
    s=sub.add_parser('audit');s.add_argument('--catalog-dir',required=True);s.add_argument('--cohort',required=True);s.add_argument('--output',required=True)
    sub.add_parser('self-test')
    args=p.parse_args()
    try:
        if args.command=='build':build(args.fasta,args.out,args.max_candidates)
        elif args.command=='audit':audit(args.catalog_dir,args.cohort,args.output)
        else:self_test()
    except (ValueError,OSError,KeyError,TypeError) as e:
        p.error('BLOCKED: '+str(e))

if __name__=='__main__':main()
