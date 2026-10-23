#!/usr/bin/env python3
"""Source-defined ProteomeTools multi-sheet catalog and disjoint validation cohort.

The catalog builder consumes only a published peptide-pool mapping workbook.
It never reads spectral intensities, prior/new validation labels or predictions.
The validation selector reads ONLY a pinned Hugging Face validation partition;
the old DEV512 identity keys are used only for disjointness, never catalog design.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

NS={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'p':'http://schemas.openxmlformats.org/package/2006/relationships'}
IDENTITY_COLS={'sequence','peptidesequence','unmodifiedsequence','aminoacidsequence','peptideaa','peptideseq','sequenceaa'}
SCORE_COLS={'identification','psm','andromeda','intensity','ms2','observedspectrum','maxquant'}
ALPHABET=set('ACDEFGHIKLMNPQRSTVWY')
SCHEMA='redeem.foundation.practical_identifier.v0752'
DATASET='InstaDeepAI/ms_proteometools'
DATASET_SHA='ffde6e4a2925f94dc28c87e33917db19a28f5869'
OLD_TRUTH_SHA='614d00425c1bae2ce20f1363af8bfb03c251d7fd7638ce4eeb9fe1e915823cd5'
OLD_REQUEST_SHA='6ccc96815bd8c016abd91a8106dd584368b1d17ff8e121c0c0b561142fcdba3b'
# Cohort deliberately restricted to initial 2017 ProteomeTools synthesis pools.
ALLOWED_POOLS={'TUM_first_pool','TUM_second_pool','SRMAtlas','TUM_isoform',
               'TUM_second_addon','TUM_proteo_TMT'}
SOURCE_URL='https://www.nature.com/articles/nmeth.4153'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


def norm(value):
    return re.sub(r'[^a-z0-9]','',str(value).lower())


def excel_column(addr):
    m=re.match(r'([A-Z]+)[0-9]+$',addr)
    if not m:return None
    i=0
    for x in m[1]:i=i*26+ord(x)-64
    return i-1


def workbook_sheets(archive):
    root=ET.fromstring(archive.read('xl/workbook.xml'))
    relroot=ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
    rels={r.attrib['Id']:r.attrib['Target'] for r in relroot}
    result=[]
    for s in root.findall('.//x:sheets/x:sheet',NS):
        target=rels[s.attrib[f'{{{NS["r"]}}}id']]
        target=target.lstrip('/') if target.startswith('/') else (target if target.startswith('xl/') else 'xl/'+target)
        # Reject path traversal and external relationships.
        from posixpath import normpath
        target=normpath(target)
        if target.startswith('../') or target not in archive.namelist():
            raise ValueError('unsafe or missing worksheet path')
        result.append((s.attrib['name'],target))
    return result


def workbook_shared_strings(archive):
    path='xl/sharedStrings.xml'
    if path not in archive.namelist():return []
    result=[]
    with archive.open(path) as handle:
        for event,elem in ET.iterparse(handle,events=('end',)):
            if elem.tag==f'{{{NS["x"]}}}si':
                result.append(''.join((t.text or '') for t in elem.iter(f'{{{NS["x"]}}}t')))
                elem.clear()
    return result


def sheet_rows(archive, sheet_path, shared, max_rows=0):
    count=0
    with archive.open(sheet_path) as handle:
        for event,elem in ET.iterparse(handle,events=('end',)):
            if elem.tag!=f'{{{NS["x"]}}}row':continue
            cells={}
            for c in elem.findall('x:c',NS):
                col=excel_column(c.attrib.get('r',''))
                if col is None:continue
                kind=c.attrib.get('t')
                if kind=='inlineStr':
                    value=''.join((t.text or '') for t in c.iter(f'{{{NS["x"]}}}t'))
                else:
                    node=c.find('x:v',NS)
                    value=node.text if node is not None and node.text is not None else ''
                    if kind=='s' and value:value=shared[int(value)]
                cells[col]=str(value).strip()
            if cells:yield cells
            elem.clear()
            count+=1
            if max_rows and count>=max_rows:break


def discover(path):
    path=Path(path)
    if not path.is_file() or not zipfile.is_zipfile(path):raise ValueError('source is not an XLSX workbook')
    found=[]
    with zipfile.ZipFile(path) as z:
        shared=workbook_shared_strings(z)
        for sheet,ref in workbook_sheets(z):
            suggested=[];samples=[]
            for offset,row in enumerate(sheet_rows(z,ref,shared,max_rows=35),1):
                if offset<=5:samples.append({'row':offset,'cells':{str(k):v[:110] for k,v in list(row.items())[:18]}})
                matches=[(c,val) for c,val in row.items() if norm(val) in IDENTITY_COLS]
                if matches:
                    suggested.append({'sheet':sheet,'sheet_xml':ref,'header_row':offset,
                                      'sequence_columns':[{'index':col,'name':label} for col,label in matches],
                                      'header':{str(k):v for k,v in row.items()},
                                      'has_possible_score_columns':any(any(w in norm(v) for w in SCORE_COLS) for v in row.values())})
            found.append({'sheet':sheet,'sheet_xml':ref,'first_five_rows':samples,'candidate_headers':suggested})
    return found


def select_header(path, sheet=None, column=None):
    entries=discover(path)
    if sheet:entries=[x for x in entries if x['sheet']==sheet]
    options=[(x,c) for e in entries for x in e['candidate_headers'] for c in x['sequence_columns']
             if not column or norm(c['name'])==norm(column)]
    # Explicit column permits a differently named source sequence field but
    # still needs unambiguous worksheet and exact header text.
    if column and not options:
        with zipfile.ZipFile(path) as z:
            shared=workbook_shared_strings(z)
            for e in entries:
                for offset,row in enumerate(sheet_rows(z,e['sheet_xml'],shared,max_rows=35),1):
                    for idx,value in row.items():
                        if norm(value)==norm(column):
                            options.append(({'sheet':e['sheet'],'sheet_xml':e['sheet_xml'],
                              'header_row':offset,'header':{str(k):v for k,v in row.items()}},
                              {'index':idx,'name':value}))
    if len(options)!=1:
        raise ValueError(f'expected exactly one sequence column in source pool workbook; found {len(options)}. Inspect source and select explicit --sheet/--column; no catalog written.')
    return options[0]


# The 2017 publication workbook is a source list, not an evaluation hit file.
# This named policy is fixed by its source structure, not selected by DEV labels.
PT2017_SYNTHESIS_SHEETS = (
    'Proteotypic Set',
    'Missing Gene Set',
    'SRMAtlas Set',
)
PT2017_SHEET_SET = 'pt2017-synthesis-v1'
# SHA supplied by the user's successful 2026-10-23 workbook inspection.
PT2017_INSPECTED_WORKBOOK_SHA256 = '6d5a1134bee97644cc3197f4a02e1eb9448cbcd1674f616a46bc385106ffbf17'


def source_sheet_name(name):
    # The publisher workbook sometimes includes literal surrounding quotes.
    name = name.strip()
    if len(name) >= 2 and name.startswith('"') and name.endswith('"'):
        name = name[1:-1].strip()
    return name


def select_synthesis_sheets(path):
    entries = discover(path)
    result = []
    for requested in PT2017_SYNTHESIS_SHEETS:
        matches = [entry for entry in entries if source_sheet_name(entry['sheet']) == requested]
        if len(matches) != 1:
            raise ValueError(f'synthesis source sheet {requested!r}: expected one match, found {len(matches)}; no catalog written')
        entry = matches[0]
        options = [(header, col) for header in entry['candidate_headers']
                   for col in header['sequence_columns']
                   if norm(col['name']) == 'sequence']
        if len(options) != 1:
            raise ValueError(f'synthesis source sheet {requested!r}: expected exactly one Sequence column; no catalog written')
        header, col = options[0]
        expected_header = {0: 'poolname', 1: 'setname', 2: 'sequence'}
        actual_header = {int(k): norm(v) for k, v in header['header'].items()}
        if (header['header_row'] != 1 or col['index'] != 2 or
                any(actual_header.get(k) != value for k,value in expected_header.items()) or
                header['has_possible_score_columns']):
            raise ValueError(f'synthesis source sheet {requested!r}: unexpected header layout or score fields; no catalog written')
        result.append((header,col))
    return result


def build(source,out,sheet=None,column=None,max_candidates=3000000,sheet_set=None,expected_source_sha256=None):
    source=Path(source);out=Path(out)
    if out.exists():raise ValueError('output already exists; refuse overwrite')
    if expected_source_sha256 is not None and sha(source) != expected_source_sha256:
        raise ValueError('workbook SHA256 differs from inspected source; no catalog written')
    if sheet_set is not None:
        if sheet_set != PT2017_SHEET_SET or sheet is not None or column is not None:
            raise ValueError('unknown sheet-set or conflicting single-sheet option; no catalog written')
        selected = select_synthesis_sheets(source)
    else:
        selected = [select_header(source,sheet,column)]
    peptides=set();stats=Counter();per_sheet=[]
    with zipfile.ZipFile(source) as z:
        shared=workbook_shared_strings(z)
        for selection,col in selected:
            local=Counter();source_values=set()
            for offset,row in enumerate(sheet_rows(z,selection['sheet_xml'],shared),1):
                if offset<=selection['header_row']:continue
                stats['source_rows']+=1;local['source_rows']+=1
                value=row.get(col['index'],'').strip().upper()
                # No PTM inference. Bare unmodified AA sequences only.
                if not value:stats['empty']+=1;local['empty']+=1;continue
                if not 7<=len(value)<=64 or not set(value)<=ALPHABET:
                    stats['invalid_or_modified']+=1;local['invalid_or_modified']+=1;continue
                if 'C' in value:
                    stats['cysteine_excluded']+=1;local['cysteine_excluded']+=1;continue
                source_values.add(value)
                if value in peptides:local['duplicate_existing']+=1
                peptides.add(value)
                if 2*len(peptides)>max_candidates:
                    raise ValueError('catalog exceeds predeclared size budget; no output written')
            if local['source_rows']<10:
                raise ValueError(f'synthesis source sheet {selection["sheet"]!r} has fewer than 10 rows; no catalog written')
            per_sheet.append({'name':selection['sheet'], 'canonical_name':source_sheet_name(selection['sheet']),
                              'sheet_xml':selection['sheet_xml'], 'sequence_column':col['name'],
                              'source_rows':local['source_rows'], 'unique_eligible_sequences':len(source_values),
                              'exclusions':dict(local)})
    if not peptides:raise ValueError('source selection contains no valid peptide sequences; no catalog written')
    out.mkdir(parents=True,exist_ok=False)
    catalog=out/'catalog.yaml'
    with catalog.open('w') as f:
        f.write('schema: '+SCHEMA+'\ncandidates:\n')
        for seq in sorted(peptides):
            for z in (2,3):f.write(f'  - key: {seq}|z{z}\n    sequence: {seq}\n    charge: {z}\n')
    evidence={'status':'PASS_SOURCE_DEFINED_CATALOG','source_category':'ProteomeTools_Nature_Methods_2017_supplementary_table_1_peptide_pool_mapping',
              'source_paper_url':SOURCE_URL,'source_file_sha256':sha(source),
              'expected_source_sha256':expected_source_sha256,
              'source_authentication':'USER_SUPPLIED_PUBLISHER_WORKBOOK_NOT_INDEPENDENTLY_SHA_ATTESTED',
              'sheet_name':selected[0][0]['sheet'] if sheet_set is None else None,
              'sequence_header':selected[0][1]['name'] if sheet_set is None else 'Sequence',
              'header_row':selected[0][0]['header_row'] if sheet_set is None else 1,
              'source_columns_used_for_catalog':['Sequence'] if sheet_set is not None else [selected[0][1]['name']],
              'sheet_selection_policy':sheet_set or 'explicit_single_sheet',
              'source_sheets':per_sheet,
              'excluded_sheets':['Quality Control Peptides'] if sheet_set else [],
              'source_sheet_count':len(selected),
              'measured_spectra_used':False,
              'evaluation_labels_used':False,'evaluation_requests_used':False,
              'peptide_selection_based_on_validation_coverage':False,
              'source_rows':stats['source_rows'],'exclusions':dict(stats),
              'unique_peptide_sequences':len(peptides),'candidate_count':len(peptides)*2,
              'catalog_sha256':sha(catalog),'allowed_charges':[2,3],
              'rules':'bare unmodified 7..64 AA, no cysteine; preserve original library identity; no digestion',
              'limitations':['Publication pool-design table is a synthesized-peptide source; not an orthogonal spectrum identity confirmation',
                             'Only ProteomeTools 2017 source scope; may not cover 2019/2021 source spectra',
                             'No PTMs or cysteine; source may contain duplicate sequences',
                             'QC peptide worksheet excluded from the named three-synthesis-sheet policy',
                             'Unchanged 256-candidate mass retrieval may exclude present peptides']}
    dump(out/'catalog_provenance.json',evidence)
    print(f"REDEEM_PT_SOURCE_BUILD=PASS peptides={len(peptides)} candidates={2*len(peptides)}")
    print('REDEEM_PT_SOURCE_CATALOG='+str(catalog))
    return evidence


def new_cohort(bridge,old_cohort,out,limit=512,seed=20261023,records_override=None,pinned_revision=DATASET_SHA):
    """Select a new disjoint DEV cohort. Old labels used only to prevent repeated pairs."""
    out=Path(out);old=Path(old_cohort)
    if out.exists():raise ValueError('output directory exists')
    if sha(old/'truth.tsv')!=OLD_TRUTH_SHA or sha(old/'requests.tsv')!=OLD_REQUEST_SHA:
        raise ValueError('frozen old DEV512 cohort hash mismatch')
    old_meta=json.loads((old/'metadata.json').read_text())
    if old_meta.get('source_partition')!='validation' or old_meta.get('source_revision')!=DATASET_SHA:
        raise ValueError('frozen old DEV512 source differs')
    with (old/'truth.tsv').open(newline='') as f:
        prior={row['truth_candidate_key'] for row in csv.DictReader(f,delimiter='\t')}
    if len(prior)!=512:raise ValueError('old dev cohort not unique sequence/charge')
    spec=importlib.util.spec_from_file_location('redeem_external_bridge',str(bridge))
    if spec is None or spec.loader is None:raise ValueError('old DEV bridge missing')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    if b.DATASET!=DATASET:raise ValueError('unexpected source bridge')
    if records_override is None:
        stream,actual=b.remote_records(pinned_revision)
        if actual!=DATASET_SHA:raise ValueError('HF dataset revision changed')
    else:
        stream=records_override;actual=DATASET_SHA
    skipped=Counter()
    def eligible():
        for r in stream:
            exp=str(r.get('experiment_name',''))
            key=f"{r.get('sequence','')}|z{r.get('precursor_charge','')}"
            if exp not in ALLOWED_POOLS:
                skipped['excluded_other_synthesis_pool']+=1;continue
            if key in prior:
                skipped['excluded_prior_dev_sequence_charge']+=1;continue
            skipped['scope_eligible_raw_records']+=1
            yield r
    rows,counters=b.select_records(eligible(),limit,seed)
    for r in rows:
        if f"{r['sequence']}|z{r['charge']}" in prior:raise ValueError('cohort overlap detected')
    out.mkdir(parents=True,exist_ok=False)
    with (out/'requests.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(b.REQ_COLS)
        for r in rows:
            for mz,intensity in r['peaks']:
                w.writerow([r['query_id'],format(r['observed_precursor_mz'],'.12g'),r['charge'],256,format(mz,'.12g'),format(intensity,'.12g')])
    with (out/'truth.tsv').open('w',newline='') as f, (out/'evidence.tsv').open('w',newline='') as g:
        t=csv.writer(f,delimiter='\t',lineterminator='\n');t.writerow(['query_id','truth_candidate_key','partition','evidence_source'])
        e=csv.writer(g,delimiter='\t',lineterminator='\n');e.writerow(['query_id','experiment_name','evidence_index','scan_number','sequence','precursor_charge','peak_count'])
        for r in rows:
            t.writerow([r['query_id'],f"{r['sequence']}|z{r['charge']}",'DEV',f'{DATASET}/validation@{actual}'])
            e.writerow([r['query_id'],r['experiment_name'],r['evidence_index'],r['scan_number'],r['sequence'],r['charge'],len(r['peaks'])])
    data={'status':'PASS_DISJOINT_SOURCE_SCOPED_EXTERNAL_DEV','source_dataset':DATASET,
          'source_revision':actual,'source_partition':'validation','selected_queries':limit,'seed':seed,
          'allowed_pools':sorted(ALLOWED_POOLS),'counters':counters,'scope_counters':dict(skipped),
          'prior_dev_truth_sha256':OLD_TRUTH_SHA,'prior_dev_requests_sha256':OLD_REQUEST_SHA,
          'disjointness':'no identical unmodified_sequence+charge between old and new DEV512',
          'candidate_catalog_used_for_selection':False,'candidate_source_used_as_spectral_truth':False,
          'protected_partition_accessed':False,'redeem_train_overlap':'NOT_CHECKED',
          'requests_sha256':sha(out/'requests.tsv'),'truth_sha256':sha(out/'truth.tsv'),
          'evidence_sha256':sha(out/'evidence.tsv')}
    dump(out/'metadata.json',data)
    print(f'REDEEM_PT_NEW_DEV=PASS queries={limit} disjoint_old=YES')
    return data


def audit_coverage(catalog_dir,cohort,out):
    catalog_dir=Path(catalog_dir);cohort=Path(cohort);out=Path(out)
    if out.exists():raise ValueError('coverage output already exists')
    cat=json.loads((catalog_dir/'catalog_provenance.json').read_text())
    meta=json.loads((cohort/'metadata.json').read_text())
    if cat.get('status')!='PASS_SOURCE_DEFINED_CATALOG' or cat['evaluation_labels_used'] or cat['measured_spectra_used']:
        raise ValueError('catalog provenance invalid')
    if meta.get('status')!='PASS_DISJOINT_SOURCE_SCOPED_EXTERNAL_DEV' or meta.get('protected_partition_accessed') is not False:
        raise ValueError('new DEV cohort provenance invalid')
    if meta.get('source_revision')!=DATASET_SHA or meta.get('source_partition')!='validation':
        raise ValueError('DEV input partition invalid')
    if sha(catalog_dir/'catalog.yaml')!=cat['catalog_sha256'] or sha(cohort/'truth.tsv')!=meta['truth_sha256']:
        raise ValueError('catalog or truth hash mismatch')
    keys=set();n=0
    with (catalog_dir/'catalog.yaml').open() as f:
        if f.readline().strip()!=f'schema: {SCHEMA}' or f.readline().strip()!='candidates:':raise ValueError('catalog schema invalid')
        for line in f:
            if line.startswith('  - key: '):
                key=line[9:].strip()
                if key in keys:raise ValueError('duplicate catalog key')
                keys.add(key);n+=1
    if n!=cat['candidate_count']:raise ValueError('catalog count mismatch')
    counts=Counter();labels=[]
    with (cohort/'truth.tsv').open(newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['partition']!='DEV':raise ValueError('non DEV label')
            labels.append(row['truth_candidate_key'])
            counts['z'+row['truth_candidate_key'].rsplit('|z',1)[1]]+=1
    if len(labels)!=meta['selected_queries'] or len(set(labels))!=len(labels):raise ValueError('labels missing/duplicate')
    ncovered=sum(key in keys for key in labels)
    result={'status':'PASS_SOURCE_DEFINED_NEW_DEV_COVERAGE','queries':len(labels),
            'catalog_candidates':n,'reference_in_catalog':ncovered,'coverage_fraction':ncovered/len(labels),
            'coverage_threshold_predeclared':0.5,'next':'ALLOW_CPU_SCORING_IF_RUNTIME_RESOURCE_BUDGET_OK' if ncovered/len(labels)>=.5 else 'STOP_NO_SCORING',
            'catalog_sha256':cat['catalog_sha256'],'cohort_truth_sha256':meta['truth_sha256'],
            'cohort_requests_sha256':meta['requests_sha256'],
            'limitations':['Measured-spectrum PSM annotations are not orthogonally confirmed synthetic identities',
                           'Catalog coverage does not prove frozen nearest256 mass-pool recall',
                           'Cross-source identity leakage/overlap with ReDeeM historical TRAIN not assessed',
                           'Source library is 2017 pool-design data; new DEV subset uses declared 2017 pools']}
    dump(out,result)
    print(f'REDEEM_PT_NEW_COVERAGE=PASS {ncovered}/{len(labels)} gate={result["next"]}')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('inspect');q.add_argument('--source',required=True)
    q=sub.add_parser('build');q.add_argument('--source',required=True);q.add_argument('--output',required=True);q.add_argument('--sheet');q.add_argument('--column');q.add_argument('--sheet-set',choices=[PT2017_SHEET_SET]);q.add_argument('--expected-source-sha256');q.add_argument('--max-candidates',type=int,default=3000000)
    q=sub.add_parser('prepare-new-dev');q.add_argument('--old-cohort',required=True);q.add_argument('--bridge',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('coverage');q.add_argument('--catalog-dir',required=True);q.add_argument('--new-cohort',required=True);q.add_argument('--output',required=True)
    sub.add_parser('self-test')
    args=p.parse_args()
    if args.command=='inspect':print(json.dumps({'source_sha256':sha(args.source),'sheet_candidates':discover(args.source)},indent=2))
    elif args.command=='build':build(args.source,args.output,args.sheet,args.column,args.max_candidates,args.sheet_set,args.expected_source_sha256)
    elif args.command=='prepare-new-dev':new_cohort(args.bridge,args.old_cohort,args.output)
    elif args.command=='coverage':audit_coverage(args.catalog_dir,args.new_cohort,args.output)
    elif args.command=='self-test':self_test()


def self_test():
    from test_redeem_pt_source_library_20261023 import run_tests
    run_tests()

if __name__=='__main__':main()
