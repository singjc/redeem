#!/usr/bin/env python3
"""Regression tests: source-only identities and strict DEV label isolation."""
import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import redeem_pt_source_library_20261023 as m


def mini_xlsx(path,header=('Peptide Sequence','Peptide Pool'),bad=False):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('xl/workbook.xml','''<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Synthesis Pools" sheetId="1" r:id="rId1"/></sheets></workbook>''')
        z.writestr('xl/_rels/workbook.xml.rels','''<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>''')
        def cell(col,idx,s):
            return f'<c r="{col}{idx}" t="inlineStr"><is><t>{s}</t></is></c>'
        seq=['PEPTIDER','PEPTIDK','CPEPTIDER','PEPTIDER','PEPTIDER','ACDEFGHIK','MQETIDEK','PEPTIDEK',
             'PEPTIDR','PEPTIDEQ','PEPTIDEM','PEPTIDEN','PEPTIDEF','PEPTIDEH','PEPTIDEW','PEPTIDEY']
        rows=[f'<row r="1">{cell("A",1,header[0])}{cell("B",1,header[1])}</row>']
        rows += [f'<row r="{i}">{cell("A",i,seq[i-2])}{cell("B",i,"2017 design")}</row>' for i in range(2,18)]
        z.writestr('xl/worksheets/sheet1.xml','<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(rows)+'</sheetData></worksheet>')



def multi_sheet_xlsx(path, remove=None, bad_header=None):
    # Synthetic replica of the observed 2017 workbook structure, including
    # literal quotation marks around the three synthesis worksheet names.
    from xml.sax.saxutils import escape
    names=['"Proteotypic Set"','"Missing Gene Set"','"SRMAtlas Set"','Quality Control Peptides']
    expected={'"Proteotypic Set"':'Proteotypic','"Missing Gene Set"':'Missing',
              '"SRMAtlas Set"':'SRMAtlas','Quality Control Peptides':'QC'}
    if remove is not None:names.remove(remove)
    seqs={
       '"Proteotypic Set"':['PEPTIDER','PEPTIDK','PEPTIDEQ','PEPTIDEN','PEPTIDEF','PEPTIDEH',
                             'PEPTIDEW','PEPTIDEY','PEPTIDEM','PEPTIDET','PEPTIDEV','PEPTIDEA'],
       '"Missing Gene Set"':['PEPTIDER','QWERTYKK','MWERTYKK','AWERTYKK','DWERTYKK','EWERTYKK',
                              'FWERTYKK','GWERTYKK','HWERTYKK','IWERTYKK','KWERTYKK','LWERTYKK'],
       '"SRMAtlas Set"':['PEPTIDK','PEPTIDERS','RPEPTIDER','NPEPTIDER','VPEPTIDER','HPEPTIDER',
                         'WPEPTIDER','YPEPTIDER','DPEPTIDER','MPEPTIDER','EPEPTIDER','FPEPTIDER'],
       'Quality Control Peptides':['QCPEPTIDE']*12,
    }
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        headings=''.join(f'<sheet name="{escape(s,{chr(34):"&quot;"})}" sheetId="{i}" r:id="rId{i}"/>' for i,s in enumerate(names,1))
        z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+headings+'</sheets></workbook>')
        rels=''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(names)+1))
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+rels+'</Relationships>')
        def cell(col,idx,val):
            return f'<c r="{col}{idx}" t="inlineStr"><is><t>{escape(val)}</t></is></c>'
        for i,s in enumerate(names,1):
            if s=='Quality Control Peptides':
                header=['Sequence','Type','QC type']
            else:
                header=['Pool name','Set Name','Sequence','QC type']
                if s==bad_header:header[2]='Peptide identifier'
            rows=[f'<row r="1">'+''.join(cell(col,1,value) for col,value in zip('ABCD',header))+'</row>']
            for j,seq in enumerate(seqs[s],2):
                row=[seq,'Retention Time Peptide','JPT-RT'] if s=='Quality Control Peptides' else [f'pool_{j}',expected[s],seq,'']
                rows.append(f'<row r="{j}">'+''.join(cell(col,j,v) for col,v in zip('ABCD',row))+'</row>')
            z.writestr(f'xl/worksheets/sheet{i}.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(rows)+'</sheetData></worksheet>')

class SourceLibraryTests(unittest.TestCase):
    def test_import_build_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);src=d/'pool.xlsx';mini_xlsx(src)
            found=m.discover(src)
            self.assertEqual(len(found),1)
            result=m.build(src,d/'catalog')
            self.assertFalse(result['measured_spectra_used'])
            self.assertFalse(result['evaluation_labels_used'])
            self.assertEqual(result['unique_peptide_sequences'],12)
            self.assertNotIn('CPEPTIDER|z2',(d/'catalog/catalog.yaml').read_text())
            self.assertIn('PEPTIDER|z3',(d/'catalog/catalog.yaml').read_text())
            self.assertEqual(result['catalog_sha256'],m.sha(d/'catalog/catalog.yaml'))
            with self.assertRaisesRegex(ValueError,'exists'):
                m.build(src,d/'catalog')
            self.assertFalse((d/'catalog/truth.tsv').exists())

    def test_ambiguous_and_missing_headers_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);src=d/'pool.xlsx';mini_xlsx(src,('Sequence','Peptide Sequence'))
            with self.assertRaisesRegex(ValueError,'exactly one'):m.build(src,d/'cat')
            self.assertFalse((d/'cat').exists())
            self.assertIn('peptide',m.select_header(src,column='Peptide Sequence')[1]['name'].lower())
            mini_xlsx(d/'no.xlsx',('Andromeda Score','Precursor mz'))
            with self.assertRaisesRegex(ValueError,'exactly one'):m.build(d/'no.xlsx',d/'no')
            self.assertFalse((d/'no').exists())

    def test_candidate_budget_no_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);src=d/'pool.xlsx';mini_xlsx(src)
            with self.assertRaisesRegex(ValueError,'budget'):m.build(src,d/'cat',max_candidates=3)
            self.assertFalse((d/'cat').exists())

    def test_new_cohort_requires_old_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);old=d/'old';old.mkdir()
            (old/'truth.tsv').write_text('query_id\ttruth_candidate_key\tpartition\tevidence_source\n')
            (old/'requests.tsv').write_text('query_id\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                m.new_cohort('missing_bridge.py',old,d/'new',records_override=[])
            self.assertFalse((d/'new').exists())

    def test_disjoint_new_dev_end_to_end_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);old=d/'old';old.mkdir()
            # 512 source reference keys, none in the intended new 2017 samples.
            letters='ADEFGHIKLMNPQRSTVWY'
            ids=[]
            for i in range(512):
                a=i;prefix=[]
                for j in range(7):
                    prefix.append(letters[a%len(letters)]);a//=len(letters)
                ids.append(''.join(prefix)+'|z2')
            with (old/'truth.tsv').open('w',newline='') as f:
                writer=csv.writer(f,delimiter='\t',lineterminator='\n')
                writer.writerow(['query_id','truth_candidate_key','partition','evidence_source'])
                for i,key in enumerate(ids):writer.writerow([f'old{i}',key,'DEV','old'])
            (old/'requests.tsv').write_text('a frozen old request fixture\n')
            m.dump(old/'metadata.json',{'source_partition':'validation','source_revision':m.DATASET_SHA})
            original=(m.OLD_TRUTH_SHA,m.OLD_REQUEST_SHA)
            m.OLD_TRUTH_SHA=m.sha(old/'truth.tsv');m.OLD_REQUEST_SHA=m.sha(old/'requests.tsv')
            try:
                bridge=Path(__file__).parent/'redeem_external_dev_benchmark_20261022.py'
                def row(seq,i,exp='TUM_first_pool'):
                    return {'sequence':seq,'modified_sequence':seq,'precursor_charge':2,'precursor_mz':700.,
                       'mz_array':[float(j+100) for j in range(20)],'intensity_array':[float(j+1) for j in range(20)],
                       'experiment_name':exp,'evidence_index':i,'scan_number':100+i}
                observations=[row(ids[0].split('|')[0],1),row('PEPTIDER',2),row('PEPTIDK',3),
                              row('PEPTIDEQ',4),row('PEPTIDEN',5),row('PEPTIDEF',6,exp='OTHER')]
                prepared=m.new_cohort(bridge,old,d/'new',limit=3,records_override=iter(observations))
                self.assertEqual(prepared['selected_queries'],3)
                self.assertGreaterEqual(prepared['scope_counters']['excluded_prior_dev_sequence_charge'],1)
                newkeys={r['truth_candidate_key'] for r in csv.DictReader((d/'new/truth.tsv').open(),delimiter='\t')}
                self.assertFalse(newkeys.intersection(ids))
                self.assertTrue(newkeys <= {'PEPTIDER|z2','PEPTIDK|z2','PEPTIDEQ|z2','PEPTIDEN|z2'})
            finally:
                m.OLD_TRUTH_SHA,m.OLD_REQUEST_SHA=original

    def test_coverage_is_label_separated_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);src=d/'pool.xlsx';mini_xlsx(src)
            m.build(src,d/'catalog')
            cohort=d/'dev';cohort.mkdir()
            (cohort/'truth.tsv').write_text('query_id\ttruth_candidate_key\tpartition\tevidence_source\nq1\tPEPTIDER|z2\tDEV\tsrc\nq2\tPEPTIDEQ|z3\tDEV\tsrc\nq3\tPEPTIDK|z2\tDEV\tsrc\nq4\tNOTINLIB|z2\tDEV\tsrc\n')
            (cohort/'requests.tsv').write_text('request header\n')
            m.dump(cohort/'metadata.json',{'status':'PASS_DISJOINT_SOURCE_SCOPED_EXTERNAL_DEV','source_revision':m.DATASET_SHA,
                'source_partition':'validation','selected_queries':4,'protected_partition_accessed':False,
                'truth_sha256':m.sha(cohort/'truth.tsv'),'requests_sha256':m.sha(cohort/'requests.tsv')})
            result=m.audit_coverage(d/'catalog',cohort,d/'cov.json')
            self.assertEqual(result['reference_in_catalog'],3)
            self.assertEqual(result['next'],'ALLOW_CPU_SCORING_IF_RUNTIME_RESOURCE_BUDGET_OK')
            (cohort/'truth.tsv').write_text((cohort/'truth.tsv').read_text()+'q5\tBAD|z2\tDEV\tsrc\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):m.audit_coverage(d/'catalog',cohort,d/'cov2.json')


    def test_three_source_sheets_union_and_qc_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);src=d/'pools.xlsx';multi_sheet_xlsx(src)
            with self.assertRaisesRegex(ValueError,'exactly one'):
                m.build(src,d/'incorrect')
            result=m.build(src,d/'catalog',sheet_set=m.PT2017_SHEET_SET)
            self.assertEqual(result['source_sheet_count'],3)
            self.assertEqual([v['canonical_name'] for v in result['source_sheets']],list(m.PT2017_SYNTHESIS_SHEETS))
            self.assertEqual(result['excluded_sheets'],['Quality Control Peptides'])
            keys=(d/'catalog/catalog.yaml').read_text()
            for value in ['PEPTIDER|z2','QWERTYKK|z3','RPEPTIDER|z2']:
                self.assertIn(value,keys)
            self.assertNotIn('QCPEPTIDE',keys)
            self.assertFalse(result['evaluation_labels_used'])
            self.assertEqual(result['candidate_count'],2*result['unique_peptide_sequences'])
            self.assertEqual(result['catalog_sha256'],m.sha(d/'catalog/catalog.yaml'))
            # Predeclared rule is deterministic, including when workbook sheets are quoted.
            again=m.build(src,d/'again',sheet_set=m.PT2017_SHEET_SET)
            self.assertEqual(result['catalog_sha256'],again['catalog_sha256'])

    def test_three_source_sheets_fail_closed_missing_or_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            src=d/'missing.xlsx';multi_sheet_xlsx(src,remove='"Missing Gene Set"')
            with self.assertRaisesRegex(ValueError,'Missing Gene Set'):
                m.build(src,d/'missing',sheet_set=m.PT2017_SHEET_SET)
            self.assertFalse((d/'missing').exists())
            bad=d/'bad.xlsx';multi_sheet_xlsx(bad,bad_header='"SRMAtlas Set"')
            with self.assertRaisesRegex(ValueError,'SRMAtlas Set'):
                m.build(bad,d/'invalid',sheet_set=m.PT2017_SHEET_SET)
            self.assertFalse((d/'invalid').exists())

    def test_pinned_publisher_workbook_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);src=d/'source.xlsx';multi_sheet_xlsx(src)
            with self.assertRaisesRegex(ValueError,'SHA256 differs'):
                m.build(src,d/'wrong',sheet_set=m.PT2017_SHEET_SET,expected_source_sha256='0'*64)
            self.assertFalse((d/'wrong').exists())
            result=m.build(src,d/'right',sheet_set=m.PT2017_SHEET_SET,expected_source_sha256=m.sha(src))
            self.assertEqual(result['expected_source_sha256'],m.sha(src))

    def test_three_source_sheets_fail_closed_budget_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);src=d/'source.xlsx';multi_sheet_xlsx(src)
            with self.assertRaisesRegex(ValueError,'conflicting'):
                m.build(src,d/'conflict',sheet='"Proteotypic Set"',sheet_set=m.PT2017_SHEET_SET)
            self.assertFalse((d/'conflict').exists())
            with self.assertRaisesRegex(ValueError,'budget'):
                m.build(src,d/'budget',sheet_set=m.PT2017_SHEET_SET,max_candidates=4)
            self.assertFalse((d/'budget').exists())

def run_tests():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SourceLibraryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit('REDEEM_PT_SOURCE_TEST=FAIL')
    print('REDEEM_PT_SOURCE_SELFTEST=PASS single_sheet_compat,three_sheet_union,QC_exclusion,strict_header,size_budget,disjoint_DEV,coverage')

if __name__=='__main__':run_tests()
