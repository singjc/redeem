#!/usr/bin/env python3
"""Immutable confirmation-contract checker. No network or TEST reads."""
import hashlib, json
from pathlib import Path
PROTOCOL_SHA='76cc1e656220e7573a082a9194fcf18af72a01e4ea72da25d9612904faf23c52'
MODEL_SHA='47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca'
CAT_SHA='a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f'
SIF_SHA='d46ed2bcc39ed50dec081762788ac60f4f3551658d44b555ce0aacf33504bf22'
REV='ffde6e4a2925f94dc28c87e33917db19a28f5869'
TEST_NAME='test-00000-of-00001-5f85882133d4ee76.parquet'
REQ_COUNT=512
POOL=256
EMBARGO={
 'dev1': ('614d00425c1bae2ce20f1363af8bfb03c251d7fd7638ce4eeb9fe1e915823cd5',512,'DEV'),
 'dev2': ('7fef846b19e56de0b93e4d22caf77fd04ff06b08c52a6db794f72fcc0ae49635',512,'DEV'),
 'dev3': ('34d8c9ac18abd56e8c0e96cc6f952816d9ac2cf7c7e4934643881e85eecde07f',512,'DEV'),
 'train':('def8392d155926c05050e69c0a580dabc29d3a43c76fbaadaabf948421896ad5',2048,'TRAIN')}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def protocol(path):
 if sha(path)!=PROTOCOL_SHA:raise ValueError('PREREGISTRATION_CHANGED: locked TEST protocol SHA mismatch')
 p=json.loads(Path(path).read_text())
 if (p.get('status')!='PREREGISTERED_BEFORE_TEST_ACCESS' or p.get('sample_size')!=512
  or p.get('library_sha256')!=CAT_SHA or p.get('model_sha256')!=MODEL_SHA
  or p.get('source_revision')!=REV or p.get('test_parquet')!='data/'+TEST_NAME
  or p.get('bootstrap_seed')!=20261025 or p.get('bootstrap_resamples')!=10000):
  raise ValueError('TEST contract fields changed')
 return p

def test_metadata(path, protocol_path, require_truth=False):
 protocol(protocol_path)
 m=json.loads(Path(path).read_text())
 expected={'status':'PASS_C2E_HELDOUT_TEST_PREPARATION','source_partition':'test',
  'source_revision':REV,'queries':512,'seed':20261025,
  'protocol_sha256':PROTOCOL_SHA,'frozen_model_sha256':MODEL_SHA,
  'disjoint_from_dev1_dev2_dev3_and_c2_train':True,
  'selection_uses_scores_or_catalog':False}
 if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('TEST cohort metadata does not match protocol')
 if require_truth and 'truth_sha256' not in m:raise ValueError('TEST truth hash missing')
 return m
