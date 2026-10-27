#!/usr/bin/env python3
"""C4 ZIP recovery: strict HTTP byte ranges, exact publisher MD5, no source revision.

Only the published nine-species-balanced.zip is eligible. Preserve existing .partial
bytes. A full publisher-MD5 match is required before creating the final .zip.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

SOURCE_URL = 'https://zenodo.org/records/13685813/files/nine-species-balanced.zip?download=1'
EXPECTED_MD5 = 'cc39b5c25c317f759a706b6d724d099a'
# This byte count is independently implied by the two interrupted curl logs.
EXPECTED_BYTES = 5_531_428_474
PART_NAME = 'nine-species-balanced.zip.partial'
ZIP_NAME = 'nine-species-balanced.zip'
CR_RE = re.compile(r'^bytes (\d+)-(\d+)/(\d+)$')

class SourceChanged(Exception):
    """Unsafe or unverified server response: preserve all existing data."""


def get_range(url: str, start: int, end: int, total: int | None = None,
              timeout: int = 90) -> tuple[bytes, int]:
    """Return exactly the requested byte range; never accept a full HTTP 200."""
    if start < 0 or end < start:
        raise ValueError('Invalid range')
    req = urllib.request.Request(url, headers={
        'Range': f'bytes={start}-{end}',
        'Accept-Encoding': 'identity',
        'User-Agent': 'ReDeeM-C4-source-recovery/20261026',
    })
    with urllib.request.urlopen(req, timeout=timeout) as res:
        if res.status != 206:
            raise SourceChanged(f'Server returned HTTP {res.status}, not 206. Range unsupported.')
        match = CR_RE.fullmatch(res.headers.get('Content-Range', '').strip())
        if not match:
            raise SourceChanged('Missing or invalid Content-Range')
        lo, hi, actual_total = map(int, match.groups())
        if (lo, hi) != (start, end) or (total is not None and actual_total != total):
            raise SourceChanged('Content-Range positions or source size differ')
        if res.headers.get('Content-Encoding', 'identity').lower() != 'identity':
            raise SourceChanged('Unexpected HTTP content encoding')
        wanted = end - start + 1
        if res.headers.get('Content-Length') is not None and int(res.headers['Content-Length']) != wanted:
            raise SourceChanged('Content-Length differs from requested range')
        data = res.read(wanted + 1)
        if len(data) != wanted:
            raise OSError(f'Truncated or oversized byte range: got {len(data)}, expected {wanted}')
        return data, actual_total


def sha_md5(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            md5.update(b)
            sha256.update(b)
    return md5.hexdigest(), sha256.hexdigest()


def verify_partial(url: str, part: Path, total: int):
    """Validate bytes from both ends of an existing partial before appending."""
    size = part.stat().st_size if part.exists() else 0
    if size > total:
        raise SourceChanged('Partial is larger than published object; preserve it')
    if not size:
        return size
    for lo, hi in ((0, min(size, 65536) - 1), (max(0, size - 65536), size - 1)):
        remote, _ = get_range(url, lo, hi, total)
        with part.open('rb') as f:
            f.seek(lo)
            local = f.read(hi - lo + 1)
        if remote != local:
            raise SourceChanged(f'Existing partial does not match source at range {lo}-{hi}; preserve it')
    return size


def fetch_with_retries(url: str, start: int, end: int, total: int, attempts: int = 12):
    for n in range(1, attempts + 1):
        try:
            return get_range(url, start, end, total)[0]
        except SourceChanged:
            raise
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            if n == attempts:
                raise RuntimeError(f'Byte range {start}-{end} failed after {attempts} attempts: {exc}') from exc
            print(f'REDEEM_C4_RANGE_RETRY={n}/{attempts} start={start} error={type(exc).__name__}', flush=True)
            time.sleep(min(5 * n, 30))
    raise AssertionError('Unreachable')


def recover(url: str, part: Path, archive: Path, mode: str, chunk_mib: int = 8,
            expected_md5: str = EXPECTED_MD5, expected_bytes: int = EXPECTED_BYTES):
    if archive.exists() or archive.is_symlink():
        raise FileExistsError('Final archive already exists; preserve it and use original inventory/probe')
    if not part.parent.is_dir() or part.is_symlink():
        raise ValueError('Partial parent missing or partial is a symbolic link')
    _, total = get_range(url, 0, 0)
    if total != expected_bytes:
        raise SourceChanged(f'Remote archive size {total} does not match pinned {expected_bytes}')
    size = verify_partial(url, part, total)
    print(f'REDEEM_C4_RANGE_PREFLIGHT=PASS remote_bytes={total} existing_partial_bytes={size}', flush=True)
    if mode == 'probe':
        print('REDEEM_C4_RANGE_PROBE=PASS no_download_started', flush=True)
        return
    if mode != 'resume':
        raise ValueError('Use probe or resume')
    chunk = chunk_mib * 1024 * 1024
    checkpoint = size // (128 * 1024 * 1024)
    with part.open('ab') as out:
        while size < total:
            end = min(total - 1, size + chunk - 1)
            data = fetch_with_retries(url, size, end, total)
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
            size += len(data)
            new_checkpoint = size // (128 * 1024 * 1024)
            if new_checkpoint != checkpoint or size == total:
                print(f'REDEEM_C4_RANGE_PROGRESS={size}/{total} ({size/total:.1%})', flush=True)
                checkpoint = new_checkpoint
    md5, sha = sha_md5(part)
    if md5 != expected_md5:
        raise SourceChanged(f'Full ZIP MD5 differs: {md5}; PARTIAL PRESERVED; DO NOT USE AS SOURCE')
    # Hard link is exclusive: a concurrent final archive cannot be overwritten.
    os.link(part, archive)
    part.unlink()
    receipt = {'status':'PASS_COMPLETE_PUBLISHED_ZENODO_SOURCE','url':url,
               'zenodo_record':'13685813','archive':str(archive),'size_bytes':total,
               'published_md5':md5,'sha256':sha,'recovery':'strict_http_206_ranges'}
    rec = archive.with_name('nine-species-balanced.recovery-provenance.json')
    if not rec.exists():
        with rec.open('x') as f:
            json.dump(receipt, f, indent=2, sort_keys=True)
            f.write('\n')
    print(f'REDEEM_C4_RANGE_DOWNLOAD=PASS publisher_md5={md5} sha256={sha}',flush=True)
    print(f'REDEEM_C4_ARCHIVE={archive}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['probe', 'resume'])
    parser.add_argument('--downloads', type=Path, default=Path.home()/'Downloads')
    parser.add_argument('--chunk-mib', type=int, choices=[2,4,8,16], default=8)
    args = parser.parse_args()
    try:
        recover(SOURCE_URL, args.downloads/PART_NAME, args.downloads/ZIP_NAME,
                args.mode, args.chunk_mib)
    except Exception as exc:
        print(f'REDEEM_C4_RANGE_RECOVERY=FAIL {type(exc).__name__}: {exc}', file=sys.stderr)
        print('EXISTING_PARTIAL_PRESERVED=YES; no catalog/model/TEST modified',file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
