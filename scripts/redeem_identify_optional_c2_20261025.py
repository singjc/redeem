#!/usr/bin/env python3
"""Opt-in, forward-only ReDeeM C2 inference adapter; no model training or labels.

Default geometry delegates to the frozen v0.77 binary and preserves the exact
native hit bytes. C2 requires the pinned 2017 catalog, model and SIF, a full
256-hit pool per spectrum, and the exact original C1 feature implementation.
The adapter is deliberately experimental, not the native Rust production CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import importlib

# Chemistry modules are loaded only in opt-in mode, AFTER checksum checks.
chemistry = None
ranker = None

EXPECTED_MODEL = '47818819cd17b95a5fce13ba624c38a99035805b0e11c9a943f846c266dd61ca'
EXPECTED_CATALOG = 'a7113cc33de29cc6561c31a6eb76d606e9b05d482eeada93d39c58ecba3d3c4f'
EXPECTED_SIF = 'd46ed2bcc39ed50dec081762788ac60f4f3551658d44b555ce0aacf33504bf22'
EXPECTED_CHEMISTRY_SOURCE = '288568b3132d362a225ce23203565232074adb1197f0c0c7830344f845508acb'
EXPECTED_RANKER_SOURCE = '9128c3db1d2996059afc81d8422b9ec33f7b629d5627b56b352fe67700b8d498'
VERSION = 'v0.77.0-practical-identifier-stream'
STATUS = 'C2_EXPERIMENTAL_NOT_PRODUCTION'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_model(model_path: Path, expected_model: str) -> dict:
    require(model_path.is_file() and sha(model_path) == expected_model,
            'C2 model missing or SHA-256 mismatch; never refit automatically')
    doc = json.loads(model_path.read_text())
    require(doc.get('schema') == 'redeem.c2.chemistry_pairwise_ranker.v1' and
            doc.get('status') == 'TRAIN_ONLY_FROZEN_COEFFICIENTS_NOT_DEV_EVALUATED',
            'Unexpected C2 model schema/status')
    require(doc.get('feature_names') == ranker.FEATURES and
            doc.get('transformations') == ranker.TRANSFORMS and
            doc.get('config') == ranker.CONFIG and
            doc.get('base_score') == 'unchanged_geometry_score' and
            doc.get('missing_fragment_error_ppm_sentinel') == 1000,
            'C2 feature order, transformations or fitting contract mismatch')
    for name in ('standardization_mean', 'standardization_std_min_005', 'coefficients'):
        values = doc.get(name)
        require(isinstance(values, list) and len(values) == len(ranker.FEATURES) and
                all(isinstance(x, (int, float)) and math.isfinite(x) for x in values),
                f'Invalid model {name}')
    require(all(x >= 0.05 for x in doc['standardization_std_min_005']),
            'Invalid C2 standardisation floor')
    require(all(abs(x) <= 1.0 for x in doc['coefficients']),
            'Invalid fitted C2 correction magnitude')
    require(doc.get('validation_labels_accessed') is False and
            doc.get('test_accessed') is False,
            'Model does not assert TRAIN-only label use')
    return doc


def verify_bundle_scripts(base: Path, chemistry_sha=EXPECTED_CHEMISTRY_SOURCE,
                          ranker_sha=EXPECTED_RANKER_SOURCE) -> None:
    for name, expected in (
        ('redeem_c2_train_features_20261023.py', chemistry_sha),
        ('redeem_c2_ranker_20261023.py', ranker_sha),
    ):
        require(sha(base / name) == expected, f'Pinned source checksum failed: {name}')


def load_bundle(base: Path, chemistry_sha=EXPECTED_CHEMISTRY_SOURCE,
                ranker_sha=EXPECTED_RANKER_SOURCE) -> None:
    global chemistry, ranker
    verify_bundle_scripts(base, chemistry_sha, ranker_sha)
    chemistry = importlib.import_module('redeem_c2_train_features_20261023')
    ranker = importlib.import_module('redeem_c2_ranker_20261023')


def identifier_command(args, expected_sif=EXPECTED_SIF) -> list[str]:
    if args.sif:
        require(not args.identifier_exe, 'Use exactly one identifier runtime')
        require(args.sif.is_file() and sha(args.sif) == expected_sif,
                'Identifier SIF does not match the frozen, parity-qualified v0.77 SIF')
        require(args.bind_root and args.bind_root.is_dir(),
                'SIF execution needs an existing --bind-root inside the container')
        require(not args.bind_root.is_symlink(),
                'Bind root must be the declared /nfs path, not an aliased symlink')
        return [args.container_runtime, 'exec', '--bind',
                f'{args.bind_root}:{args.bind_root}', str(args.sif),
                'foundation_identify_stream_v0770']
    require(args.identifier_exe is not None, 'Specify --sif or --identifier-exe')
    require(args.ranker == 'geometry',
            'Experimental C2 requires the exact parity-qualified SIF (not an arbitrary binary)')
    return [str(args.identifier_exe)]


def command_check(cmd: list[str]) -> None:
    version = subprocess.run(cmd + ['--version'], capture_output=True, text=True, check=True)
    require(version.stdout.strip() == VERSION, 'Identifier version does not match frozen v0.77')
    smoke = subprocess.run(cmd + ['--self-test'], capture_output=True, text=True, check=True)
    require('v0770_practical_identifier_stream_self_test=PASS' in smoke.stdout,
            'Identifier did not pass frozen self-test')


def load_native(req: Path, hit_path: Path, n_queries: int) -> tuple[dict, list[dict]]:
    observations = chemistry.load_requests(req, n_queries)
    hits = list(chemistry.read_rows(hit_path, chemistry.HIT_HEADER))
    last: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for row in hits:
        q = row['query_id']
        require(q in observations, 'Native result contains unknown query')
        r = int(row['rank'])
        require(r == last.get(q, 0) + 1 and r <= 256 and
                int(row['charge']) == observations[q]['charge'],
                'Incomplete / inconsistent native candidate rank or charge')
        last[q] = r
        identity = (q, row['key'])
        require(identity not in seen, 'Duplicate native candidate identity')
        seen.add(identity)
    require(set(last) == set(observations) and all(v == 256 for v in last.values()) and
            len(hits) == n_queries * 256,
            'C2 requires exactly 256 complete candidates per spectrum')
    return observations, hits


def generate_features(observations: dict, hits: list[dict], output: Path) -> None:
    with output.open('x', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(chemistry.FEATURE_HEADER)
        for hit in hits:
            q = hit['query_id']
            values = chemistry.features(observations[q], hit)
            record = {'query_id': q, 'original_rank': int(hit['rank']),
                      'candidate_index': int(hit['candidate_index']),
                      'key': hit['key'], 'charge': int(hit['charge']), **values}
            writer.writerow([chemistry.fmt(record[k]) for k in chemistry.FEATURE_HEADER])


def apply_frozen_c2(features_path: Path, output: Path, doc: dict,
                    expected_queries: int) -> None:
    """Pure-Python deployment of original six-feature C2 formula (no NumPy on Codon).

    The model, coefficient order, feature transformations and clipping match the
    frozen TRAIN-only apply implementation. Test against its NumPy version on
    new synthetic candidates, never on the protected TEST.
    """
    groups: dict[str, list[tuple[int, str, float, float]]] = {}
    previous: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    with features_path.open(newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        require(reader.fieldnames == chemistry.FEATURE_HEADER, 'Unexpected chemistry feature order')
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()),
                    'Truncated chemistry TSV')
            q = row['query_id']; rank = int(row['original_rank'])
            require(rank == previous.get(q, 0) + 1 and rank <= 256,
                    'C2 feature input must have ordered complete pools')
            previous[q] = rank
            key = row['key']
            require((q, key) not in seen, 'Repeated C2 candidate')
            seen.add((q, key))
            geom = ranker.numeric(row['geometry_score'], 'geometry_score')
            vals = []
            for name, transform in zip(ranker.FEATURES, ranker.TRANSFORMS):
                raw = row[name]
                v = 1000.0 if name == 'median_backbone_mass_error_ppm' and raw == '' else ranker.numeric(raw, name)
                require(v >= 0, f'Negative physical feature {name}')
                if transform == 'log1p':
                    v = math.log1p(min(v, 1e6))
                elif transform == 'log1p100':
                    v = math.log1p(min(v * 100.0, 1e6))
                vals.append(v)
            normalized = [max(-6.0, min(6.0, (v - m) / sd)) for v, m, sd in
                          zip(vals, doc['standardization_mean'], doc['standardization_std_min_005'])]
            score = geom + sum(v * weight for v, weight in zip(normalized, doc['coefficients']))
            require(math.isfinite(score), 'Invalid C2 candidate score')
            groups.setdefault(q, []).append((rank, key, geom, score))
    require(len(groups) == expected_queries and all(r == 256 for r in previous.values()) and
            len(seen) == expected_queries * 256,
            'C2 inference requires 256 complete candidates per query')
    with output.open('x', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['query_id', 'new_rank', 'candidate_key', 'original_rank',
                    'original_geometry', 'frozen_c2_score'])
        for q in sorted(groups):
            ordered = sorted(groups[q], key=lambda entry: (-entry[3], entry[0]))
            for new_rank, (old_rank, key, geom, score) in enumerate(ordered, 1):
                w.writerow([q, new_rank, key, old_rank, format(geom, '.10g'),
                            format(score, '.10g')])


def validate_reranking(native: list[dict], ranked_path: Path, output: Path,
                       queries: int) -> None:
    native_map = {(r['query_id'], int(r['rank'])): r for r in native}
    ids_per_query: dict[str, set[int]] = {}
    seen: set[tuple[str, int]] = set()
    with output.open('x', newline='') as o:
        writer = csv.writer(o, delimiter='\t', lineterminator='\n')
        writer.writerow(chemistry.HIT_HEADER)
        with ranked_path.open(newline='') as f:
            r = csv.DictReader(f, delimiter='\t')
            require(r.fieldnames == ['query_id', 'new_rank', 'candidate_key', 'original_rank',
                                     'original_geometry', 'frozen_c2_score'],
                    'Unexpected C2 score table columns')
            last_rank: dict[str, int] = {}
            for row in r:
                q = row['query_id']; nr = int(row['new_rank']); old = int(row['original_rank'])
                require(nr == last_rank.get(q, 0) + 1 and nr <= 256,
                        'Incomplete C2 ranks')
                last_rank[q] = nr
                key = (q, old)
                require(key in native_map and key not in seen, 'C2 changed the candidate universe')
                seen.add(key)
                original = native_map[key]
                require(original['key'] == row['candidate_key'], 'C2 candidate key changed')
                require(math.isfinite(float(row['frozen_c2_score'])), 'Invalid C2 score')
                ids_per_query.setdefault(q, set()).add(old)
                writer.writerow([q, str(nr), original['candidate_index'], original['key'],
                                 original['charge'], original['theoretical_neutral_mass'],
                                 original['absolute_neutral_mass_error'], original['geometry_score']])
    require(len(last_rank) == queries and len(seen) == len(native_map) and
            all(len(ids) == 256 and ids == set(range(1, 257)) for ids in ids_per_query.values()),
            'C2 output lost or added candidates')


def add_c2_summary(native: Path, dest: Path, model_sha: str, catalog_sha: str,
                   hits_sha: str, features_sha: str, ranked_sha: str) -> None:
    text = native.read_text()
    require(text.startswith('key\tvalue\n') and 'status\tPASS\n' in text,
            'Invalid v0.77 native summary')
    dest.write_text(text + ''.join(f'{k}\t{v}\n' for k, v in [
        ('ranker', 'c2-experimental'), ('status_scope', STATUS),
        ('model_sha256', model_sha), ('catalog_sha256', catalog_sha),
        ('native_hits_sha256', hits_sha), ('chemistry_features_sha256', features_sha),
        ('c2_scores_sha256', ranked_sha), ('truth_labels_read', 'NO'),
        ('production_approved', 'NO'),
    ]))


def install_outputs(staged: dict[Path, Path]) -> None:
    installed = []
    try:
        for target, source in staged.items():
            require(not target.exists() and not target.is_symlink(),
                    f'Refuse to overwrite an existing output: {target}')
            # Hard-link exclusive creation, same directory / filesystem staging.
            os.link(source, target, follow_symlinks=False)
            installed.append(target)
    except BaseException:
        for target in installed:
            target.unlink(missing_ok=True)
        raise


def run(args, *, expected_model=EXPECTED_MODEL, expected_catalog=EXPECTED_CATALOG,
        expected_sif=EXPECTED_SIF, chemistry_sha=EXPECTED_CHEMISTRY_SOURCE,
        ranker_sha=EXPECTED_RANKER_SOURCE) -> dict:
    outputs = [args.output_hits, args.output_summary]
    if args.ranker == 'c2-experimental':
        require(args.model and args.output_native_hits and args.output_scores,
                'C2 requires --model, --output-native-hits and --output-scores')
        outputs += [args.output_native_hits, args.output_scores]
    else:
        require(not args.model and not args.output_native_hits and not args.output_scores,
                'Geometry must not take C2-only arguments')
    require(len(set(outputs)) == len(outputs), 'Distinct output paths required')
    inputs = [args.catalog, args.spectra] + ([args.model] if args.model else [])
    require(all(p.is_file() for p in inputs), 'Required input file missing')
    require(all(p.parent.exists() and p.parent.is_dir() for p in outputs),
            'Output parent directory must exist')
    require(all(os.stat(p.parent).st_dev == os.stat(args.output_hits.parent).st_dev
                for p in outputs),
            'All output directories must be on the same filesystem for atomic install')
    require(not set(outputs) & set(inputs), 'Output must not overwrite an input')
    require(all(not p.exists() and not p.is_symlink() for p in outputs),
            'Output already exists; preserve previous result')

    if args.ranker == 'c2-experimental':
        require(sha(args.catalog) == expected_catalog, 'Frozen C2 catalog SHA mismatch')
        load_bundle(Path(__file__).parent, chemistry_sha, ranker_sha)
        doc = parse_model(args.model, expected_model)
        require(args.queries is not None and 1 <= args.queries <= 8192,
                'C2 requires --queries between 1 and 8192')
    command = identifier_command(args, expected_sif)
    command_check(command)

    # Keep all intermediate files local; no TRAIN/DEV/TEST labels are ever accepted.
    with tempfile.TemporaryDirectory(prefix='.redeem-c2-infer-', dir=args.output_hits.parent) as dirname:
        tmp = Path(dirname)
        raw_hits = tmp / 'native_hits.tsv'; raw_summary = tmp / 'native_summary.tsv'
        call = command + ['identify-stream', '--catalog', str(args.catalog),
                          '--spectra', str(args.spectra), '--output-hits', str(raw_hits),
                          '--output-summary', str(raw_summary)]
        subprocess.run(call, check=True)
        require(raw_hits.is_file() and raw_summary.is_file(), 'Frozen identifier did not produce outputs')
        if args.ranker == 'geometry':
            install_outputs({args.output_hits: raw_hits, args.output_summary: raw_summary})
            return {'status': 'PASS_NATIVE_GEOMETRY_UNMODIFIED',
                    'native_hits_sha256': sha(raw_hits), 'ranker': 'geometry'}

        observations, hits = load_native(args.spectra, raw_hits, args.queries)
        feature_tsv = tmp / 'features.tsv'
        generate_features(observations, hits, feature_tsv)
        require(sha(Path(__file__).parent / 'redeem_c2_train_features_20261023.py') == chemistry_sha,
                'Chemistry implementation changed during scoring')
        rank_tsv = tmp / 'c2_ranked.tsv'
        apply_frozen_c2(feature_tsv, rank_tsv, doc, args.queries)
        c2_hits = tmp / 'c2_hits.tsv'
        validate_reranking(hits, rank_tsv, c2_hits, args.queries)
        result_summary = tmp / 'c2_summary.tsv'
        add_c2_summary(raw_summary, result_summary, expected_model, expected_catalog,
                       sha(raw_hits), sha(feature_tsv), sha(rank_tsv))
        # Install all requested files only once every validation succeeded.
        install_outputs({args.output_hits: c2_hits,
                         args.output_summary: result_summary,
                         args.output_native_hits: raw_hits,
                         args.output_scores: rank_tsv})
        return {'status': STATUS, 'ranker': 'c2-experimental', 'queries': args.queries,
                'model_sha256': expected_model, 'catalog_sha256': expected_catalog,
                'native_hits_sha256': sha(raw_hits), 'reranked_hits_sha256': sha(c2_hits),
                'feature_sha256': sha(feature_tsv)}


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranker', choices=['geometry', 'c2-experimental'], default='geometry')
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--spectra', type=Path, required=True)
    parser.add_argument('--output-hits', type=Path, required=True)
    parser.add_argument('--output-summary', type=Path, required=True)
    parser.add_argument('--identifier-exe', type=Path)
    parser.add_argument('--sif', type=Path)
    parser.add_argument('--container-runtime', default='singularity', choices=['singularity', 'apptainer'])
    parser.add_argument('--bind-root', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--queries', type=int, default=0)
    parser.add_argument('--output-native-hits', type=Path)
    parser.add_argument('--output-scores', type=Path)
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f'REDEEM_OPTIONAL_INFERENCE=BLOCKED {type(e).__name__}: {e}', file=sys.stderr)
        raise SystemExit(2)
    print('REDEEM_OPTIONAL_INFERENCE=' + json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    cli()
