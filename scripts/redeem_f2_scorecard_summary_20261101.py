#!/usr/bin/env python3
"""F2-A scorecard summary: per-source metrics, paired bootstrap CIs, and noise floor.

Input: the TSV written by `foundation_f2_scorecard score` (one row per label/task/record).
Optional: --epochs LABEL=validation_metrics.tsv ... to add the epoch-to-epoch spread of
existing training runs (native metrics on DEV1024; a different cohort, used only as a
noise estimate).

Metrics per task: RT harmonized MAE, CCS MAE (A^2), MS2 mean cosine. Summary = mean over
sources (each source equal weight) and pooled. Paired differences vs REFERENCE use the same
records. Bootstrap resamples records within each source (stratified), 2000 replicates.
Descriptive measurement only; nothing is selected or tuned on this cohort.
"""
import argparse, csv, math, random, statistics, sys
from collections import defaultdict

LOWER_BETTER = {'rt': True, 'ccs': True, 'ms2': False}


def per_record_value(task, target, pred):
    return abs(pred - target) if task in ('rt', 'ccs') else pred


def load(path):
    data = defaultdict(dict)  # (label, task) -> {record: (source, value)}
    for line in open(path):
        if line.startswith('#') or line.startswith('label\t'):
            continue
        label, task, source, rec, target, pred, _ = line.rstrip('\n').split('\t')
        if pred == 'NA':
            continue
        data[(label, task)][int(rec)] = (source, per_record_value(task, float(target), float(pred)))
    return data


def summarise(values_by_source):
    """values_by_source: {source: [v...]} -> (macro mean over sources, pooled mean)."""
    means = [statistics.fmean(v) for v in values_by_source.values() if v]
    pooled = [x for v in values_by_source.values() for x in v]
    return statistics.fmean(means), statistics.fmean(pooled)


def bootstrap(paired_by_source, reps, rng):
    """paired_by_source: {source: [(a, b)...]} -> 95% CI of macro mean(a) - mean(b)."""
    out = []
    items = list(paired_by_source.items())
    for _ in range(reps):
        diffs = []
        for _, pairs in items:
            sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            diffs.append(statistics.fmean(a - b for a, b in sample))
        out.append(statistics.fmean(diffs))
    out.sort()
    return out[int(0.025 * reps)], out[int(0.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scores')
    ap.add_argument('--reference', default='v052')
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--epochs', nargs='*', default=[])
    a = ap.parse_args()
    rng = random.Random(a.seed)
    data = load(a.scores)
    labels = sorted({l for l, _ in data})
    tasks = [t for t in ('rt', 'ccs', 'ms2') if any((l, t) in data for l in labels)]
    print('#metric\trt=harmonized MAE (lower better); ccs=MAE A^2 (lower better); ms2=mean cosine (higher better)')
    print('section\ttask\tlabel\tsources\tn\tmacro\tpooled')
    for t in tasks:
        for l in labels:
            d = data.get((l, t), {})
            by = defaultdict(list)
            for s, v in d.values():
                by[s].append(v)
            if by:
                m, p = summarise(by)
                print(f'summary\t{t}\t{l}\t{len(by)}\t{len(d)}\t{m:.5f}\t{p:.5f}')
    print('section\ttask\tlabel\tsource\tn\tvalue')
    for t in tasks:
        for l in labels:
            by = defaultdict(list)
            for s, v in data.get((l, t), {}).values():
                by[s].append(v)
            for s in sorted(by):
                print(f'per_source\t{t}\t{l}\t{s}\t{len(by[s])}\t{statistics.fmean(by[s]):.5f}')
    print('section\ttask\tlabel_minus_reference\tn_common\tmacro_diff\tci95_low\tci95_high\tci_half_width')
    for t in tasks:
        ref = data.get((a.reference, t), {})
        for l in labels:
            if l == a.reference:
                continue
            cur = data.get((l, t), {})
            common = sorted(set(ref) & set(cur))
            by = defaultdict(list)
            for r in common:
                by[cur[r][0]].append((cur[r][1], ref[r][1]))
            if not by:
                continue
            diff = statistics.fmean(statistics.fmean(x - y for x, y in v) for v in by.values())
            lo, hi = bootstrap(by, a.reps, rng)
            print(f'paired\t{t}\t{l}\t{len(common)}\t{diff:+.5f}\t{lo:+.5f}\t{hi:+.5f}\t{(hi - lo) / 2:.5f}')
    if a.epochs:
        print('section\tmetric\trun\tepochs\tmean\tsd_epoch_to_epoch\tsd_last4')
        cols = {'rt': 'rt_mae_native', 'ccs': 'ccs_mae', 'ms2': 'ms2_cosine_similarity'}
        for spec in a.epochs:
            run, path = spec.split('=', 1)
            rows = list(csv.DictReader(open(path), delimiter='\t'))
            for t, c in cols.items():
                v = [float(r[c]) for r in rows if r.get(c) not in (None, '', 'NA')]
                if len(v) < 3:
                    continue
                steps = [v[i + 1] - v[i] for i in range(len(v) - 1)]
                sd_step = statistics.pstdev(steps) / math.sqrt(2)  # per-epoch noise if steps are noise
                last = v[-4:]
                print(f'epoch_noise\t{t}\t{run}\t{len(v)}\t{statistics.fmean(v):.5f}\t{sd_step:.5f}\t{statistics.pstdev(last):.5f}')
    print('F2_SUMMARY=COMPLETE')


if __name__ == '__main__':
    sys.exit(main())
