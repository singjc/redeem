#!/usr/bin/env python3
"""F2-C summary: per-stage, per-term gradient norm share and cosine with the stage total.

Input: the probe job output folder (P0..P3/train.log with foundation_term_gradients lines).
Stage index: 0/1 property A/B, 2 representation, 3 mobility, 4 inverse.
Share = term_norm / sum of term norms in that line. Reported as the median over probed cycles,
plus the cycle-0 value (exact checkpoint weights for the warm-start phases).
"""
import statistics, sys
from collections import defaultdict

STAGES = {0: 'property_a', 1: 'property_b', 2: 'representation', 3: 'mobility', 4: 'inverse'}


def parse(path):
    rows = []
    for line in open(path):
        if not line.startswith('foundation_term_gradients'):
            continue
        kv = dict(x.split('=', 1) for x in line.rstrip('\n').split('\t')[1:])
        names = [k[:-5] for k in kv if k.endswith('_norm') and k != 'total_norm']
        rows.append((int(kv['stage']), int(kv['cycle']), float(kv['total_norm']),
                     float(kv['sum_residual']),
                     {n: (float(kv[n + '_value']), float(kv[n + '_norm']),
                          None if kv[n + '_cos_total'] == 'NA' else float(kv[n + '_cos_total']))
                      for n in names}))
    return rows


def main():
    out = sys.argv[1]
    print('phase\tstage\tterm\tn\tcycle0_share\tmedian_share\tmedian_norm\tmedian_cos_total\tmedian_value')
    for phase in ('P0', 'P1', 'P2', 'P3'):
        rows = parse(f'{out}/{phase}/train.log')
        by = defaultdict(lambda: defaultdict(list))
        first = {}
        for stage, cycle, _tot, res, terms in rows:
            assert abs(res) <= 1e-3 * max(1.0, sum(abs(v[0]) for v in terms.values())), (phase, stage, cycle, res)
            s = sum(v[1] for v in terms.values()) or 1.0
            for n, (val, norm, cos) in terms.items():
                by[stage][n].append((norm / s, norm, cos, val))
                if cycle == 0:
                    first[(stage, n)] = norm / s
        for stage in sorted(by):
            for n, xs in by[stage].items():
                med = lambda i: statistics.median(x[i] for x in xs if x[i] is not None)
                print(f'{phase}\t{STAGES[stage]}\t{n}\t{len(xs)}\t{first.get((stage, n), float("nan")):.3f}\t'
                      f'{med(0):.3f}\t{med(1):.4f}\t{med(2):.3f}\t{med(3):.4f}')
    print('F2C_SUMMARY=COMPLETE')


if __name__ == '__main__':
    main()
