#!/usr/bin/env python3
"""F2-E evaluation: preregistered budget rule (rolling handoff 2026-11-07 10:09).

Seed-paired: arm e{i}_ep30 vs baseline r{i}_ep10 (i = 0, 1, 2). Metric = mean over sources.
BUDGET_HELPS if mean improvement >= MDE and all 3 pairs improve; NO_EFFECT if |mean| < MDE.
SATURATING if the ep20->30 gain < 1/3 of the ep10->20 gain (arm means).

usage: f2e_eval.py ARM_VAL_SCORES BASELINE_VAL_SCORES(22808699) V052_VAL_SCORES(15264522)
       f2e_eval.py --self-test
"""
import statistics, sys
from collections import defaultdict

MDE = {'rt': 0.58, 'ccs': 2.28, 'ms2': 0.124}
LOWER = {'rt': True, 'ccs': True, 'ms2': False}


def source_means(path, labels):
    acc = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        if line.startswith('#') or line.startswith('label\t'):
            continue
        label, task, source, _rec, target, pred, _ = line.rstrip('\n').split('\t')
        if label in labels and pred != 'NA':
            v = abs(float(pred) - float(target)) if task in ('rt', 'ccs') else float(pred)
            acc[(label, task)][source].append(v)
    return {k: statistics.fmean(statistics.fmean(v) for v in s.values()) for k, s in acc.items()}


def gain(task, before, after):
    """Improvement in the 'better' direction (positive = better)."""
    return before - after if LOWER[task] else after - before


def decide(task, pairs, ep10, ep20, ep30):
    """pairs: [(baseline_ep10, arm_ep30)] per seed; ep10/20/30: arm means."""
    gains = [gain(task, b, a) for b, a in pairs]
    delta = statistics.fmean(gains)
    if delta >= MDE[task] and all(g > 0 for g in gains):
        verdict = 'BUDGET_HELPS'
    elif abs(delta) < MDE[task]:
        verdict = 'NO_EFFECT'
    else:
        verdict = 'INCONSISTENT'
    g1, g2 = gain(task, ep10, ep20), gain(task, ep20, ep30)
    saturating = g1 > 0 and g2 < g1 / 3
    return verdict, delta, gains, saturating, g1, g2


def evaluate(arm_path, base_path, v052_path, out=print):
    arm = source_means(arm_path, {f'e{i}_ep{e}' for i in range(3) for e in (10, 20, 30)})
    base = source_means(base_path, {f'r{i}_ep10' for i in range(3)})
    v = source_means(v052_path, {'v052'})
    result = {}
    for task in ('rt', 'ccs', 'ms2'):
        pairs = [(base[(f'r{i}_ep10', task)], arm[(f'e{i}_ep30', task)]) for i in range(3)]
        m = {e: statistics.fmean(arm[(f'e{i}_ep{e}', task)] for i in range(3)) for e in (10, 20, 30)}
        verdict, delta, gains, sat, g1, g2 = decide(task, pairs, m[10], m[20], m[30])
        gap = gain(task, m[30], v[('v052', task)])  # > 0: v0.52 still better
        out(f'rule\t{task}\tverdict={verdict}\tdelta={delta:+.4f}\tMDE={MDE[task]}\t'
            f'pair_gains={",".join(f"{g:+.4f}" for g in gains)}\t'
            f'arm_ep10={m[10]:.4f}\tarm_ep20={m[20]:.4f}\tarm_ep30={m[30]:.4f}\t'
            f'gain10_20={g1:+.4f}\tgain20_30={g2:+.4f}\tsaturating={sat}\t'
            f'v052={v[("v052", task)]:.4f}\tgap_to_v052={gap:+.4f}')
        result[task] = (verdict, sat)
    out('F2E_EVAL=COMPLETE')
    return result


def self_test():
    # RT improves by 1.0 in every pair, gains 10->20 = 0.6, 20->30 = 0.1 -> HELPS, SATURATING.
    v, d, g, s, *_ = decide('rt', [(4.5, 3.5), (4.6, 3.6), (4.7, 3.7)], 4.3, 3.7, 3.6)
    assert v == 'BUDGET_HELPS' and abs(d - 1.0) < 1e-9 and s
    # Small change -> NO_EFFECT.
    assert decide('rt', [(4.5, 4.4), (4.6, 4.5), (4.7, 4.8)], 4.6, 4.55, 4.5)[0] == 'NO_EFFECT'
    # Large mean but one pair worse -> INCONSISTENT.
    assert decide('rt', [(4.5, 2.5), (4.6, 2.6), (4.7, 4.9)], 4.6, 3.5, 3.3)[0] == 'INCONSISTENT'
    # MS2 is higher-better.
    assert decide('ms2', [(0.80, 0.95), (0.78, 0.93), (0.70, 0.85)], 0.80, 0.86, 0.91)[0] == 'BUDGET_HELPS'
    print('F2E_EVAL_SELF_TEST=PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test()
    else:
        evaluate(*sys.argv[1:4])
