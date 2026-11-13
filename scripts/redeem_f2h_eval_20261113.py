#!/usr/bin/env python3
"""F2-H evaluation: Harmonized RT target vs rtlong ep60 (rule: rolling handoff 2026-11-13 07:25,
approved 07:28). Seed-paired, MDE RT 0.58 / CCS 2.28 / MS2 0.124 (F2-B).

usage: redeem_f2h_eval_20261113.py F2H_VAL_SCORES RTLONG_VAL_SCORES [F2H_TRAIN RTLONG_TRAIN]
       redeem_f2h_eval_20261113.py --self-test
The optional TRAIN pair gives the descriptive mechanism check (60-SPD TRAIN-fit slope).
"""
import os, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from redeem_f2e_eval_20261107 import MDE, gain, source_means  # noqa: E402

TASKS = ('rt', 'ccs', 'ms2')
SPD60 = ('pxd058337_60spd_fractionation', 'pxd058337_60spd_fractionationgpf', 'pxd058337_60spd_gpf')


def rt_label(gains):
    d = statistics.fmean(gains)
    if d <= -MDE['rt']:
        return 'HARM_HURTS_RT'
    if d >= MDE['rt'] and all(g > 0 for g in gains):
        return 'HARM_HELPS_RT'
    if all(g > 0 for g in gains):
        return 'HARM_SMALL_RT'
    return 'HARM_NO_EFFECT' if abs(d) < MDE['rt'] else 'INCONSISTENT'


def guard(ccs_gain, ms2_gain):
    return ccs_gain > -MDE['ccs'] and ms2_gain > -MDE['ms2']


def rt_rows(path, labels):
    d = defaultdict(list)
    for line in open(path):
        if line.startswith('#') or line.startswith('label\t'):
            continue
        label, task, src, _r, tgt, pred, _ = line.rstrip('\n').split('\t')
        if task == 'rt' and label in labels and pred != 'NA':
            d[(label, src)].append((float(pred), float(tgt)))
    return d


def slope(pairs):
    xs, ys = zip(*pairs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in pairs) / sum((x - mx) ** 2 for x in xs)


def evaluate(h_val, c_val, h_train=None, c_train=None, out=print):
    H = [f'rtharm{s}_ep60' for s in range(3)]
    C = [f'rtlong{s}_ep60' for s in range(3)]
    h, c = source_means(h_val, set(H)), source_means(c_val, set(C))
    gains = {}
    for t in TASKS:
        g = [gain(t, c[(C[s], t)], h[(H[s], t)]) for s in range(3)]
        gains[t] = g
        out(f'task\t{t}\tctrl_rtlong_ep60={statistics.fmean(c[(x, t)] for x in C):.4f}\t'
            f'rtharm_ep60={statistics.fmean(h[(x, t)] for x in H):.4f}\tgain={statistics.fmean(g):+.4f}\t'
            f'pairs={",".join(f"{x:+.4f}" for x in g)}\tMDE={MDE[t]}')
    ok = guard(statistics.fmean(gains['ccs']), statistics.fmean(gains['ms2']))
    out(f'rule\tprimary={rt_label(gains["rt"])}\tGUARD={"PASS" if ok else "HARM_COSTS_OTHER"}')
    hv, cv = rt_rows(h_val, set(H)), rt_rows(c_val, set(C))
    for src in SPD60:
        mh = statistics.fmean(statistics.fmean(abs(p - y) for p, y in hv[(x, src)]) for x in H)
        mc = statistics.fmean(statistics.fmean(abs(p - y) for p, y in cv[(x, src)]) for x in C)
        out(f'mechanism_val\t{src}\trtlong_MAE={mc:.3f}\trtharm_MAE={mh:.3f}')
    if h_train and c_train:
        ht, ct = rt_rows(h_train, set(H)), rt_rows(c_train, set(C))
        for src in SPD60:
            sh = statistics.fmean(slope(ht[(x, src)]) for x in H)
            sc = statistics.fmean(slope(ct[(x, src)]) for x in C)
            out(f'mechanism_train_slope\t{src}\trtlong={sc:.3f}\trtharm={sh:.3f}\t(1.0 = no scale error)')
    out('F2H_EVAL=COMPLETE')


def self_test():
    assert rt_label([0.6, 0.7, 0.8]) == 'HARM_HELPS_RT'
    assert rt_label([0.1, 0.2, 0.15]) == 'HARM_SMALL_RT'
    assert rt_label([0.1, -0.2, 0.1]) == 'HARM_NO_EFFECT'
    assert rt_label([-0.7, -0.6, -0.6]) == 'HARM_HURTS_RT'
    assert guard(-1.0, -0.05) and not guard(-2.5, 0.0) and not guard(0.0, -0.2)
    assert abs(slope([(1, 2), (2, 4), (3, 6)]) - 2.0) < 1e-12
    print('F2H_EVAL_SELF_TEST=PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test()
    else:
        evaluate(*sys.argv[1:5])
