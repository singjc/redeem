#!/usr/bin/env python3
"""F2-I evaluation: ccsx (Harmonized RT + mobility batch 32 + CCS length/mass prior) vs F2-H rtharm
ep60 (rule: rolling handoff 2026-11-13 09:21, approved 09:19). Seed-paired, MDE RT 0.58 / CCS 2.28 /
MS2 0.124 (F2-B).

usage: redeem_f2i_eval_20261113.py CCSX_VAL RTHARM_VAL CCSX_TRAIN RTHARM_TRAIN BENCHMARK_TSV
       redeem_f2i_eval_20261113.py --self-test
Primary CCS: gain >= MDE and all 3 pairs > 0 -> CCS_HELPS; all pairs > 0 -> CCS_SMALL; else
CCS_NO_EFFECT. Guard: RT gain > -0.58 and MS2 gain > -0.124, else HARM. The TRAIN CCS signed bias
by peptide length is descriptive only (the mechanism check).
"""
import os, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from redeem_f2e_eval_20261107 import MDE, gain, source_means  # noqa: E402

TASKS = ('rt', 'ccs', 'ms2')
BINS = ((7, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 99))


def ccs_label(gains):
    d = statistics.fmean(gains)
    if d >= MDE['ccs'] and all(g > 0 for g in gains):
        return 'CCS_HELPS'
    if all(g > 0 for g in gains):
        return 'CCS_SMALL'
    return 'CCS_NO_EFFECT'


def guard(rt_gain, ms2_gain):
    return rt_gain > -MDE['rt'] and ms2_gain > -MDE['ms2']


def ccs_rows(path, labels):
    """record -> (target, mean prediction over the given labels)."""
    acc, tgt = defaultdict(list), {}
    for line in open(path):
        if line.startswith('#') or line.startswith('label\t'):
            continue
        label, task, _src, rec, y, p, _ = line.rstrip('\n').split('\t')
        if task == 'ccs' and label in labels and p != 'NA':
            acc[rec].append(float(p))
            tgt[rec] = float(y)
    return {r: (tgt[r], statistics.fmean(v)) for r, v in acc.items()}


def length_bias(rows, length):
    out = []
    for lo, hi in BINS:
        rs = [r for r in rows if lo <= length.get(r, 0) <= hi]
        if rs:
            out.append((lo, hi, len(rs), statistics.fmean(rows[r][1] - rows[r][0] for r in rs),
                        statistics.fmean(abs(rows[r][1] - rows[r][0]) for r in rs)))
    return out


def evaluate(x_val, c_val, x_train, c_train, bench, out=print):
    arm = os.environ.get('F2I_ARM', 'ccsx')  # ccsx (F2-I) or ccslm (F2-I2, C2 only)
    if arm not in ('ccsx', 'ccslm'):
        raise SystemExit('F2I_ARM must be ccsx or ccslm')
    X = [f'{arm}{s}_ep60' for s in range(3)]
    C = [f'rtharm{s}_ep60' for s in range(3)]
    x, c = source_means(x_val, set(X)), source_means(c_val, set(C))
    gains = {}
    for t in TASKS:
        g = [gain(t, c[(C[s], t)], x[(X[s], t)]) for s in range(3)]
        gains[t] = g
        out(f'task\t{t}\tctrl_rtharm_ep60={statistics.fmean(c[(k, t)] for k in C):.4f}\t'
            f'{arm}_ep60={statistics.fmean(x[(k, t)] for k in X):.4f}\tgain={statistics.fmean(g):+.4f}\t'
            f'pairs={",".join(f"{v:+.4f}" for v in g)}\tMDE={MDE[t]}')
    ok = guard(statistics.fmean(gains['rt']), statistics.fmean(gains['ms2']))
    out(f'rule\tprimary={ccs_label(gains["ccs"])}\tGUARD={"PASS" if ok else "HARM"}')
    xt, ct = ccs_rows(x_train, set(X)), ccs_rows(c_train, set(C))
    need = set(xt) | set(ct)
    length = {}
    for line in open(bench):
        f = line.split('\t')
        if f[0] in need:
            length[f[0]] = len(f[4])
    cb = {b[:2]: b for b in length_bias(ct, length)}
    for lo, hi, n, bias, mae in length_bias(xt, length):
        c_n, c_bias, c_mae = cb[(lo, hi)][2:] if (lo, hi) in cb else (0, float('nan'), float('nan'))
        out(f'mechanism_train_ccs_length\t{lo}-{hi}\tn={n}\trtharm_bias={c_bias:+.2f}\trtharm_MAE={c_mae:.2f}\t'
            f'{arm}_bias={bias:+.2f}\t{arm}_MAE={mae:.2f}')
    out('F2I_EVAL=COMPLETE')


def self_test():
    assert ccs_label([2.5, 2.4, 2.3]) == 'CCS_HELPS'
    assert ccs_label([3.0, 3.5, -0.1]) == 'CCS_NO_EFFECT'
    assert ccs_label([0.5, 0.2, 0.1]) == 'CCS_SMALL'
    assert ccs_label([0.5, -0.2, 0.1]) == 'CCS_NO_EFFECT'
    assert guard(-0.5, -0.1) and not guard(-0.6, 0.0) and not guard(0.0, -0.13)
    rows = {'a': (100.0, 102.0), 'b': (200.0, 196.0), 'c': (300.0, 300.0)}
    lb = length_bias(rows, {'a': 8, 'b': 9, 'c': 35})
    assert lb[0][:3] == (7, 10, 2) and abs(lb[0][3] + 1.0) < 1e-12 and abs(lb[0][4] - 3.0) < 1e-12
    assert lb[1][:3] == (31, 99, 1) and lb[1][3] == 0.0
    print('F2I_EVAL_SELF_TEST=PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test()
    else:
        evaluate(*sys.argv[1:6])
