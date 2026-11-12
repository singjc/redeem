#!/usr/bin/env python3
"""F2-G evaluation: last-k checkpoint averaging vs epoch_060 (rule: handoff 2026-11-12 21:52).

usage: redeem_f2g_eval_20261112.py F2G_SCORES RTLONG_VAL_SCORES   (control rows rtlong{s}_ep60)
       redeem_f2g_eval_20261112.py --self-test
"""
import os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from redeem_f2e_eval_20261107 import MDE, gain, source_means  # noqa: E402

TASKS = ('rt', 'ccs', 'ms2')


def label(task, gains):
    d = statistics.fmean(gains)
    if d <= -MDE[task]:
        return 'AVG_HURTS'
    if d >= MDE[task] and all(g > 0 for g in gains):
        return 'AVG_HELPS'
    if all(g > 0 for g in gains):
        return 'AVG_SMALL_GAIN'
    return 'AVG_NO_EFFECT' if abs(d) < MDE[task] else 'INCONSISTENT'


def decide(labels, spread):
    """labels/spread: {task: str}/{task: bool} for one arm. True means 'propose EMA'."""
    good = sum(labels[t] == 'AVG_HELPS' or spread[t] for t in TASKS)
    return good >= 2 and not any(labels[t] == 'AVG_HURTS' for t in TASKS)


def evaluate(f2g, ctrl, out=print):
    a = source_means(f2g, {f'a{k}_s{s}' for k in (5, 10) for s in range(3)})
    c = source_means(ctrl, {f'rtlong{s}_ep60' for s in range(3)})
    for k in (5, 10):
        labels, spread = {}, {}
        for t in TASKS:
            cv = [c[(f'rtlong{s}_ep60', t)] for s in range(3)]
            av = [a[(f'a{k}_s{s}', t)] for s in range(3)]
            g = [gain(t, x, y) for x, y in zip(cv, av)]
            labels[t] = label(t, g)
            sd_c, sd_a = statistics.stdev(cv), statistics.stdev(av)
            spread[t] = sd_a <= 0.5 * sd_c
            out(f'A{k}\t{t}\tctrl={statistics.fmean(cv):.4f}\tavg={statistics.fmean(av):.4f}\t'
                f'gain={statistics.fmean(g):+.4f}\tpairs={",".join(f"{x:+.4f}" for x in g)}\t'
                f'MDE={MDE[t]}\tlabel={labels[t]}\tsd_ctrl={sd_c:.4f}\tsd_avg={sd_a:.4f}\t'
                f'SPREAD_REDUCED={spread[t]}')
        out(f'decision\tA{k}\tPROPOSE_EMA={decide(labels, spread)}')
    out('F2G_EVAL=COMPLETE')


def self_test():
    assert label('rt', [0.6, 0.7, 0.8]) == 'AVG_HELPS'
    assert label('rt', [0.1, 0.2, 0.1]) == 'AVG_SMALL_GAIN'
    assert label('rt', [0.1, -0.2, 0.1]) == 'AVG_NO_EFFECT'
    assert label('rt', [-0.6, -0.7, -0.6]) == 'AVG_HURTS'
    T, F = True, False
    assert decide({'rt': 'AVG_HELPS', 'ccs': 'AVG_NO_EFFECT', 'ms2': 'AVG_NO_EFFECT'},
                  {'rt': F, 'ccs': T, 'ms2': F})
    assert not decide({'rt': 'AVG_HELPS', 'ccs': 'AVG_HURTS', 'ms2': 'AVG_NO_EFFECT'},
                      {'rt': F, 'ccs': T, 'ms2': T})
    assert not decide({'rt': 'AVG_SMALL_GAIN', 'ccs': 'AVG_SMALL_GAIN', 'ms2': 'AVG_SMALL_GAIN'},
                      {'rt': F, 'ccs': F, 'ms2': T})
    print('F2G_EVAL_SELF_TEST=PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test()
    else:
        evaluate(*sys.argv[1:3])
