#!/usr/bin/env python3
"""F2-D evaluation: preregistered UNDERFIT / GENERALIZATION / MIXED rule per task.

Rule text: rolling handoff 2026-11-07 00:08 (+ 00:11 addendum). Metric = mean over sources.
usage: f2d_eval.py TRAIN_SCORES SEEN_TSV VAL_V052_SCORES(15264522) VAL_R_SCORES(22808699)
       f2d_eval.py --self-test
"""
import statistics, sys
from collections import defaultdict

MDE = {'rt': 0.58, 'ccs': 2.28, 'ms2': 0.124}  # F2-B one-seed MDE; >= 2x every F2-A CI half-width
UNIFIED = ['r0_ep10', 'r1_ep10', 'r2_ep10']
SEEDS = {'r0_ep10': '20261006', 'r1_ep10': '20261007', 'r2_ep10': '20261008'}


def load(path, labels):
    d = defaultdict(lambda: defaultdict(dict))  # (label, task) -> source -> {rec: value}
    for line in open(path):
        if line.startswith('#') or line.startswith('label\t'):
            continue
        label, task, source, rec, target, pred, _ = line.rstrip('\n').split('\t')
        if label in labels and pred != 'NA':
            v = abs(float(pred) - float(target)) if task in ('rt', 'ccs') else float(pred)
            d[(label, task)][source][int(rec)] = v
    return d


def smean(by_source, keep=None):
    means, n = [], 0
    for recs in by_source.values():
        vals = [v for r, v in recs.items() if keep is None or r in keep]
        if vals:
            means.append(statistics.fmean(vals))
            n += len(vals)
    return (statistics.fmean(means) if means else float('nan')), n


def load_seen(path):
    seen = defaultdict(set)  # seed -> {rec}
    cols = None
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        if f[0] == 'task':
            cols = f[3:]
        elif cols and len(f) == 3 + len(cols) and f[0] in ('rt', 'ms2'):
            for c, flag in zip(cols, f[3:]):
                if flag == '1':
                    seen[c.removeprefix('seen_')].add(int(f[2]))
    return seen


def decide(task, u_tr, u_val, v_tr, v_val):
    lower = task != 'ms2'
    g_u = (u_val - u_tr) if lower else (u_tr - u_val)
    g_v = (v_val - v_tr) if lower else (v_tr - v_val)
    if lower:
        underfit = u_tr >= 0.75 * u_val and u_tr > v_val
        fits = u_tr <= v_val
    else:
        underfit = u_tr <= v_val and g_u <= 0.25 * (v_val - u_val)
        fits = u_tr >= v_val
    general = fits and g_u > g_v + MDE[task]
    verdict = 'UNDERFIT' if underfit else 'GENERALIZATION' if general else 'MIXED'
    return verdict, g_u, g_v


def evaluate(train_path, seen_path, v052_path, r_path, out=print):
    tr = load(train_path, set(UNIFIED) | {'v052'})
    vv = load(v052_path, {'v052'})
    vr = load(r_path, set(UNIFIED))
    seen = load_seen(seen_path)
    verdicts = {}
    out('task\tsubset\tlabel\ttrain\tval\tn_train')
    for task in ('rt', 'ccs', 'ms2'):
        v_tr, _ = smean(tr[('v052', task)])
        v_val, _ = smean(vv[('v052', task)])
        out(f'{task}\tall\tv052\t{v_tr:.4f}\t{v_val:.4f}\t-')
        rows = {}
        for lab in UNIFIED:
            a, n = smean(tr[(lab, task)])
            val, _ = smean(vr[(lab, task)])
            rows[lab] = (a, val)
            out(f'{task}\tall\t{lab}\t{a:.4f}\t{val:.4f}\t{n}')
            if task != 'ccs':
                s, ns = smean(tr[(lab, task)], seen[SEEDS[lab]])
                all_recs = {r for src in tr[(lab, task)].values() for r in src}
                u, nu = smean(tr[(lab, task)], all_recs - seen[SEEDS[lab]])
                out(f'{task}\tseen\t{lab}\t{s:.4f}\t{val:.4f}\t{ns}')
                out(f'{task}\tunseen\t{lab}\t{u:.4f}\t{val:.4f}\t{nu}')
                rows[lab] = rows[lab] + (s, ns)
        u_val = statistics.fmean(r[1] for r in rows.values())
        u_all = statistics.fmean(r[0] for r in rows.values())
        subsets = [('all', u_all)]
        if task != 'ccs':
            seen_n = sum(r[3] for r in rows.values()) / 3
            have = [r[2] for r in rows.values() if r[3] > 0]
            if have:
                subsets.append(('seen', statistics.fmean(have)))
        for name, u_tr in subsets:
            verdict, g_u, g_v = decide(task, u_tr, u_val, v_tr, v_val)
            note = ''
            if name == 'seen' and seen_n < 200:
                note = f'\tDESCRIPTIVE_ONLY(mean_seen_n={seen_n:.0f}<200)'
            out(f'rule\t{task}\t{name}\tunified_train={u_tr:.4f}\tunified_val={u_val:.4f}\t'
                f'v052_train={v_tr:.4f}\tv052_val={v_val:.4f}\tG_unified={g_u:+.4f}\t'
                f'G_v052={g_v:+.4f}\tverdict={verdict}{note}')
            verdicts[(task, name)] = verdict
    out('F2D_EVAL=COMPLETE')
    return verdicts


def self_test():
    # Synthetic: unified as bad on TRAIN as on Val and worse than v052 -> UNDERFIT.
    assert decide('rt', 4.4, 4.5, 2.5, 3.3)[0] == 'UNDERFIT'
    # Unified fits TRAIN better than v052 Val but generalizes poorly -> GENERALIZATION.
    assert decide('rt', 2.0, 4.5, 2.5, 3.3)[0] == 'GENERALIZATION'
    assert decide('ms2', 0.95, 0.78, 0.93, 0.91)[0] == 'GENERALIZATION'
    assert decide('ms2', 0.79, 0.78, 0.93, 0.91)[0] == 'UNDERFIT'
    # Fits TRAIN like v052 but gap not larger than v052's by MDE -> MIXED.
    assert decide('rt', 3.2, 3.5, 2.5, 3.3)[0] == 'MIXED'
    # Neither fits (3.34 > 3.3) nor underfit (3.34 < 0.75 * 4.5) -> MIXED.
    assert decide('rt', 3.34, 4.5, 2.5, 3.3)[0] == 'MIXED'
    print('F2D_EVAL_SELF_TEST=PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test()
    else:
        evaluate(*sys.argv[1:5])
