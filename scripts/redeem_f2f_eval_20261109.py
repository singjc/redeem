#!/usr/bin/env python3
"""F2-F evaluation: preregistered rules for arm (a) MS2-TOK and arm (b) RT-LONG.

Rule text: rolling handoff 2026-11-09 10:50. Control = F2-E 30-epoch arm e{i}_ep30
(scores in redeem_f2e_budget_20261107/score/val_23502270). Seed-paired (i = 0, 1, 2).
Reuses source_means/gain/MDE from the F2-E evaluator (same folder).

usage: f2f_eval.py ms2tok ARM_VAL_SCORES F2E_VAL_SCORES V052_VAL_SCORES
       f2f_eval.py rtlong ARM_VAL_SCORES F2E_VAL_SCORES V052_VAL_SCORES
       f2f_eval.py --self-test
"""
import os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from redeem_f2e_eval_20261107 import MDE, gain, source_means  # noqa: E402

TASKS = ('rt', 'ccs', 'ms2')


def paired(task, ctrl, arm):
    gains = [gain(task, c, a) for c, a in zip(ctrl, arm)]
    return statistics.fmean(gains), gains


def label(task, delta, gains):
    if delta >= MDE[task] and all(g > 0 for g in gains):
        return 'HELPS'
    if abs(delta) < MDE[task]:
        return 'NO_EFFECT'
    return 'INCONSISTENT'


def ms2tok_verdict(ms2_delta, ms2_gains, arm_ms2_mean, v052_ms2, rt_delta, ccs_delta):
    primary = {'HELPS': 'TOKENS_HELP', 'NO_EFFECT': 'TOKENS_NO_EFFECT'}.get(
        label('ms2', ms2_delta, ms2_gains), 'INCONSISTENT')
    close_gap = all(g > 0 for g in ms2_gains) and arm_ms2_mean >= v052_ms2 - 0.010
    # "Worse" = negative gain; harm if MAE rises by more than the MDE.
    harm = rt_delta <= -MDE['rt'] or ccs_delta <= -MDE['ccs']
    return primary, close_gap, harm


def rtlong_verdict(rt_delta, rt_gains, ep30, ep45, ep60):
    primary = {'HELPS': 'LONG_HELPS_RT', 'NO_EFFECT': 'LONG_NO_EFFECT'}.get(
        label('rt', rt_delta, rt_gains), 'INCONSISTENT')
    g1, g2 = gain('rt', ep30, ep45), gain('rt', ep45, ep60)
    return primary, (g1 > 0 and g2 < g1 / 3), g1, g2


def evaluate(arm, arm_path, ctrl_path, v052_path, out=print):
    last = 30 if arm == 'ms2tok' else 60
    epochs = (30,) if arm == 'ms2tok' else (30, 45, 60)
    a = source_means(arm_path, {f'{arm}{i}_ep{e}' for i in range(3) for e in epochs})
    c = source_means(ctrl_path, {f'e{i}_ep30' for i in range(3)})
    v = source_means(v052_path, {'v052'})
    res = {}
    for t in TASKS:
        ctrl = [c[(f'e{i}_ep30', t)] for i in range(3)]
        armv = [a[(f'{arm}{i}_ep{last}', t)] for i in range(3)]
        d, g = paired(t, ctrl, armv)
        res[t] = (d, g, statistics.fmean(armv), statistics.fmean(ctrl))
        out(f'task\t{t}\tcontrol_f2e_ep30={res[t][3]:.4f}\t{arm}_ep{last}={res[t][2]:.4f}\t'
            f'delta={d:+.4f}\tpair_gains={",".join(f"{x:+.4f}" for x in g)}\tMDE={MDE[t]}\t'
            f'label={label(t, d, g)}\tv052={v[("v052", t)]:.4f}')
    if arm == 'ms2tok':
        p, close, harm = ms2tok_verdict(res['ms2'][0], res['ms2'][1], res['ms2'][2],
                                        v[('v052', 'ms2')], res['rt'][0], res['ccs'][0])
        out(f'rule\tms2tok\tprimary={p}\tTOKENS_CLOSE_GAP={close}\tHARM={harm}')
    else:
        m = {e: statistics.fmean(a[(f'rtlong{i}_ep{e}', 'rt')] for i in range(3)) for e in epochs}
        p, sat, g1, g2 = rtlong_verdict(res['rt'][0], res['rt'][1], m[30], m[45], m[60])
        out(f'rule\trtlong\tprimary={p}\tRT_SATURATING={sat}\tep30={m[30]:.4f}\tep45={m[45]:.4f}\t'
            f'ep60={m[60]:.4f}\tgain30_45={g1:+.4f}\tgain45_60={g2:+.4f}')
    out('F2F_EVAL=COMPLETE')


def self_test():
    # MS2 gain 0.05 in all pairs, mean 0.905 vs v052 0.909 -> NO_EFFECT, CLOSE_GAP, no harm.
    p, close, harm = ms2tok_verdict(0.05, [0.04, 0.05, 0.06], 0.905, 0.909, 0.1, -0.5)
    assert (p, close, harm) == ('TOKENS_NO_EFFECT', True, False)
    # One pair worse -> no CLOSE_GAP; RT worse by 0.6 MAE -> HARM.
    p, close, harm = ms2tok_verdict(0.05, [0.08, 0.08, -0.01], 0.905, 0.909, -0.6, 0.0)
    assert not close and harm
    # Large consistent MS2 gain -> TOKENS_HELP.
    assert ms2tok_verdict(0.13, [0.12, 0.13, 0.14], 0.95, 0.909, 0, 0)[0] == 'TOKENS_HELP'
    # RT: gains 0.7 all pairs, 30->45 = 0.5, 45->60 = 0.1 -> HELPS and SATURATING.
    p, sat, *_ = rtlong_verdict(0.7, [0.6, 0.7, 0.8], 4.0, 3.5, 3.4)
    assert p == 'LONG_HELPS_RT' and sat
    assert rtlong_verdict(0.2, [0.1, 0.2, 0.3], 4.0, 3.9, 3.8)[0] == 'LONG_NO_EFFECT'
    print('F2F_EVAL_SELF_TEST=PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test()
    else:
        evaluate(*sys.argv[1:5])
