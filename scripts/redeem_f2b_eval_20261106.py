#!/usr/bin/env python3
"""F2-B step-1 evaluation: preregistered rules 2-5 (rule text: handoff 2026-11-02 14:45).

Input: scores.tsv from job 22808699 (labels r{0,1,2}_ep{6,8,10}) and v0.52 source means from
F2-A scores (15264522). Metric per task = mean over sources (equal weight), same as F2-A.
"""
import statistics, sys
from collections import defaultdict

F2A_MDE = {'rt': 0.25, 'ccs': 0.6, 'ms2': 0.01}
LOWER = {'rt': True, 'ccs': True, 'ms2': False}


def source_means(path, labels):
    acc = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        if line.startswith('#') or line.startswith('label\t'):
            continue
        label, task, source, _rec, target, pred, _ = line.rstrip('\n').split('\t')
        if label not in labels or pred == 'NA':
            continue
        v = abs(float(pred) - float(target)) if task in ('rt', 'ccs') else float(pred)
        acc[(label, task)][source].append(v)
    return {k: statistics.fmean(statistics.fmean(v) for v in s.values()) for k, s in acc.items()}


def main():
    new, old = sys.argv[1], sys.argv[2]
    labels = {f'r{r}_ep{e}' for r in range(3) for e in (6, 8, 10)}
    m = source_means(new, labels)
    v = source_means(old, {'v052'})
    print('metric\ttask\t' + '\t'.join(f'r{r}' for r in range(3)))
    for e in (6, 8, 10):
        for t in ('rt', 'ccs', 'ms2'):
            print(f'ep{e}\t{t}\t' + '\t'.join(f"{m[(f'r{r}_ep{e}', t)]:.4f}" for r in range(3)))
    print()
    for t in ('rt', 'ccs', 'ms2'):
        vals = [m[(f'r{r}_ep10', t)] for r in range(3)]
        mean, sd = statistics.fmean(vals), statistics.stdev(vals)
        mde = max(F2A_MDE[t], 2 * sd)
        gap = abs(m[('r0_ep10', t)] - v[('v052', t)])
        half_gap = gap / 2
        # Rule 5b: epoch 10 better than epoch 8 in every replicate.
        better = [(m[(f'r{r}_ep10', t)] < m[(f'r{r}_ep8', t)]) if LOWER[t]
                  else (m[(f'r{r}_ep10', t)] > m[(f'r{r}_ep8', t)]) for r in range(3)]
        vgap = [abs(x - v[('v052', t)]) for x in vals]
        print(f'rule\t{t}\tseed_mean={mean:.4f}\tseed_sd={sd:.4f}\tcv={100*sd/mean:.1f}%\t'
              f'mde_one_seed={mde:.4f}\tv052={v[("v052", t)]:.4f}\tgap_r0={gap:.4f}\t'
              f'gap_range={min(vgap):.4f}-{max(vgap):.4f}\t'
              f'rule4={"MULTI_SEED_REQUIRED" if 2*sd > half_gap else "single_seed_ok"}\t'
              f'ep10_better_than_ep8={sum(better)}/3')
    print('F2B_EVAL=COMPLETE')


if __name__ == '__main__':
    main()
