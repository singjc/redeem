#!/usr/bin/env python3
"""F1-F: what kind of unified MS2 error explains the cosine gap to v0.52 (job 62823987 files).

Preregistered in the rolling handoff (2026-11-01 02:52 UTC). CPU, descriptive, no new inference.
Inputs (same directory): foundation_comparison_fragments.tsv (v0.52 legacy, per fragment),
production_validation_2048.tsv (unified, dense MS2 per Validation ordinal),
paired_ms2_records.tsv (matched 404), validation_first2048.tsv (ordinal -> corpus index).
Self-check: per-record cosine must reproduce paired_ms2_records.tsv.
"""
import csv, json, math, sys
from collections import defaultdict

def cos(p, t):
    d = sum(a * b for a, b in zip(p, t))
    n = math.sqrt(sum(a * a for a in p) + 1e-8) * math.sqrt(sum(b * b for b in t) + 1e-8)
    return d / n

def main():
    order = [int(l.split('\t')[0]) for l in open('validation_first2048.tsv').read().split('\n') if l]
    ordinal = {idx: o for o, idx in enumerate(order)}
    dense = {}
    for r in csv.DictReader(open('production_validation_2048.tsv'), delimiter='\t'):
        if r['predicted_ms2']:
            dense[int(r['record_index'])] = json.loads(r['predicted_ms2'])
    paired = {int(r['record_index']): r for r in csv.DictReader(open('paired_ms2_records.tsv'), delimiter='\t')}
    frags = defaultdict(list)  # record -> [(cleavage, channel, target, legacy, length)]
    seqlen = {}
    for r in csv.DictReader(open('foundation_comparison_fragments.tsv'), delimiter='\t'):
        i = int(r['record_index'])
        if i in paired:
            frags[i].append((int(r['cleavage_index']), int(r['channel']), float(r['target_intensity']),
                             float(r['foundation_intensity'])))
            seqlen[i] = len(r['sequence'])
    rows = {}
    max_dev = 0.0
    for i, fl in frags.items():
        u = dense[ordinal[i]]
        t = [f[2] for f in fl]; l = [f[3] for f in fl]; p = [u[f[0]][f[1]] for f in fl]; ch = [f[1] for f in fl]
        rows[i] = (t, l, p, ch)
        max_dev = max(max_dev, abs(cos(l, t) - float(paired[i]['legacy_cosine'])),
                      abs(cos(p, t) - float(paired[i]['production_cosine'])))
    assert len(rows) == len(paired) == 404, len(rows)
    print(f'self_check_records\t{len(rows)}\nself_check_max_cosine_deviation\t{max_dev:.2e}')
    assert max_dev < 1e-4, 'cosine does not reproduce paired_ms2_records.tsv'

    def mean_cos(ids, f):
        return sum(cos(f(rows[i]), rows[i][0]) for i in ids) / len(ids)
    channels = sorted({c for r in rows.values() for c in r[3]})

    def fit_scale(ids):  # per-channel least-squares multiplier on unified predictions
        num = defaultdict(float); den = defaultdict(float)
        for i in ids:
            t, _, p, ch = rows[i]
            for a, b, c in zip(t, p, ch):
                num[c] += a * b; den[c] += b * b
        return {c: num[c] / den[c] if den[c] else 1.0 for c in channels}

    gammas = [0.5 + 0.05 * k for k in range(31)]
    def fit_gamma(ids, scale=None):
        def f(g):
            return lambda r: [(s * max(x, 0.0)) ** g for x, s in zip(r[2], [scale[c] if scale else 1.0 for c in r[3]])]
        return max(gammas, key=lambda g: mean_cos(ids, f(g))), f

    ids = sorted(rows)
    halves = ([i for i in ids if i % 2 == 0], [i for i in ids if i % 2 == 1])
    res = defaultdict(list)
    for fit, ev in (halves, halves[::-1]):
        raw = mean_cos(ev, lambda r: r[2]); leg = mean_cos(ev, lambda r: r[1])
        sc = fit_scale(fit)
        a = mean_cos(ev, lambda r: [x * sc[c] for x, c in zip(r[2], r[3])])
        g, f = fit_gamma(fit); b = mean_cos(ev, f(g))
        g2, f2 = fit_gamma(fit, sc); c2 = mean_cos(ev, f2(g2))
        for k, v in (('raw', raw), ('legacy', leg), ('a_scale', a), ('b_gamma', b), ('c_both', c2),
                     ('gamma_b', g), ('gamma_c', g2)):
            res[k].append(v)
        res['scales'].append(sc)
    m = {k: sum(v) / 2 for k, v in res.items() if k != 'scales'}
    gap = m['legacy'] - m['raw']
    print(f"heldout_mean_cos_legacy\t{m['legacy']:.5f}\nheldout_mean_cos_unified_raw\t{m['raw']:.5f}\ngap\t{gap:.5f}")
    for k in ('a_scale', 'b_gamma', 'c_both'):
        print(f"recovery_{k}\t{(m[k] - m['raw']) / gap:.3f}\tcos={m[k]:.5f}")
    print(f"fitted_gamma_b\t{res['gamma_b']}\nfitted_gamma_c\t{res['gamma_c']}")
    print('per_channel_scale_fits\t' + json.dumps([{c: round(s[c], 3) for c in channels} for s in res['scales']]))
    best = max((m[k] - m['raw']) / gap for k in ('a_scale', 'b_gamma', 'c_both'))
    verdict = 'EXPLAINS' if best >= 0.5 else 'PER_FRAGMENT' if best < 0.2 else 'PARTIAL'
    print(f'F1F_VERDICT\t{verdict}\tbest_recovery={best:.3f}')

    # Descriptive only.
    for c in channels:
        t = [x for r in rows.values() for x, cc in zip(r[0], r[3]) if cc == c]
        l = [x for r in rows.values() for x, cc in zip(r[1], r[3]) if cc == c]
        p = [x for r in rows.values() for x, cc in zip(r[2], r[3]) if cc == c]
        print(f'desc_channel\t{c}\tn={len(t)}\tmean_target={sum(t)/len(t):.4f}\tmean_v052={sum(l)/len(l):.4f}\tmean_unified={sum(p)/len(p):.4f}')
    lens = sorted(seqlen[i] for i in ids); q = (lens[len(lens) // 3], lens[2 * len(lens) // 3])
    for name, sel in (('short', lambda n: n <= q[0]), ('mid', lambda n: q[0] < n <= q[1]), ('long', lambda n: n > q[1])):
        s = [i for i in ids if sel(seqlen[i])]
        print(f'desc_length\t{name}\tn={len(s)}\tcos_v052={mean_cos(s, lambda r: r[1]):.4f}\tcos_unified={mean_cos(s, lambda r: r[2]):.4f}')
    print('F1F_ANALYSIS=COMPLETE')

if __name__ == '__main__':
    sys.exit(main())
