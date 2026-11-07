#!/usr/bin/env python3
"""F2-D: mark which TRAIN cohort records the unified property stages saw.

Rebuilds the exact research-curriculum property batches: for each epoch e,
order = TRAIN indices (benchmark order) shuffled with xorshift seed (seed ^ e),
cycles = min(len // (2 * pbs), max_cycles), seen = order[: cycles * 2 * pbs].
Same code as training.rs `shuffle_indices` + `research_property_batches`.
CCS uses consensus examples (identity aggregates), so per-record "seen" is not
defined for CCS and is not reported.

usage: f2d_seen.py BENCHMARK.tsv COHORT.tsv SEED=EXPOSURE.tsv [SEED=EXPOSURE.tsv...]
       (pbs=16, cycles=1024, epochs=10). Fails unless the rebuilt cumulative unique count
       equals the trainer's training_exposure.tsv for every epoch.
       f2d_seen.py --self-test
"""
import sys

MASK = (1 << 64) - 1


def shuffle(values, seed):
    state = max(seed, 1)
    for index in range(len(values) - 1, 0, -1):
        state ^= (state << 13) & MASK
        state ^= state >> 7
        state ^= (state << 17) & MASK
        swap = state % (index + 1)
        values[index], values[swap] = values[swap], values[index]


def seen_set(train, seed, epochs=10, pbs=16, max_cycles=1024, per_epoch=None):
    seen = set()
    for epoch in range(epochs):
        order = list(train)
        shuffle(order, seed ^ epoch)
        cycles = min(len(order) // (2 * pbs), max_cycles)
        seen.update(order[: cycles * 2 * pbs])
        if per_epoch is not None:
            per_epoch.append(len(seen))
    return seen


def self_test():
    # Determinism, permutation and batch-count checks. Exact Rust parity is checked in main()
    # against the trainer's own training_exposure.tsv.
    v = list(range(5))
    shuffle(v, 1)
    w = list(range(5))
    shuffle(w, 1)
    assert v == w and sorted(v) == list(range(5)) and v != list(range(5))
    assert seen_set(range(100), 3, epochs=1, pbs=2, max_cycles=5).__len__() == 20
    assert len(seen_set(range(100), 3, epochs=50, pbs=2, max_cycles=5)) > 20
    print('F2D_SEEN_SELF_TEST=PASS')


def main():
    if sys.argv[1:] == ['--self-test']:
        return self_test()
    bench, cohort, *specs = sys.argv[1:]
    train = [int(l.split('\t')[0]) for l in open(bench)
             if not l.startswith('#') and l.split('\t')[2] == 'train']
    rows = [l.rstrip('\n').split('\t') for l in open(cohort)
            if not l.startswith('#') and not l.startswith('task\t')]
    sets = {}
    for spec in specs:
        s, exposure = spec.split('=', 1)
        logged = [int(l.split('\t')[6]) for l in open(exposure) if not l.startswith('epoch')]
        rebuilt = []
        sets[s] = seen_set(train, int(s), per_epoch=rebuilt)
        assert rebuilt == logged, f'seed {s}: rebuilt {rebuilt} != logged {logged}'
    seeds = list(sets)
    print('seed\tunique_seen\ttrain_records\tfraction')
    for s, st in sets.items():
        print(f'{s}\t{len(st)}\t{len(train)}\t{len(st) / len(train):.4f}')
    print('task\tsource\tcorpus_record_index\t' + '\t'.join(f'seen_{s}' for s in seeds))
    for task, source, idx in rows:
        if task == 'ccs':
            continue
        print(f'{task}\t{source}\t{idx}\t' + '\t'.join(str(int(int(idx) in sets[s])) for s in seeds))
    print('F2D_SEEN=COMPLETE')


if __name__ == '__main__':
    main()
