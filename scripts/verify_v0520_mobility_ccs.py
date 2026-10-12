#!/usr/bin/env python3
"""Fail-closed paired native mobility/CCS comparison against historical v0.52."""
import csv
import math
import sys
from collections import Counter
from pathlib import Path

KEY = ("corpus_record_index", "sequence", "charge", "precursor_mz")
NUMERIC = ("target_native_mobility", "target_ccs", "predicted_native_mobility", "predicted_ccs", "ccs_factor")
FIELDS = KEY + NUMERIC
TOLERANCE = 1.0e-4


def read(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or set(reader.fieldnames) != set(FIELDS):
            raise ValueError(f"incorrect mobility/CCS schema: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty mobility/CCS table: {path}")
    keyed = {}
    for row in rows:
        key = row["corpus_record_index"]
        if key in keyed:
            raise ValueError(f"duplicate corpus record {key}: {path}")
        for name in NUMERIC + ("precursor_mz",):
            if not math.isfinite(float(row[name])):
                raise ValueError(f"nonfinite {name} for {key}")
        if float(row["ccs_factor"]) <= 0 or float(row["target_ccs"]) <= 0:
            raise ValueError(f"invalid CCS target/factor for {key}")
        keyed[key] = row
    return keyed


def compare(reference, native):
    a, b = read(reference), read(native)
    if a.keys() != b.keys():
        missing = sorted(a.keys() - b.keys())
        extra = sorted(b.keys() - a.keys())
        raise ValueError(f"different Validation identities; missing={missing[:5]}, extra={extra[:5]}")
    for index in a:
        for field in KEY:
            if a[index][field] != b[index][field]:
                raise ValueError(f"identity mismatch {index}: {field}")
        for field in NUMERIC:
            x, y = float(a[index][field]), float(b[index][field])
            if abs(x - y) > TOLERANCE:
                raise ValueError(f"numeric mismatch {index}: {field}: {x} vs {y}, delta={abs(x-y)}")
    print("v0520_native_mobility_ccs_parity=PASS")
    print(f"ccs_records={len(a)}")
    print(f"absolute_tolerance={TOLERANCE}")
    print("historical_test_consumed=NO")
    print("train_holdout_read_by_this_comparison=NO")


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp)/"a.tsv", Path(tmp)/"b.tsv"
        row = dict(zip(FIELDS, ["123", "PEPTIDEK", "2", "400.00000000",
                                 "0.75000000", "600.00000000", "0.76000000",
                                 "601.00000000", "790.00000000"]))
        row2 = row | {"corpus_record_index": "789", "sequence": "ANOTHERK"}
        def write(p, rows):
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
                w.writeheader(); w.writerows(rows)
        write(a,[row,row2]); write(b,[row2,row]); compare(a,b)
        write(b,[row]);
        try: compare(a,b)
        except ValueError: pass
        else: raise AssertionError("missing record accepted")
        write(b,[row,{**row2,"predicted_ccs":"602.00000000"}])
        try: compare(a,b)
        except ValueError: pass
        else: raise AssertionError("wrong CCS accepted")
        write(b,[row,row]);
        try: compare(a,b)
        except ValueError: pass
        else: raise AssertionError("duplicate accepted")
    print("v0520_mobility_ccs_comparator_self_test=PASS")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) == 3:
        compare(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        raise SystemExit("usage: verify_v0520_mobility_ccs.py --self-test | REFERENCE.tsv NATIVE.tsv")
