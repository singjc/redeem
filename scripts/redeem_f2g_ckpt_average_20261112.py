#!/usr/bin/env python3
"""F2-G: uniform weight average of safetensors checkpoints (numpy only, F32 only).

Rule: rolling handoff 2026-11-12 21:52. Every tensor is the element-wise f32 mean of the
inputs (accumulated in f64, cast once). Tensor names, shapes, dtypes and the header key
order of the FIRST input are kept. metadata.yaml is copied from the LAST input, so
completed_epochs stays that of the newest checkpoint.

usage: redeem_f2g_ckpt_average_20261112.py OUT_DIR CKPT_DIR [CKPT_DIR ...]
       redeem_f2g_ckpt_average_20261112.py --self-test
"""
import json, os, shutil, struct, sys, tempfile
import numpy as np


def read(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    meta = header.pop("__metadata__", None)
    tensors = {}
    for name, spec in header.items():
        if spec["dtype"] != "F32":
            raise SystemExit(f"unsupported dtype {spec['dtype']} for {name}")
        a, b = spec["data_offsets"]
        tensors[name] = np.frombuffer(data[a:b], dtype="<f4").reshape(spec["shape"])
    return list(header), tensors, meta


def write(path, order, tensors, meta):
    header, blobs, offset = {}, [], 0
    if meta is not None:
        header["__metadata__"] = meta
    for name in order:
        raw = np.ascontiguousarray(tensors[name], dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(tensors[name].shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)  # safetensors pads the header to 8 bytes
    with open(path, "xb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for raw in blobs:
            f.write(raw)


def average(out_dir, ckpts):
    order, first, meta = read(os.path.join(ckpts[0], "model.safetensors"))
    acc = {k: v.astype(np.float64) for k, v in first.items()}
    for c in ckpts[1:]:
        o, t, _ = read(os.path.join(c, "model.safetensors"))
        if sorted(o) != sorted(order) or any(t[k].shape != acc[k].shape for k in order):
            raise SystemExit(f"tensor set or shape mismatch in {c}")
        for k in order:
            acc[k] += t[k]
    mean = {k: (v / len(ckpts)).astype(np.float32) for k, v in acc.items()}
    os.makedirs(out_dir, exist_ok=False)
    write(os.path.join(out_dir, "model.safetensors"), order, mean, meta)
    shutil.copyfile(os.path.join(ckpts[-1], "metadata.yaml"), os.path.join(out_dir, "metadata.yaml"))
    with open(os.path.join(out_dir, "average_sources.txt"), "x") as f:
        f.write("".join(c + "\n" for c in ckpts))


def self_test():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(1)
        t = {"a.weight": rng.standard_normal((3, 5)).astype(np.float32),
             "b.bias": rng.standard_normal(7).astype(np.float32)}
        for i in range(3):
            os.makedirs(f"{d}/c{i}")
            write(f"{d}/c{i}/model.safetensors", ["b.bias", "a.weight"], t, None)
            open(f"{d}/c{i}/metadata.yaml", "w").write(f"completed_epochs: {i}\n")
        # 1. Header round-trip: read then write gives the same bytes.
        o, r, m = read(f"{d}/c0/model.safetensors")
        write(f"{d}/rt.safetensors", o, r, m)
        assert open(f"{d}/rt.safetensors", "rb").read() == open(f"{d}/c0/model.safetensors", "rb").read()
        # 2. The mean of identical copies is byte-identical to the input.
        average(f"{d}/same", [f"{d}/c0", f"{d}/c1", f"{d}/c2"])
        assert open(f"{d}/same/model.safetensors", "rb").read() == open(f"{d}/c0/model.safetensors", "rb").read()
        assert open(f"{d}/same/metadata.yaml").read() == "completed_epochs: 2\n"
        # 3. A known two-checkpoint mean is exact.
        u = {"a.weight": np.full((3, 5), 1.0, np.float32), "b.bias": np.full(7, -2.0, np.float32)}
        w = {"a.weight": np.full((3, 5), 3.0, np.float32), "b.bias": np.full(7, 4.0, np.float32)}
        for name, tens in (("u", u), ("w", w)):
            os.makedirs(f"{d}/{name}")
            write(f"{d}/{name}/model.safetensors", ["a.weight", "b.bias"], tens, None)
            open(f"{d}/{name}/metadata.yaml", "w").write("x: 1\n")
        average(f"{d}/uw", [f"{d}/u", f"{d}/w"])
        _, got, _ = read(f"{d}/uw/model.safetensors")
        assert np.all(got["a.weight"] == 2.0) and np.all(got["b.bias"] == 1.0)
        # 4. A shape mismatch fails closed.
        os.makedirs(f"{d}/bad")
        write(f"{d}/bad/model.safetensors", ["a.weight", "b.bias"],
              {"a.weight": np.zeros((5, 3), np.float32), "b.bias": np.zeros(7, np.float32)}, None)
        open(f"{d}/bad/metadata.yaml", "w").write("x: 1\n")
        try:
            average(f"{d}/never", [f"{d}/u", f"{d}/bad"])
            raise AssertionError("shape mismatch not caught")
        except SystemExit:
            pass
    print("F2G_AVERAGE_SELF_TEST=PASS")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) >= 4:
        average(sys.argv[1], sys.argv[2:])
        print(f"F2G_AVERAGE=COMPLETE n={len(sys.argv) - 2} out={sys.argv[1]}")
    else:
        raise SystemExit(__doc__)
