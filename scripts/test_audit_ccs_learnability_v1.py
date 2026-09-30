#!/usr/bin/env python3
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "audit_ccs_learnability_v1.py"
spec = importlib.util.spec_from_file_location("audit_ccs_learnability_v1", TARGET)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.self_test()
