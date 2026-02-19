# Examples for redeem-classifiers

This folder contains runnable example binaries demonstrating how to use the models and the semi-supervised learner. Examples are written so they compile both with and without optional features (e.g., `xgboost`, `svm`).

Quick notes
- Examples that need native libraries (XGBoost) require enabling the corresponding Cargo feature: `--features xgboost`.
- Heavy imports are feature-gated inside each example so the examples compile even when optional features are not enabled — you will see a helpful message in that case.

Prerequisites
- Rust toolchain (stable or nightly with Cargo). Install via rustup: https://rustup.rs
- If you want to run XGBoost examples you need a system `libxgboost` available at runtime. Options:
  - Install via your system package manager if available.
  - Build XGBoost from source and install the shared library (libxgboost.so / libxgboost.dylib) into a standard library path or set `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` accordingly.
  - Alternatively, use conda to install XGBoost (`conda install -c conda-forge xgboost`) and point the dynamic loader to the conda `lib/` directory.

Common commands
- Build and run an example that does NOT require features (example: GBDT example):

```bash
# from repository root
cargo run --manifest-path crates/redeem-classifiers/Cargo.toml --example gbdt_semi_supervised_learning
```

- Run the XGBoost semi-supervised example (requires `xgboost` feature and libxgboost at runtime):

```bash
cargo run --manifest-path crates/redeem-classifiers/Cargo.toml --example xgb_semi_supervised_learning --features xgboost -- --scale --normalize-scores
```

Notes about the above command:
- The `--features xgboost` flag tells Cargo to compile the crate with XGBoost support.
- Arguments after the second `--` are passed to the example binary. The examples accept two optional flags that control preprocessing inside the `SemiSupervisedLearner`:
  - `--scale` — if present, the learner will scale features using a standard scaler before training.
  - `--normalize-scores` — if present, final prediction scores will be normalized (zero-mean, unit-variance) before being saved or printed.

- Run the synthetic XGBoost example (quick smoke test):

```bash
cargo run --manifest-path crates/redeem-classifiers/Cargo.toml --example xgb_synthetic --features xgboost
```

- Run the SVM semi-supervised example (requires `svm` feature):

```bash
cargo run --manifest-path crates/redeem-classifiers/Cargo.toml --example svm_semi_supervised_learning --features svm -- --scale
```

