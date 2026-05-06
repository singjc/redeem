<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/singjc/redeem/raw/develop/img/redeem_logo_new.png" alt="ReDeem_Logo" width="500">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/singjc/redeem/raw/develop/img/redeem_logo_new_dark.png" alt="ReDeem_Logo" width="500">
    <img alt="ReDeem Logo" comment="Placeholder to transition between light color mode and dark color mode - this image is not directly used." src="https://github.com/singjc/redeem/raw/develop/img/redeem_logo_new_dark.png">
  </picture>
</p>

---

# ReDeeM: Repository for Deep Learning Models for Mass Spectrometry

[![Rust](https://github.com/singjc/redeem/actions/workflows/rust.yml/badge.svg)](https://github.com/singjc/redeem/actions/workflows/rust.yml)
[![PyPI](https://img.shields.io/pypi/v/redeem_properties.svg)](https://pypi.org/project/redeem_properties/)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://redeem-properties.readthedocs.io/)

ReDeeM is a Rust workspace for mass spectrometry proteomics, providing deep learning models for peptide property prediction and machine learning classifiers for PSM rescoring. It is designed to be used as a library in other tools (e.g. [Sage](https://github.com/lazear/sage)).

## Crates

| Crate | Description | Status | Docs |
|-------|-------------|--------|------|
| [`redeem-cli`](redeem-cli/) | Command-line interface for ReDeeM | *Not published* | [README](redeem-cli/) |
| [`redeem-classifiers`](redeem-classifiers/) | Semi-supervised PSM rescoring (GBDT, XGBoost, SVM) | *Not published* | [README](redeem-classifiers/) |
| [`redeem-openms-ffi`](redeem-openms-ffi/) | Prebuilt C ABI bridge for OpenMS integration | *Not published* | [README](redeem-openms-ffi/) |
| [`redeem-properties`](redeem-properties/) | Peptide property prediction (RT, CCS, MS2) using candle | [![Crates.io](https://img.shields.io/crates/v/redeem-properties.svg)](https://crates.io/crates/redeem-properties) | [README](redeem-properties/) |
| [`redeem-properties-py`](redeem-properties-py/) | Python bindings for `redeem-properties` via PyO3 | [![PyPI](https://img.shields.io/pypi/v/redeem_properties.svg)](https://pypi.org/project/redeem_properties/) | [README](redeem-properties-py/) · [Docs](https://redeem-properties.readthedocs.io/) |

## Installation

### Rust

`redeem-properties` is available on crates.io. Other crates are still in development.

```toml
[dependencies]
redeem-properties = "0.1"
redeem-classifiers = { git = "https://github.com/singjc/redeem.git", branch = "develop" }
```

### Python

```bash
pip install redeem_properties
```

For DataFrame output support:

```bash
pip install "redeem_properties[pandas]"   # or [polars]
```

### OpenMS FFI bundles

Prebuilt `redeem-openms-ffi` static-library bundles are published as GitHub
release assets for supported OpenMS target platforms. OpenMS consumes those
artifacts automatically by platform when `WITH_REDEEM=ON`:

```bash
cmake -S /path/to/OpenMS -B build \
  -DWITH_REDEEM=ON
```

You can pin a specific release with `-DREDEEM_FFI_VERSION=v0.1.0`, or override
the download entirely with `-DREDEEM_FFI_LIBRARY=/path/to/libredeem_openms_ffi.a`.
See [redeem-openms-ffi/README.md](redeem-openms-ffi/README.md) for bundle
contents, workflow-dispatch testing, and manual packaging instructions.

## Quick Example

### Python

```python
import redeem_properties as rp

# Create a unified predictor (loads pretrained models by default)
model = rp.PropertyPrediction()

peptides = [
    "SKEEET[+79.9663]SIDVAGKP",
    "LPILVPSAKKAIYM",
    "RTPKIQVYSRHPAE",
]

df = model.predict_df(
    peptides,
    charges=[2, 3],
    nces=20,
    instruments="timsTOF",
    annotate_mz=True,
    annotate_mobility=True
)
```

```python
>>> df.head()
                    peptide  charge  nce instrument         rt         ccs  ion_mobility  precursor_mz ion_type  fragment_charge  ordinal    intensity          mz
0  SKEEET[+79.9663]SIDVAGKP       2   20    timsTOF  26.884516  535.355408      1.324961     785.35581        b                1        2   790.534363  216.134268
1  SKEEET[+79.9663]SIDVAGKP       2   20    timsTOF  26.884516  535.355408      1.324961     785.35581        b                1        3   822.035767  345.176861
2  SKEEET[+79.9663]SIDVAGKP       2   20    timsTOF  26.884516  535.355408      1.324961     785.35581        b                1        4  1272.754517  474.219454
3  SKEEET[+79.9663]SIDVAGKP       2   20    timsTOF  26.884516  535.355408      1.324961     785.35581        b                1        5  1806.533691  603.262047
4  SKEEET[+79.9663]SIDVAGKP       2   20    timsTOF  26.884516  535.355408      1.324961     785.35581        y                1        9   218.158798  967.449573
```

See the [Python bindings README](redeem-properties-py/README.md) for full usage, including MS2 prediction, DataFrame output, and the unified `PropertyPrediction` helper.

### Rust

```rust
use redeem_properties::pretrained::{locate_pretrained_model, PretrainedModel};
use redeem_properties::models::rt_model::RTModelWrapper;
use candle_core::Device;
use std::sync::Arc;

let model_path = locate_pretrained_model(PretrainedModel::RedeemRtCnnTf)?;
let model = RTModelWrapper::new(&model_path, None::<&str>, "rt_cnn_tf", Device::Cpu)?;

let sequences = vec![Arc::from(b"PEPTIDEK".as_slice())];
let mods = vec![Arc::from(b"".as_slice())];
let mod_sites = vec![Arc::from(b"".as_slice())];
let result = model.predict(&sequences, &mods, &mod_sites)?;
```

See each crate's README for detailed API documentation and examples.
