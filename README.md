<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/singjc/redeem/raw/develop/img/redeem_logo_new.png" alt="ReDeem_Logo" width="500">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/singjc/redeem/raw/develop/img/redeem_logo_new.png" alt="ReDeem_Logo" width="500">
    <img alt="ReDeem Logo" comment="Placeholder to transition between light color mode and dark color mode - this image is not directly used." src="https://github.com/singjc/redeem/raw/develop/img/redeem_logo_new.png">
  </picture>
</p>

---

# ReDeeM: Repository for Deep Learning Models for Mass Spectrometry

ReDeeM is a Rust crate designed for implementing deep learning models specifically tailored for mass spectrometry data. The primary goal of this project is to facilitate the prediction of peptide properties and to develop classifier scoring models (TDA). 

### Usage

#### Rust

The ReDeeM crates are designed to be used as a library in other projects, i.e. in Sage. To use the ReDeeM crates, add the following to your `Cargo.toml` file:

```toml
[dependencies]
redeem-properties = { git = "https://github.com/singjc/redeem.git", branch = "master" }
redeem-classifiers = { git = "https://github.com/singjc/redeem.git", branch = "master" }
```

**Note**: The ReDeeM crates are still under development and are not yet available on crates.io.

#### Python

Python bindings are now available for the peptide property prediction models. Install using pip (once published) or build from source:

```bash
# Install from source
cd crates/redeem-properties-py
pip install maturin
maturin develop --release

# Or install from PyPI (when available)
pip install redeem-properties-py
```

Example usage in Python:

```python
from redeem_properties_py import RTModel

# Load a pre-trained RT model
model = RTModel(
    model_path="path/to/rt_model.safetensors",
    arch="rt_cnn_lstm",
    use_cuda=False
)

# Predict retention times
sequences = ["PEPTIDE", "SEQUENCE"]
mods = ["", ""]
mod_sites = ["", ""]
rt_predictions = model.predict(sequences, mods, mod_sites)
```

See the [Python bindings README](crates/redeem-properties-py/README.md) for more details.

### Current Crates

The ReDeeM project consists of three primary crates:

1. **redeem-properties**: 
   - This crate focuses on deep learning models for peptide property prediction. It implements models for predicting retention time (RT), ion mobility (IM), and MS2 fragment intensities using the Candle library.
   - The models can be trained, fine-tuned on new data and can be saved in the safetensor format for later use.
   
   - Current Models
  
    Model | Name | Architecture | Implemented
    --- | --- | --- | ---
    AlphaPept RT Model | `rt_cnn_lstm` | CNN-LSTM | :heavy_check_mark:
    AlphaPept MS2 Model | `ms2_bert` | Bert | :heavy_check_mark:
    AlphaPept CCS Model | `ccs_cnn_lstm` | CNN-LSTM | :heavy_check_mark:
    RT Model | `rt_tf_lstm` | CNN-Transformer | :heavy_check_mark:
    CCS Model | `ccs_tf_lstm` | CNN-Transformer | :heavy_check_mark:

2. **redeem-classifiers**:
   - This crate is aimed at developing semi-supervised scoring classifier models. The goal is to create models for separating target peptides from decoys.
  
   - Current Models
  
    Model | Name | Architecture | Implemented
    --- | --- | --- | ---
    XGBoost Classifier | `redeem_classifiers::XGBoostClassifier` | XGBoost | :heavy_check_mark:
    GBDT Classifier | `redeem_classifiers::GBDTClassifier` | GBDT | :heavy_check_mark:
    SVM Classifier | `redeem_classifiers::SVMClassifier` | SVM | :heavy_check_mark:

3. **redeem-properties-py**:
   - Python bindings for the redeem-properties crate, providing easy access to peptide property prediction models from Python.
   - Exposes RTModel, CCSModel, and MS2Model classes for retention time, collision cross-section, and MS2 fragment intensity prediction.
   - Built using PyO3 and maturin for seamless Rust-Python interoperability.
   - See [Python bindings documentation](crates/redeem-properties-py/README.md) for usage examples.

> [!NOTE]
> To use the XGBoost classifier, or the SVM classifier, you need to compile with the `--features xgboost` or `--features linfa` flag respectively.

> [!IMPORTANT]
> The XGBoost crate is a wrapper around the original XGBoost library, which requires clang/c++ to be installed on the system. On Ubuntu, you can do the following:
    
    ```bash
    sudo apt update
    sudo apt install build-essential
    sudo apt install clang
    sudo apt install libstdc++-12-dev
    ```
