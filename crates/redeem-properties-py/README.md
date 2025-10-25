# ReDeeM Properties Python Bindings

Python bindings for the ReDeeM peptide property prediction models. This package provides a Python interface to Rust-based deep learning models for predicting peptide properties in mass spectrometry.

## Features

- **Retention Time (RT) Prediction**: Predict peptide retention times using CNN-LSTM or CNN-Transformer architectures
- **Collision Cross-Section (CCS) Prediction**: Predict peptide collision cross-sections for ion mobility spectrometry
- **MS2 Fragment Intensity Prediction**: Predict MS2 fragment ion intensities using BERT-based models

## Installation

### From Source

```bash
# Install maturin (if not already installed)
pip install maturin

# Build and install the package
cd crates/redeem-properties-py
maturin develop --release
```

### Using pip (when published)

```bash
pip install redeem-properties-py
```

## Usage

### Retention Time Prediction

```python
from redeem_properties_py import RTModel

# Load a pre-trained RT model
model = RTModel(
    model_path="path/to/rt_model.safetensors",
    arch="rt_cnn_lstm",  # or "rt_cnn_tf"
    use_cuda=False  # Set to True if CUDA is available
)

# Predict retention times
sequences = ["PEPTIDE", "SEQUENCE", "EXAMPLE"]
mods = ["", "", ""]  # Modification strings (empty if no modifications)
mod_sites = ["", "", ""]  # Modification sites

rt_predictions = model.predict(sequences, mods, mod_sites)
print(f"Predicted RTs: {rt_predictions}")
```

### Collision Cross-Section Prediction

```python
from redeem_properties_py import CCSModel

# Load a pre-trained CCS model
model = CCSModel(
    model_path="path/to/ccs_model.safetensors",
    arch="ccs_cnn_lstm",  # or "ccs_cnn_tf"
    use_cuda=False
)

# Predict collision cross-sections
sequences = ["PEPTIDE", "SEQUENCE"]
mods = ["", ""]
mod_sites = ["", ""]
charges = [2, 3]  # Charge states

ccs_predictions = model.predict(sequences, mods, mod_sites, charges)
print(f"Predicted CCS: {ccs_predictions}")
```

### MS2 Fragment Intensity Prediction

```python
from redeem_properties_py import MS2Model

# Load a pre-trained MS2 model
model = MS2Model(
    model_path="path/to/ms2_model.safetensors",
    arch="ms2_bert",
    use_cuda=False
)

# Predict MS2 fragment intensities
sequences = ["PEPTIDE"]
mods = [""]
mod_sites = [""]
charges = [2]
nces = [30.0]  # Normalized collision energies

ms2_predictions = model.predict(sequences, mods, mod_sites, charges, nces)
# Returns a list of 2D numpy arrays (one per peptide)
print(f"MS2 intensities shape: {ms2_predictions[0].shape}")
```

## Model Architectures

### Retention Time Models
- `rt_cnn_lstm`: AlphaPept-style CNN-LSTM architecture
- `rt_cnn_tf`: CNN-Transformer architecture

### CCS Models
- `ccs_cnn_lstm`: AlphaPept-style CNN-LSTM architecture
- `ccs_cnn_tf`: CNN-Transformer architecture

### MS2 Models
- `ms2_bert`: BERT-based architecture for fragment intensity prediction

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- Optional: CUDA for GPU acceleration

## Development

To build the package for development:

```bash
# Install development dependencies
pip install maturin pytest

# Build in debug mode
maturin develop

# Run tests
pytest tests/
```

## License

This project is licensed under the same terms as the main ReDeeM repository.

## Citation

If you use this software in your research, please cite the ReDeeM repository.

## Links

- [Main Repository](https://github.com/singjc/redeem)
- [Documentation](https://github.com/singjc/redeem)
