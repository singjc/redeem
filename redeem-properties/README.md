# redeem-properties

A Rust library for peptide property prediction using deep learning, built on top of the [candle](https://github.com/huggingface/candle) ML framework.

`redeem-properties` provides high-performance, pure-Rust implementations of models for predicting:

- **Retention Time (RT)** — chromatographic elution time
- **Collision Cross Section (CCS)** — ion mobility
- **MS2 Fragment Intensities** — tandem mass spectrum prediction

## Features

- **Multiple model architectures** per property type (CNN-LSTM, CNN-Transformer, BERT)
- **Pretrained model registry** with automatic lookup and caching
- **Fine-tuning / transfer learning** with configurable training loops, early stopping, and learning rate warmup
- **CPU and CUDA** support via candle backends
- **No Python runtime required** — everything runs natively in Rust
- **Modification-aware** — supports arbitrary post-translational modifications via AlphaPeptDeep-style encoding

## Installation

Add `redeem-properties` to your `Cargo.toml`:

```toml
[dependencies]
redeem-properties = "0.1"
```

### From source (development)

To build from the latest development version with all features:

```toml
[dependencies]
redeem-properties = { git = "https://github.com/singjc/redeem.git", branch = "develop" }
```

### Feature flags

| Feature | Description |
|---------|-------------|
| `cuda` | Enable CUDA GPU acceleration via candle |
| `embed-pretrained` | Embed pretrained model weights into the binary at compile time (requires model files on disk; only works when building from source) |

```toml
[dependencies]
redeem-properties = { version = "0.1", features = ["cuda"] }
```

Or with git source:

```toml
[dependencies]
redeem-properties = { git = "https://github.com/singjc/redeem.git", branch = "develop", features = ["cuda"] }
```

## Quick Start

### Loading a pretrained model

```rust
use redeem_properties::pretrained::{locate_pretrained_model, PretrainedModel};
use redeem_properties::models::rt_model::RTModelWrapper;
use candle_core::Device;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    // Locate the pretrained model file on disk
    let model_path = locate_pretrained_model(PretrainedModel::RedeemRtCnnTf)?;
    let device = Device::Cpu;

    // Create the RT model wrapper
    let model = RTModelWrapper::new(&model_path, None::<&str>, "rt_cnn_tf", device)?;

    // Predict retention times
    let sequences: Vec<Arc<[u8]>> = vec![
        Arc::from(b"AGHCEWQMKYR".as_slice()),
        Arc::from(b"PEPTIDEK".as_slice()),
    ];
    let mods: Vec<Arc<[u8]>> = vec![Arc::from(b"".as_slice()); 2];
    let mod_sites: Vec<Arc<[u8]>> = vec![Arc::from(b"".as_slice()); 2];

    let result = model.predict(&sequences, &mods, &mod_sites)?;
    println!("{:?}", result);

    Ok(())
}
```

### Loading from a custom model file

```rust
use redeem_properties::models::rt_model::RTModelWrapper;
use candle_core::Device;

let model = RTModelWrapper::new(
    "path/to/model.safetensors",       // or .pth
    Some("path/to/model_const.yaml"),   // optional constants sidecar
    "rt_cnn_tf",                        // architecture
    Device::Cpu,
)?;
```

### CCS prediction

```rust
use redeem_properties::models::ccs_model::CCSModelWrapper;
use candle_core::Device;
use std::sync::Arc;

let model = CCSModelWrapper::new(
    "path/to/ccs.safetensors",
    None::<&str>,
    "ccs_cnn_tf",
    Device::Cpu,
)?;

let sequences = vec![Arc::from(b"AGHCEWQMKYR".as_slice())];
let mods = vec![Arc::from(b"".as_slice())];
let mod_sites = vec![Arc::from(b"".as_slice())];
let charges = vec![2];

let result = model.predict(&sequences, &mods, &mod_sites, charges)?;
```

### MS2 fragment intensity prediction

```rust
use redeem_properties::models::ms2_model::MS2ModelWrapper;
use candle_core::Device;
use std::sync::Arc;

let model = MS2ModelWrapper::new(
    "path/to/ms2.pth",
    None::<&str>,
    "ms2_bert",
    Device::Cpu,
)?;

let sequences = vec![Arc::from(b"AGHCEWQMKYR".as_slice())];
let mods = vec![Arc::from(b"".as_slice())];
let mod_sites = vec![Arc::from(b"".as_slice())];
let charges = vec![2];
let nces = vec![25];
let instruments = vec![Some(Arc::from(b"QE".as_slice()))];

let result = model.predict(
    &sequences, &mods, &mod_sites,
    charges, nces, instruments,
)?;
```

### Training / fine-tuning

All model wrappers expose a `train()` method for fine-tuning on your own data:

```rust
use redeem_properties::models::rt_model::RTModelWrapper;
use redeem_properties::models::model_interface::ModelInterface;
use redeem_properties::utils::data_handling::{PeptideData, TargetNormalization};
use candle_core::Device;

// Start from a pretrained model or create a new one
let mut model = RTModelWrapper::new(
    "path/to/rt.safetensors",
    None::<&str>,
    "rt_cnn_tf",
    Device::Cpu,
)?;

// Prepare training data as Vec<PeptideData>
let training_data: Vec<PeptideData> = /* ... */;
let val_data: Vec<PeptideData> = /* ... */;

let metrics = model.train(
    &training_data,
    Some(&val_data),
    modifications,          // HashMap of modification definitions
    64,                     // batch_size
    64,                     // val_batch_size
    1e-4,                   // learning_rate
    100,                    // epochs
    10,                     // early_stopping_patience
    TargetNormalization::MinMax(0.0, 1.0),
    None,                   // train_var_prefixes (None = train all)
    Some(0.1),              // warmup_fraction
)?;
```

## Supported Architectures

| Property | Architecture | Identifier | Struct |
|----------|-------------|------------|--------|
| RT | CNN-LSTM | `rt_cnn_lstm` | `RTCNNLSTMModel` |
| RT | CNN-Transformer | `rt_cnn_tf` | `RTCNNTFModel` |
| CCS | CNN-LSTM | `ccs_cnn_lstm` | `CCSCNNLSTMModel` |
| CCS | CNN-Transformer | `ccs_cnn_tf` | `CCSCNNTFModel` |
| MS2 | BERT | `ms2_bert` | `MS2BertModel` |

## Pretrained Models

The `pretrained` module provides a registry of bundled model identifiers:

| Identifier | Architecture | Property |
|-----------|-------------|----------|
| `AlphapeptdeepRtCnnLstm` | CNN-LSTM | RT |
| `AlphapeptdeepCcsCnnLstm` | CNN-LSTM | CCS |
| `AlphapeptdeepMs2Bert` | BERT | MS2 |
| `RedeemRtCnnTf` | CNN-Transformer | RT |
| `RedeemCcsCnnTf` | CNN-Transformer | CCS |

Model files are searched in order:

1. `$REDEEM_PRETRAINED_MODELS_DIR/<path>`
2. User-local models directory (platform-dependent; see `default_pretrained_models_dir()`)
3. `$CARGO_MANIFEST_DIR/assets/pretrained_models/<path>` (development/local builds)
4. `./assets/pretrained_models/<path>` (current working directory)

## Crate Structure

```
redeem-properties/
├── src/
│   ├── lib.rs                    # Public API re-exports
│   ├── building_blocks/          # Reusable neural network layers
│   │   ├── bilstm.rs             #   Bidirectional LSTM
│   │   ├── building_blocks.rs    #   Transformer blocks, embeddings, attention
│   │   ├── featurize.rs          #   Peptide → tensor encoding
│   │   ├── nn.rs                 #   Linear layers, activations
│   │   └── sequential.rs         #   Sequential container
│   ├── models/                   # Model implementations
│   │   ├── model_interface.rs    #   ModelInterface trait + training loop
│   │   ├── rt_model.rs           #   RTModelWrapper (arch dispatcher)
│   │   ├── rt_cnn_lstm_model.rs  #   RT CNN-LSTM implementation
│   │   ├── rt_cnn_transformer_model.rs  # RT CNN-Transformer
│   │   ├── ccs_model.rs          #   CCSModelWrapper
│   │   ├── ccs_cnn_lstm_model.rs #   CCS CNN-LSTM
│   │   ├── ccs_cnn_tf_model.rs   #   CCS CNN-Transformer
│   │   ├── ms2_model.rs          #   MS2ModelWrapper
│   │   └── ms2_bert_model.rs     #   MS2 BERT
│   ├── pretrained/               # Pretrained model registry & locator
│   │   └── mod.rs
│   └── utils/                    # Shared utilities
│       ├── data_handling.rs      #   PeptideData, batching, normalization
│       ├── peptdeep_utils.rs     #   Modification parsing, AA indices
│       ├── mz_utils.rs           #   m/z calculation helpers
│       ├── stats.rs              #   Loss metrics, training statistics
│       ├── logging.rs            #   Progress bar wrapper
│       └── utils.rs              #   LR schedulers, tensor utilities
├── examples/                     # Runnable examples
└── assets/                       # Static assets and models
    ├── modification.tsv          #   Modification metadata
    └── pretrained_models/        #   Optional embedded models (dev/binary)
```

## Model Formats

The library supports loading model weights from:

- **SafeTensors** (`.safetensors`) — recommended, memory-mapped
- **PyTorch pickle** (`.pth`, `.pt`) — for AlphaPeptDeep compatibility

An optional YAML sidecar file (`<model_path>.model_const.yaml`) can accompany any model to store normalization constants and training metadata.

## Python Bindings

For a high-level Python API, see the companion [`redeem-properties-py`](../redeem-properties-py) crate, which wraps this library via PyO3:

```bash
pip install redeem_properties
```

```python
import redeem_properties

model = redeem_properties.RTModel.from_pretrained("rt")
rt_values = model.predict(["PEPTIDE", "SEQU[+42.0106]ENCE"])
```

## License

See the repository [LICENSE](../LICENSE) file.
