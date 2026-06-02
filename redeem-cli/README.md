# redeem-cli

Command-line interface for [ReDeeM](../README.md), providing peptide property prediction and PSM scoring from the terminal.

The `redeem` binary wraps the [`redeem-properties`](../redeem-properties/) and [`redeem-classifiers`](../redeem-classifiers/) libraries into a single CLI with two top-level subcommands: **`properties`** and **`classifiers`**.

> [!NOTE]
> The CLI is still under development

## Installation

### From source

```bash
cargo install --path .
```

### With optional features

| Feature | Description | Extra requirements |
|---------|-------------|-------------------|
| `cuda` | GPU acceleration via candle CUDA backend | CUDA toolkit |
| `svm` | Enable SVM classifier backend | None |
| `xgboost` | Enable XGBoost classifier backend | `clang`, `libstdc++-dev` |

```bash
# Build with XGBoost support
cargo install --path . --features xgboost

# Build with all optional classifiers
cargo install --path . --features "xgboost,svm"
```

## Usage

```
redeem <COMMAND>

Commands:
  properties   Train or run peptide property prediction models
  classifiers  Run classification tools such as scoring PSMs
```

### Properties — Train

Train a new peptide property prediction model (RT, CCS, or MS2) from scratch or fine-tune from a checkpoint.

```bash
redeem properties train <CONFIG> \
    [-d <TRAIN_DATA>] \
    [-v <VALIDATION_DATA>] \
    [-o <OUTPUT_FILE>] \
    [-m <MODEL_ARCH>] \
    [-c <CHECKPOINT_FILE>]
```

| Argument | Description |
|----------|-------------|
| `CONFIG` | Path to a JSON training configuration file (required) |
| `-d`, `--train_data` | Override the training data path from the config |
| `-v`, `--validation_data` | Override the validation data path from the config |
| `-o`, `--output_file` | Override the output path for the trained `.safetensors` model |
| `-m`, `--model_arch` | Override the model architecture (`rt_cnn_lstm`, `rt_cnn_tf`, `ms2_bert`, `ccs_cnn_lstm`) |
| `-c`, `--checkpoint_file` | Path to a `.safetensors` checkpoint to resume from |

**Example:**

```bash
redeem properties train config.json \
    -d train.tsv \
    -v val.tsv \
    -o model.safetensors \
    -m rt_cnn_tf
```

### Properties — Inference

Predict peptide properties using a trained model or a built-in pretrained model.

```bash
redeem properties inference <CONFIG> \
    [--pretrained <NAME>] \
    [-m <MODEL_PATH>] \
    [-d <INFERENCE_DATA>] \
    [-o <OUTPUT_FILE>]
```

| Argument | Description |
|----------|-------------|
| `CONFIG` | Path to a JSON inference configuration file (required) |
| `--pretrained` | Use a pretrained model by name (e.g. `redeem-rt`, `redeem-ccs`, `alphapeptdeep-ccs`). This also overrides `model_arch` to the matching architecture. |
| `-m`, `--model` | Path to a custom trained `.safetensors` model |
| `-d`, `--inference_data` | Override the input data path from the config |
| `-o`, `--output_file` | Override the output prediction file path (`.tsv` or `.csv`) |

**Example:**

```bash
redeem properties inference inference_config.json \
    --pretrained redeem-rt \
    -d peptides.tsv \
    -o predictions.tsv
```

### Classifiers — Score

Score a Percolator `.pin` file using the semi-supervised classifier.

```bash
redeem classifiers score <PIN> [CONFIG] \
    [-o <OUTPUT>] \
    [--rank-grouping <STRATEGY>] \
    [--model-type <TYPE>] \
    [--dedup] \
    [--no-report]
```

| Argument | Description |
|----------|-------------|
| `PIN` | Path to a Percolator `.pin` input file (required) |
| `CONFIG` | Optional path to a JSON classifier configuration file |
| `-o`, `--output` | Output path for the scored TSV (defaults to stdout) |
| `--rank-grouping` | Grouping strategy: `percolator` or `spec-id` |
| `--model-type` | Override classifier type: `gbdt`, `xgboost`, or `svm` |
| `--dedup` | Deduplicate PSMs in the final output |
| `--no-report` | Disable HTML report generation |

**Example:**

```bash
redeem classifiers score results.pin \
    -o scored.tsv \
    --model-type gbdt \
    --rank-grouping spec-id
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDEEM_LOG` | `error,redeem=info` | Controls log verbosity via `env_logger` filter syntax |

```bash
# Enable debug logging for all redeem modules
REDEEM_LOG=debug redeem properties inference config.json
```

## License

See the repository [LICENSE](../LICENSE) file.
