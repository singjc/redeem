# `redeem-topaz`

`redeem-topaz` is the Rust/Candle implementation of the TOPAZ trace-first DIA
peak-group scorer.

The crate is responsible for:

- reading normalized feature rows from OSW via `redeem-io`
- extracting fixed-width XIC windows around `FEATURE.EXP_RT`
- optionally extracting fixed-width XIM windows around `FEATURE.EXP_IM`
- encoding those traces with convolutional branches and coelution heads
- scoring candidate rows and bags with multiple-instance learning (MIL)
- optionally training and applying XRUN cross-run calibration
- writing score tables and diagnostics for downstream inspection

## Model overview

TOPAZ scores a DIA candidate peak group by combining:

1. **Heuristic/library features**
   - Scalar scores already present in the OSW file, such as library similarity
     and RT-based features.
2. **XIC trace branch**
   - Fixed-length chromatogram windows extracted around the candidate apex RT.
   - Optional MS1 channels are ordered first, followed by MS2 fragment traces.
3. **Optional XIM trace branch**
   - Fixed-length ion-mobilogram windows extracted for the same candidate.
   - These are keyed by `FEATURE_ID`, not `PRECURSOR_ID`, because XIMs are
     candidate-specific.
4. **Coelution/statistical heads**
   - Explicit channel agreement features, apex alignment features, and related
     shape summaries computed from the XIC/XIM tensors.
5. **Candidate scorer**
   - An MLP that fuses heuristic features, learned embeddings, and coelution
     summaries into one candidate logit.
6. **Bag-level MIL**
   - Candidate rows are grouped by `(run_id, precursor_id)`.
   - The bag score is a masked max over candidate logits, matching the original
     Python semantics.

## Tensor conventions

TOPAZ uses the following tensor shape notation throughout the crate:

- `N`: number of flat candidate rows
- `B`: number of bags
- `K`: padded number of candidate slots per bag
- `D`: number of scalar heuristic features
- `L`: fixed trace length after crop/pad
- `C`: number of channels in a given trace tensor
- `C_total`: total channels presented to an encoder, usually `ms1_cmax + ms2_cmax`
- `E`: learned embedding width produced by an encoder branch

Typical inputs are:

- XIC row tensor: `(N, C_total, L)`
- XIM row tensor: `(N, C_total, L)`
- Bagged feature tensor: `(B, K, D)`
- Bagged trace tensor: `(B, K, C_total, L)`

Channel ordering is always:

1. MS1 channels
2. MS2 channels

Missing MS1 channels are zero-padded rather than treated as an error.

## XIC and XIM inputs

There are now three ways to point TOPAZ at chromatogram/mobilogram parquet
files:

1. `xic_path` / `xim_path`
   - one parquet file containing all requested runs
2. `xic_paths` / `xim_paths`
   - a list of parquet files, typically one per run
   - TOPAZ will inspect the parquet metadata and infer the run-to-path mapping
3. `xic_map_path` / `xim_map_path`
   - an explicit TSV mapping of `run_id -> parquet path`
   - this is the safest option when OpenSWATH/OSW run IDs do not line up with
     the parquet-internal run identifiers

If both list-style paths and map files are present, the explicit map takes
precedence.

## Preprocessed bundles

For large datasets, parquet decoding and fixed-window extraction can dominate
both training and inference wall time. TOPAZ can now split that work into a
separate preprocessing step.

### Workflow

1. Build a reusable bundle once:

```bash
redeem topaz preprocess topaz_preprocess.json
```

If the preprocess config omits `xim_trace`, the CLI will infer it from either
an embedded `model.xim` block or a `checkpoint` field in the same JSON file.

2. Train, infer, or XRUN-train directly from the bundle:

```bash
redeem topaz train topaz_train.json --preprocessed topaz_inputs.topazdata
redeem topaz infer topaz_infer.json --preprocessed topaz_inputs.topazdata
redeem topaz xrun-train xrun_train.json --preprocessed topaz_inputs.topazdata
```

### Bundle format

Preprocessed inputs are stored in an uncompressed zip archive, typically named
`*.topazdata`. The archive contains:

- `manifest.json`
  - preprocessing/version metadata
  - source provenance
  - feature-column order
  - trace/XIM extraction settings
  - chunk table with row counts
- `chunks/000000/...`
  - row metadata (`feature_id`, `precursor_id`, `run_id`, `exp_rt`, `exp_im`, ...)
  - heuristic feature matrix
  - XIC tensors
  - optional XIM tensors

Each chunk is written in a simple binary layout so it can be loaded quickly
without re-running parquet/MSNumpress decoding.

### Validation

TOPAZ validates the bundle before use and fails early if it does not match the
active run. The manifest checks include:

- XIC trace shape (`l`, `ms1_cmax`, `ms2_cmax`, `normalize_max`)
- optional XIM trace shape
- OSW reader configuration
- heuristic feature-column coverage
- checkpoint/model compatibility for inference and XRUN

This is deliberate: preprocessed inputs are only safe to reuse when the stored
tensor layout still matches what the model expects.

## XIM extraction semantics

For diaPASEF-style workflows, the XIM branch uses the ion-mobility metadata in
the OSW `FEATURE` table:

- centering uses `FEATURE.EXP_IM` when available
- if `EXP_IM_LEFTWIDTH` and `EXP_IM_RIGHTWIDTH` are both present and valid, the
  fixed-width mobilogram keeps only that interior region and zeros everything
  outside it
- invalid boundaries (missing, negative, or reversed) fall back to using the
  full centered mobilogram window

This keeps the input shape fixed while preserving the peak-picker result when
it is trustworthy.

## Checkpoints

TOPAZ checkpoints are stored as an uncompressed zip bundle with the extension
`topaz.model`.

The bundle can contain:

- `base.safetensors`
- `base.json`
- `xrun.safetensors`
- `xrun.json`

This packaging keeps the base model and optional XRUN calibrator together
without relying on multiple loose files.

Legacy split checkpoints (`.safetensors`, `.json`, `.xrun.safetensors`,
`.xrun.json`) are still accepted on load.

## Caching

TOPAZ supports both in-memory and optional on-disk caching for decoded parquet
payloads:

- XIC cache: keyed by `(path, run_id, precursor_id)`
- XIM cache: keyed by `(path, run_id, feature_id)`

The in-memory caches are capacity-limited LRU-style caches over decoded traces.
The optional disk caches store decoded payloads so repeated runs do not need to
re-decode MSNumpress/zlib data.

## Example utility

The crate includes a troubleshooting example that renders the exact fixed-width
XIC/XIM tensors fed into the model:

```bash
cargo run -p redeem-topaz --example inspect_inputs --features io-sqlite,io-parquet -- \
  --osw sample.osw \
  --xic-map xic_map.tsv \
  --xim-map xim_map.tsv \
  --feature-ids 101,205,309 \
  --out inspect.html
```

This is useful when checking:

- centering around `EXP_RT` / `EXP_IM`
- MS1/MS2 channel ordering
- zero-padding behavior
- boundary masking for XIM windows

## Crate layout

- `src/building_blocks`
  - reusable encoders, coelution heads, bagging, and input transforms
- `src/model`
  - the concrete TOPAZ architecture
- `src/train`
  - losses, schedulers, and trainer/pipeline logic
- `src/infer`
  - trace extraction, scoring, diagnostics, and report-oriented helpers
- `src/xrun`
  - cross-run calibrator, sequence building, and sweep logic
- `src/checkpoint.rs`
  - `topaz.model` save/load logic
- `src/run.rs`
  - high-level train/infer/XRUN entry points used by `redeem-cli`

## Intended scope

`redeem-topaz` is intentionally focused on the TOPAZ family of DIA scorers.
General file-format logic such as OSW access, parquet decoding, and MSNumpress
handling lives in [`redeem-io`](../redeem-io/README.md).
