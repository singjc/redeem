# `redeem-io`

`redeem-io` contains the storage and interchange code shared by ReDeeM models.

The crate keeps file-format concerns separate from model code so model crates
can operate on typed Rust structs rather than SQLite rows, parquet records, or
compressed binary payloads.

## Responsibilities

`redeem-io` currently provides:

- OSW SQLite readers and score-table writers
- XIC parquet readers for OpenMS chromatogram exports
- XIM parquet readers for OpenMS ion-mobilogram exports
- MSNumpress decoding for OpenMS parquet payloads
- shared in-memory domain types for chromatograms and mobilograms

## Main data types

### OSW rows

`osw.rs` normalizes OSW feature-table rows into `FeatureRow`.

That row contains:

- identifiers: `feature_id`, `precursor_id`, `run_id`
- grouping key: `group_id`
- peak apexes: `exp_rt`, optional `exp_im`
- optional XIM boundaries: `exp_im_left_width`, `exp_im_right_width`
- decoy flag
- selected scalar feature values

The goal is that downstream code does not need to know whether the source table
was `FEATURE_MS2`, `FEATURE_MS1`, or `FEATURE_TRANSITION`.

### XICs

`xic.rs` models precursor chromatograms:

- `PrecursorXic`
  - all traces belonging to one precursor in one run
- `TransitionTrace`
  - one chromatogram channel, such as a fragment ion or precursor isotope
- `XicPoint`
  - one RT/intensity sample

These are keyed by `(run_id, precursor_id)`.

### XIMs

`xim.rs` models candidate-specific ion mobilograms:

- `FeatureXim`
  - all mobilogram traces belonging to one feature in one run
- `MobilogramTrace`
  - one mobilogram channel
- `XimPoint`
  - one mobility/intensity sample

These are keyed by `(run_id, feature_id)`, because mobilograms are extracted at
the RT apex of a specific candidate peak group rather than being shared across
all candidates for a precursor.

## Parquet readers

### XIC parquet

`xic_parquet.rs` reads OpenMS `*.xic` parquet files. These files usually contain
one row per chromatogram trace, with compressed RT and intensity arrays.

The reader supports builder-style filters such as:

- precursor ID
- run ID
- MS level
- detecting-transition flag
- decoy flag

### XIM parquet

`xim_parquet.rs` reads OpenMS `*.xim` parquet files. These are structurally
similar to `*.xic` files, but they carry mobility/intensity arrays and are
grouped by `FEATURE_ID`.

The reader supports builder-style filters such as:

- feature ID
- run ID
- MS level
- mobilogram type
- detecting-transition flag
- decoy flag

For TOPAZ the important mobilogram types are currently:

- `ms1`
- `ms2`

Other mobilogram types can be added later if the model begins consuming them.

## Compression handling

OpenMS parquet payloads may be stored as:

- raw doubles
- zlib-compressed raw doubles
- MSNumpress linear payloads
- MSNumpress slof payloads
- or zlib-wrapped MSNumpress payloads

`msnumpress.rs` contains the Rust decoder used by both parquet readers.

## Typical usage

From another crate:

```rust
use redeem_io::osw::{read_feature_rows, OswReadConfig};
use redeem_io::xic_parquet::XicParquetReader;

let table = read_feature_rows("sample.osw", &OswReadConfig::default())?;

let mut reader = XicParquetReader::new("run.xic");
let precursors = reader
    .filter_run_id(123)
    .filter_precursor_id([1001_u64, 1002_u64])
    .fetch()?;
# Ok::<(), anyhow::Error>(())
```

## Relationship to `redeem-topaz`

`redeem-topaz` uses `redeem-io` for:

- OSW feature loading
- score-table writeback
- XIC/XIM parquet access
- MSNumpress decoding

This separation is intentional: if additional ReDeeM models need the same OSW
or parquet infrastructure, they should depend on `redeem-io` rather than
duplicating the format logic.
