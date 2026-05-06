# `redeem-openms-ffi`

Thin C ABI bridge for calling `redeem-properties` from OpenMS.

This crate is intended to be consumed as a prebuilt `staticlib`, not as a direct
dependency inside the OpenMS build. OpenMS can either download the matching
prebuilt bundle automatically or use a manual archive path through the
`REDEEM_FFI_LIBRARY` CMake cache variable.

## OpenMS integration scope

The OpenMS build uses the minimal `redeem-properties` feature set:

- RT inference via `rt_cnn_tf`
- CCS inference via `ccs_cnn_tf`
- MS2 intensity inference via `ms2_bert`
- optional fine-tuning from local TSV data via the same three model families
- local-file model loading only

That means the prebuilt OpenMS FFI bundle intentionally excludes the
pretrained-model download helpers and the legacy RT/CCS LSTM variants.

The C ABI now exposes two main workflows:

- batch prediction for RT, CCS, and MS2 intensities
- optional fine-tuning from a local TSV before prediction, with per-model
  enable flags and optional `.safetensors` output checkpoints

## Prebuilt bundle layout

OpenMS expects a prebuilt static library:

- Linux/macOS: `libredeem_openms_ffi.a`
- Windows (MSVC): `redeem_openms_ffi.lib`

When possible, keep a metadata file next to the archive:

- `redeem-openms-ffi.native-static-libs.txt`

That file should contain the raw output of:

```bash
cargo rustc -p redeem-openms-ffi --release -- --print native-static-libs
```

OpenMS can use it to refine the native link libraries required by the Rust
archive on each platform.

## Manual bundle creation

On Unix-like systems, you can build and stage a local bundle with:

```bash
./redeem-openms-ffi/package-prebuilt.sh
```

Or for a specific target:

```bash
./redeem-openms-ffi/package-prebuilt.sh x86_64-unknown-linux-gnu /tmp/redeem-openms-ffi-bundle
```

The script creates a directory containing:

- `lib/` with the static archive
- `lib/redeem-openms-ffi.native-static-libs.txt`
- `README.md`
- `LICENSE`

## Using the bundle from OpenMS

By default, OpenMS can auto-download the matching bundle for the current
platform:

```bash
cmake -S /path/to/OpenMS -B build \
  -DWITH_REDEEM=ON
```

To pin a specific release instead of using the latest published bundle:

```bash
cmake -S /path/to/OpenMS -B build \
  -DWITH_REDEEM=ON \
  -DREDEEM_FFI_VERSION=v0.1.0
```

For local development or manual packaging, you can still point OpenMS at the
extracted archive directly:

```bash
cmake -S /path/to/OpenMS -B build \
  -DWITH_REDEEM=ON \
  -DREDEEM_FFI_LIBRARY=/path/to/redeem-openms-ffi-bundle/lib/libredeem_openms_ffi.a
```

## Testing the release workflow manually

The `Release OpenMS FFI` GitHub Actions workflow can be run manually without
creating a new GitHub release.

- Leave `release_tag` empty to build workflow artifacts only.
- Set `release_tag` to an existing tag such as `v0.1.0` to upload the rebuilt
  bundles to that release.
- Set `release_tag` to `latest` to upload to the latest published release.

The optional `ref` input controls which commit/tag/branch is built. This lets
you test the workflow against a branch while still attaching assets to an
existing release tag if needed.
