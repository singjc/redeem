ARG CUDA_VERSION=12.2.2

# -----------------------------------------------------------------------------
# Rust / CUDA builder
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG RUST_TOOLCHAIN=stable
ARG CUDA_COMPUTE_CAP=80
ARG CARGO_BUILD_JOBS=1
ARG CARGO_RELEASE_LTO=off
ARG CARGO_RELEASE_CODEGEN_UNITS=8
ARG CARGO_RELEASE_DEBUG=0
ARG CMAKE_BUILD_PARALLEL_LEVEL=2

RUN apt-get -o Acquire::Retries=5 update && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        git \
        libssl-dev \
        libstdc++-12-dev \
        pkg-config && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf \
        --retry 6 --retry-all-errors --retry-delay 5 \
        --connect-timeout 30 --max-time 300 \
        https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"

ENV PATH=/root/.cargo/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}
ENV CARGO_INCREMENTAL=0
# Override only the container's Cargo release profile. The workspace normally
# uses fat LTO + one codegen unit + full debug info, which causes very high
# peak LLVM/linker RAM usage. Cargo officially supports these profile settings
# through CARGO_PROFILE_<name>_* environment variables.
ENV CARGO_PROFILE_RELEASE_LTO=${CARGO_RELEASE_LTO}
ENV CARGO_PROFILE_RELEASE_CODEGEN_UNITS=${CARGO_RELEASE_CODEGEN_UNITS}
ENV CARGO_PROFILE_RELEASE_DEBUG=${CARGO_RELEASE_DEBUG}

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY redeem-openms-ffi ./redeem-openms-ffi
COPY redeem-classifiers ./redeem-classifiers
COPY redeem-cli ./redeem-cli
COPY redeem-properties ./redeem-properties
COPY redeem-properties-py ./redeem-properties-py
COPY scripts ./scripts

# Build the production CLI and the current foundation research binaries.  Keep
# Cargo parallelism deliberately conservative for workstation image builds;
# this controls peak host RAM and linker pressure without changing the emitted
# model architecture or runtime numerics.
RUN set -eux; \
    echo "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}"; \
    cargo build --release --locked --bin redeem --features cuda; \
    FOUNDATION_EXAMPLES="\
foundation_initialize_model \
foundation_prepare_benchmark \
foundation_prepare_corpus_benchmark \
foundation_train_corpus \
foundation_train_diffusion \
foundation_train_causal \
foundation_train_reverse_causal \
foundation_train_unified \
foundation_generate_unified \
foundation_evaluate_checkpoint \
foundation_embedding \
foundation_benchmark_unified_forward \
foundation_train_trainable_interaction_reranker \
foundation_train_compatibility_pretraining \
foundation_train_compatibility_pretraining_v0171 \
foundation_train_compatibility_pretraining_v0180 \
foundation_export_alphapeptdeep_comparison"; \
    EXAMPLE_ARGS=""; \
    for example_name in ${FOUNDATION_EXAMPLES}; do \
        test -f "redeem-properties/examples/${example_name}.rs"; \
        EXAMPLE_ARGS="${EXAMPLE_ARGS} --example ${example_name}"; \
    done; \
    cargo build --release --locked -p redeem-properties --features cuda ${EXAMPLE_ARGS}; \
    mkdir -p /opt/redeem/bin; \
    cp target/release/redeem /opt/redeem/bin/redeem; \
    for example_name in ${FOUNDATION_EXAMPLES}; do \
        cp "target/release/examples/${example_name}" "/opt/redeem/bin/${example_name}"; \
    done

# -----------------------------------------------------------------------------
# Runtime base. Apt may run while the Rust builder is active, but the expensive
# Python/AlphaPeptDeep installation is deliberately placed in the final stage
# *after* COPY --from=builder, forcing BuildKit to wait for Rust compilation.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime-base

RUN apt-get -o Acquire::Retries=5 update && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        bash \
        ca-certificates \
        libgomp1 \
        libssl3 \
        libstdc++6 \
        python3 \
        python3-pip \
        python3-venv \
        time && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV VIRTUAL_ENV=/opt/redeem/venv
ENV PATH=/opt/redeem/venv/bin:/opt/redeem/bin:/usr/local/bin:${PATH}
ENV REDEEM_PEPTDEEP_HOME=/opt/redeem/peptdeep
ENV MPLCONFIGDIR=/tmp/redeem-matplotlib
ENV NUMBA_CACHE_DIR=/tmp/redeem-numba
ENV XDG_CACHE_HOME=/tmp/redeem-cache

FROM runtime-base AS runtime

ARG PEPTDEEP_VERSION=1.5.1
ARG PEPTDEEP_TORCH_VERSION=2.5.1
ARG PEPTDEEP_TORCH_VARIANT=cpu
ARG INSTALL_ALPHAPEPTDEEP=1
ARG PRELOAD_ALPHAPEPTDEEP_MODELS=1
ARG PYTHON_BUILD_THREADS=1

# IMPORTANT: this dependency edge serializes the two high-pressure phases.
# Python / torch / AlphaPeptDeep installation cannot begin until the complete
# CUDA/Rust builder has finished and its binaries have been copied.
COPY --from=builder /opt/redeem/bin/ /opt/redeem/bin/

WORKDIR /work

COPY scripts/compare_foundation_alphapeptdeep.py /opt/redeem/scripts/compare_foundation_alphapeptdeep.py
COPY redeem-properties/nbs/redeem_foundation_validation_report.ipynb /opt/redeem/notebooks/redeem_foundation_validation_report.ipynb
COPY scripts/redeem-foundation /usr/local/bin/redeem-foundation
COPY scripts/redeem-container-info /usr/local/bin/redeem-container-info

ENV MAX_JOBS=${PYTHON_BUILD_THREADS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${PYTHON_BUILD_THREADS}
ENV OMP_NUM_THREADS=${PYTHON_BUILD_THREADS}
ENV OPENBLAS_NUM_THREADS=${PYTHON_BUILD_THREADS}
ENV MKL_NUM_THREADS=${PYTHON_BUILD_THREADS}

# AlphaPeptDeep comparison is intentionally CPU-only. ReDeeM/Candle remains CUDA-enabled.
# Installing the official CPU torch wheel first prevents pip from pulling a second
# CUDA runtime (currently cu124 for torch 2.5.1), which conflicts with the image's
# CUDA 12.2 runtime and is unnecessary for the small held-out comparison.
# Keep each expensive Python step in its own Docker layer so successful downloads
# remain cached even if a later model-preload step fails.
RUN python3 -m venv "${VIRTUAL_ENV}" && \
    "${VIRTUAL_ENV}/bin/python" -m pip install --no-cache-dir --retries 10 --timeout 120 --upgrade pip setuptools wheel

RUN if [ "${INSTALL_ALPHAPEPTDEEP}" = "1" ]; then \
        test "${PEPTDEEP_TORCH_VARIANT}" = "cpu"; \
        "${VIRTUAL_ENV}/bin/python" -m pip install --no-cache-dir --retries 10 --timeout 120 \
            --index-url https://download.pytorch.org/whl/cpu \
            "torch==${PEPTDEEP_TORCH_VERSION}"; \
    fi

RUN if [ "${INSTALL_ALPHAPEPTDEEP}" = "1" ]; then \
        "${VIRTUAL_ENV}/bin/python" -m pip install --no-cache-dir --retries 10 --timeout 120 \
            "peptdeep[stable]==${PEPTDEEP_VERSION}" \
            "nbconvert>=7,<8" \
            "matplotlib>=3.7"; \
        "${VIRTUAL_ENV}/bin/python" -m pip check; \
        "${VIRTUAL_ENV}/bin/python" -c 'import torch; print("AlphaPeptDeep torch:", torch.__version__, "cuda_available=", torch.cuda.is_available()); assert "+cpu" in torch.__version__; assert not torch.cuda.is_available()'; \
    fi

RUN if [ "${INSTALL_ALPHAPEPTDEEP}" = "1" ] && [ "${PRELOAD_ALPHAPEPTDEEP_MODELS}" = "1" ]; then \
        mkdir -p /opt/redeem; \
        success=0; \
        for attempt in 1 2 3 4 5; do \
            if HOME=/opt/redeem "${VIRTUAL_ENV}/bin/peptdeep" install-models --overwrite True; then \
                success=1; \
                break; \
            fi; \
            echo "AlphaPeptDeep model download attempt ${attempt} failed; retrying..." >&2; \
            sleep $((attempt * 10)); \
        done; \
        test "${success}" = "1"; \
    fi

# Foundation corpus ingestion deliberately streams compressed inputs through
# system decompressors (zstd -dc / gzip -dc). Install these in a late runtime
# layer so adding them does not invalidate the expensive Rust/CUDA or Python/APD
# build layers that are already cached.
RUN apt-get -o Acquire::Retries=5 update && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        gzip \
        zstd && \
    command -v gzip && \
    command -v zstd && \
    rm -rf /var/lib/apt/lists/*

RUN chmod 0755 \
        /usr/local/bin/redeem-foundation \
        /usr/local/bin/redeem-container-info \
        /opt/redeem/scripts/compare_foundation_alphapeptdeep.py && \
    /usr/local/bin/redeem-container-info --build-check

CMD ["redeem-foundation", "help"]
