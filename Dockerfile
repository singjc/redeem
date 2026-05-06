# Stage 1: Builder
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libssl-dev \
    pkg-config \
    clang \
    libstdc++-12-dev \
    cmake \
    git && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CUDA_COMPUTE_CAP=70

# Set working directory
WORKDIR /app

# Copy source files
COPY Cargo.toml ./
COPY Cargo.lock ./
COPY redeem-openms-ffi ./redeem-openms-ffi
COPY redeem-classifiers ./redeem-classifiers
COPY redeem-cli ./redeem-cli
COPY redeem-properties ./redeem-properties
COPY redeem-properties-py ./redeem-properties-py

# Build release binary with CUDA
RUN cargo build --release --bin redeem --features cuda

# Stage 2: Runtime
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS runtime

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libstdc++6 \
    libgomp1 \
    && update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/redeem /app/redeem

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Add binary directory to PATH
ENV PATH="/app:${PATH}"

ENTRYPOINT ["redeem"]
