# Use Ubuntu as base image for better compatibility with both Python and Rust
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RUST_BACKTRACE=1

# install add-apt-repository command
RUN apt-get update && apt-get install -y software-properties-common
# add ppa for yices
RUN add-apt-repository ppa:sri-csl/formal-methods

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python dependencies
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    # Rust dependencies
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    # Git (still needed for some build processes)
    git \
    vim \
    unzip \
    # System utilities
    sudo \
    # Azure CLI for authentication (optional)
    ca-certificates \
    gnupg \
    lsb-release \
    yices2 \
    z3 \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -s /bin/bash appuser
RUN adduser appuser sudo
# Allow sudo without password for convenience (optional - remove for production)
RUN echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER appuser

# Install Rust
WORKDIR /home/appuser
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/appuser/.cargo/bin:${PATH}" 
RUN cargo install verusfmt

# Clone and build Verus
WORKDIR /home/appuser
RUN git clone https://github.com/verus-lang/verus.git
WORKDIR /home/appuser/verus
RUN git checkout 33269ac6a0ea33a08109eefe5016c1fdd0ce9fbd

# Build Verus (following the proper Verus build process)
WORKDIR /home/appuser/verus/source
RUN bash -c "source ../tools/activate && ./tools/get-z3.sh && vargo build --release"

# install haskell
ENV BOOTSTRAP_HASKELL_NONINTERACTIVE=1
RUN curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh

# install mir json
RUN rustup toolchain install nightly-2025-09-14 --force --component rustc-dev,rust-src
WORKDIR /home/appuser
RUN git clone https://github.com/GaloisInc/mir-json.git
WORKDIR /home/appuser/mir-json
RUN git checkout 2e76735a62a7667069b4bfda6121837a96832cbb
RUN cargo +nightly-2025-09-14 install --path . --locked
RUN mir-json-translate-libs
ENV CRUX_RUST_LIBRARY_PATH=/home/appuser/mir-json/rlibs
ENV SAW_RUST_LIBRARY_PATH=/home/appuser/mir-json/rlibs

# Copy application source code
WORKDIR /home/appuser
COPY --chown=appuser:appuser . /home/appuser/verus-proof-synthesis/

WORKDIR /home/appuser/verus-proof-synthesis
# Create virtual environment and install Python dependencies
RUN python3.10 -m venv /home/appuser/venv
ENV PATH="/home/appuser/venv/bin:/home/appuser/verus/source/target-verus/release:$PATH"
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# You can override this when running the container
CMD ["/bin/bash"]

# Add labels for metadata
LABEL maintainer="verus-proof-synthesis"
LABEL description="Docker image for Verus proof synthesis project"
LABEL version="0.1.0" 
