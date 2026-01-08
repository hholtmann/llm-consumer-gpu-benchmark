# LLM Benchmark Docker Image - OPTIMIZED VERSION
# Pre-installed with vLLM, AIPerf, lm-eval, DCGM, and DCGM Exporter
# Build this on x86_64 Linux machine with NVIDIA GPU
# Optimizations:
# - Using base image instead of runtime (Python packages bundle CUDA)
# - Removing build dependencies after use
# - Cleaning up test files and caches
# - Removing redundant CUDA installation from base image

# Stage 1: Get dcgm-exporter binary from official NVIDIA image
FROM nvcr.io/nvidia/k8s/dcgm-exporter:4.4.2-4.7.0-ubuntu22.04 AS dcgm-exporter-source

# Stage 2: Main build - Using DEVEL for nvcc compiler
# Required for flashinfer JIT compilation of FP8 KV cache kernels on Blackwell GPUs
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DISABLE_REQUIRE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

# Install system dependencies and DCGM in one layer
# Build tools are needed temporarily for pip packages but removed after
# nvtop and htop for GPU/CPU monitoring
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    wget \
    ca-certificates \
    build-essential \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA DCGM for GPU monitoring
# Note: This adds ~1.68GB. If you only need dcgm-exporter, consider removing this
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g') && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends datacenter-gpu-manager && \
    rm cuda-keyring_1.0-1_all.deb && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /opt/nvidia/entrypoint.d 2>/dev/null || true

# Install Python packages in separate layers for better Docker Hub compatibility
# Split large installations to avoid 10GB+ single layers that fail to push

# Layer 1: Upgrade base pip tools (~100MB)
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    rm -rf /root/.cache/pip

# Layer 2: Install aiperf (~500MB)
RUN pip3 install --no-cache-dir aiperf==0.3.0 && \
    find /usr/local/lib/python3.10 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10 -type f -name '*.pyc' -delete 2>/dev/null || true

# Layer 3: Install pydantic (~50MB)
RUN pip3 install --no-cache-dir --force-reinstall "pydantic>=2.12.0,<3.0.0" && \
    find /usr/local/lib/python3.10 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10 -type f -name '*.pyc' -delete 2>/dev/null || true

# Layer 4: Install vLLM v0.12.0 (~8GB) - The largest package, isolated for better push reliability
RUN pip3 install --no-cache-dir vllm==0.12.0 && \
    find /usr/local/lib/python3.10 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10 -type f -name '*.pyc' -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.10 -type f -name '*.pyo' -delete 2>/dev/null || true && \
    rm -rf /root/.cache/pip

# Layer 5: Install remaining packages (~2GB)
RUN pip3 install --no-cache-dir \
        lm-eval==0.4.9.1 \
        boto3==1.41.2 \
        requests==2.32.5 \
        prometheus-client==0.23.1 && \
    # Keep build-essential and python3.10-dev - vLLM's Triton needs them at runtime for kernel compilation
    # Final cleanup
    apt-get clean && \
    find /usr/local/lib/python3.10 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10 -type f -name '*.pyc' -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.10 -type f -name '*.pyo' -delete 2>/dev/null || true && \
    # Skip test directory removal - some packages like numpy and scipy need them at runtime
    rm -rf /root/.cache/pip && \
    rm -rf /var/lib/apt/lists/*

# Copy DCGM libraries and dcgm-exporter from official NVIDIA image
COPY --from=dcgm-exporter-source /usr/local/dcgm /usr/local/dcgm
COPY --from=dcgm-exporter-source /usr/bin/dcgm-exporter /usr/local/bin/dcgm-exporter
COPY --from=dcgm-exporter-source /etc/dcgm-exporter /etc/dcgm-exporter

# Set library path for DCGM
ENV LD_LIBRARY_PATH=/usr/local/dcgm/lib64:${LD_LIBRARY_PATH}

# Create working directory
WORKDIR /workspace

# Expose ports
EXPOSE 8000
EXPOSE 9400

# Health check for vLLM
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/bin/bash"]
