# Use devel image instead of runtime for compilation
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set CUDA environment variables - CRITICAL for diffvg compilation
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    cmake \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libx11-6 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch first (critical for diffvg compilation)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN pip install ffmpeg

# Copy and install requirements (excluding torch to avoid conflicts)
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# Verify CUDA installation
# RUN nvcc --version
# RUN python -c "import torch; print('PyTorch installed with CUDA version:', torch.version.cuda)"

ENV DIFFVG_CUDA=1

# COPY ./diffvg ./diffvg
# RUN cd diffvg && \
#     rm -rf build/ && \
#     find . -name "CMakeCache.txt" -delete && \
#     python setup.py install && \
#     cd .. 
# Build diffvg with comprehensive CUDA environment
# RUN git clone https://github.com/BachiLi/diffvg && \
#     cd diffvg && \
#     git submodule update --init --recursive && \
#     export CC=gcc && \
#     export CXX=g++ && \
#     export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/gcc' && \
#     export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0" && \
#     export MAX_JOBS=4 && \
#     WITH_CUDA=1 \
#     CUDA_HOME=/usr/local/cuda \
#     CUDA_ROOT=/usr/local/cuda \
#     FORCE_CUDA=1 \
#     DIFFVG_CUDA=1 \
#     # python setup.py build_ext --inplace && \
#     python setup.py install && \
#     cd .. 

ENV PYTHONPATH=/app:/root

COPY . .