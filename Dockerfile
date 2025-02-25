# Use an NVIDIA CUDA devel image so that CUDA headers are available
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04


# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install required tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages
RUN python3 -m pip install --upgrade pip && \
    pip install numpy torch onnx onnxmltools onnxruntime xgboost

# Set working directory
WORKDIR /app

# Download the ONNX Runtime GPU tarball (version 1.20.1)
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-1.20.1.tgz && \
    rm onnxruntime-linux-x64-gpu-1.20.1.tgz

# Copy your source files into the container
COPY benchmark.cpp .
COPY create_nn_model.py .
COPY create_xgb_model.py .
COPY run_all.sh .

# Set LD_LIBRARY_PATH so that the dynamic linker finds the ONNX Runtime shared libraries and any others you need.
# (You can combine paths as needed.)
ENV LD_LIBRARY_PATH=/app/onnxruntime-linux-x64-gpu-1.20.1/lib:$LD_LIBRARY_PATH

# Compile the benchmark binary.
# We now add the CUDA include directory (-I/usr/local/cuda/include) so that the compiler can find cuda_runtime_api.h.
# Compile the benchmark binary using the provided compile command

# Run the create_nn_model.py script with benchmark as argument
CMD ["./run_all.sh"]

