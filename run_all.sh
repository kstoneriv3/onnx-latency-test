#!/bin/bash
set -e

echo "CPU Specs (lscpu):"

lscpu

echo "RAM Specs (free -h):"
free -h

echo "GPU Specs (nvidia-smi):"
nvidia-smi



echo "Compiling benchmark.cpp..."
g++ -std=c++17 \
    -I/app/onnxruntime-linux-x64-gpu-1.20.1/include \
    -I/usr/local/cuda/include \
    benchmark.cpp \
    -L/app/onnxruntime-linux-x64-gpu-1.20.1/lib \
    -L/usr/local/cuda/lib64 \
    -lonnxruntime -lcudart \
    -Wl,-rpath,/app/onnxruntime-linux-x64-gpu-1.20.1/lib \
    -o benchmark

echo "Compilation successful. Running benchmark..."

echo "Creating a NN model..."
python3 create_nn_model.py

echo "Benchmarking with the NN model..."
./benchmark
echo "Done benchmarking with the NN model."

echo "Creating an XGB model..."
python3 create_xgb_model.py

echo "Benchmarking with the XGB model..."
./benchmark
echo "Done benchmarking with the XGB model."
