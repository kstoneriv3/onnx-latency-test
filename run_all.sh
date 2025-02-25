#!/bin/bash
set -e

python3 create_nn_model.py

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
./benchmark

