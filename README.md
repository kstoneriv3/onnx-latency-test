# onnx-latency-test

This is my attempt to quantify the latency of the inference of ML models using ONNX runtime from cpp. The experiment can be run using docker container or by manually setting up environment and then running the script.

## How to Run Experiment Using Docker

```bash
$ docker build -t onnx-latency-test .
$ docker run --gpus all onnx-latency-test:latest
```


## How to Run Experiment Locally

Please make sure to set up Cuda, CuDNN, Python dependencies and so on by referencing the Dockerfile. Then run:
```
$ ./run_all.sh
```

## Example Output

```
$ docker run --gpus 0 onnx-gpu-example:latest

...

CPU Specs (lscpu):

...

Model name:                           Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
NUMA node0 CPU(s):                    0-11

...

RAM Specs (free -h):
              total        used        free      shared  buff/cache   available
Mem:           31Gi        10Gi       882Mi       511Mi        19Gi        19Gi
Swap:         8.0Gi       1.0Mi       8.0Gi
GPU Specs (nvidia-smi):
Wed Feb 26 01:36:12 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1070        Off | 00000000:01:00.0  On |                  N/A |
| N/A   71C    P2              39W / 125W |    366MiB /  8192MiB |      3%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

...

Compiling benchmark.cpp...
Compilation successful. Running benchmark...
Creating a NN model...
Model exported to model.onnx
Model graph dumped to model_dump.txt
Benchmarking with the NN model...
Average inference latency on CPU: 1.42347 ms
Average inference latency on GPU: 0.22238 ms
Time to copy input from host to device: 0.006304 ms
Time to copy output from device to host: 0.001632 ms
Done benchmarking with the NN model.
Creating an XGB model...
XGBoost model converted and saved to model.onnx
Average ONNX model inference latency: 0.0095 ms
Model graph dumped to model_dump.txt
Benchmarking with the XGB model...
Average inference latency on CPU: 0.002002 ms
Average inference latency on GPU: 0.00417267 ms
Time to copy input from host to device: 0.007744 ms
Time to copy output from device to host: 0.002592 ms
Done benchmarking with the XGB model.
```
