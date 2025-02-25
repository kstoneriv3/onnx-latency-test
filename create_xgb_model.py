import numpy as np
import xgboost as xgb
import time
import onnx
import onnxruntime as ort
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

def generate_data(n_samples=1000, n_features=100):
    """Generate synthetic regression data."""
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.sum(X, axis=1)  # For example, regression target is the sum of features.
    return X, y

def train_xgboost_model(X, y):
    """Train an XGBoost regression model."""
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": 0.1,
        # Use 'auto' or 'gpu_hist' (if GPU is available and properly configured)
        "tree_method": "auto"
    }
    num_round = 50
    bst = xgb.train(params, dtrain, num_round)
    return bst

def convert_model_to_onnx(bst, n_features, onnx_filename="model.onnx"):
    """Convert the trained XGBoost model to ONNX format and save it."""
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(bst, initial_types=initial_types)
    onnx.save_model(onnx_model, onnx_filename)
    print(f"XGBoost model converted and saved to {onnx_filename}")

def benchmark_onnx_model(model_path, iterations=1000):
    """Benchmark ONNX model inference latency using ONNX Runtime."""
    # Create an ONNX Runtime session.
    session = ort.InferenceSession(model_path)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    # Replace dynamic dimensions (None or <= 0) with 1.
    input_shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_shape]
    
    # Create dummy input.
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warm-up runs.
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = session.run(None, {input_name: dummy_input})
    end = time.perf_counter()
    avg_latency_ms = ((end - start) / iterations) * 1000
    print(f"Average ONNX model inference latency: {avg_latency_ms:.4f} ms")
    return session

def dump_onnx_model(onnx_filename="model.onnx", dump_filename="model_dump.txt"):
    """Dump a human‑readable version of the ONNX model graph to a text file."""
    model = onnx.load(onnx_filename)
    model_graph = onnx.helper.printable_graph(model.graph)
    with open(dump_filename, "w") as f:
        f.write(model_graph)
    print(f"Model graph dumped to {dump_filename}")

def main():
    # Generate synthetic data.
    X, y = generate_data(n_samples=1000, n_features=100)
    
    # Train the XGBoost model.
    bst = train_xgboost_model(X, y)
    
    # Convert the XGBoost model to ONNX.
    convert_model_to_onnx(bst, n_features=100, onnx_filename="model.onnx")
    
    # Benchmark the exported ONNX model.
    benchmark_onnx_model("model.onnx", iterations=1000)
    
    # Dump the ONNX model graph to a text file.
    dump_onnx_model("model.onnx", dump_filename="model_dump.txt")

if __name__ == "__main__":
    main()

