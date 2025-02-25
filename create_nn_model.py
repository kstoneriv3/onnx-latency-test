import torch
import torch.nn as nn
import numpy as np
import onnx
import onnx.helper

# Define a feedforward network with 3 hidden layers, each with 1000 units.
class FeedForwardNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=10, output_size=10, num_hidden_layers=3):
        super(FeedForwardNN, self).__init__()
        layers = []
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def export_model_to_onnx(model, filename="model.onnx"):
    # Set the model to evaluation mode.
    model.eval()
    # Create a dummy input tensor with the appropriate shape.
    dummy_input = torch.randn(1, 100)  # Batch size of 1, input dimension 100.
    # Export the model to ONNX.
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True
    )
    print(f"Model exported to {filename}")

def dump_onnx_model(onnx_filename="model.onnx", dump_filename="model_dump.txt"):
    # Load the ONNX model.
    model = onnx.load(onnx_filename)
    # Get a human-readable string of the model's graph.
    model_graph = onnx.helper.printable_graph(model.graph)
    # Write the graph to a text file.
    with open(dump_filename, "w") as f:
        f.write(model_graph)
    print(f"Model graph dumped to {dump_filename}")

def main():
    # Create the model.
    model = FeedForwardNN(input_size=100, hidden_size=2000, output_size=10, num_hidden_layers=3)
    
    # (Optional) You could run a forward pass here if desired:
    # dummy_input = torch.randn(1, 100)
    # output = model(dummy_input)
    # print("Model output:", output)
    
    # Export the model to an ONNX file.
    export_model_to_onnx(model, filename="model.onnx")
    # Dump the ONNX model graph to a text file.
    dump_onnx_model(onnx_filename="model.onnx", dump_filename="model_dump.txt")

if __name__ == "__main__":
    main()

