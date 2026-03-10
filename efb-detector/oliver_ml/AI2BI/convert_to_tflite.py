import torch
import torch.nn as nn
from model import GNNModel
import onnx
import tensorflow as tf
from onnx2tf import convert

model = GNNModel()
model: nn.Module

model.load_state_dict(torch.load("disease_classifier.pt"))

model.eval()

example_input = torch.randn(1, 3, 48, 48)  # Example input
onnx_filename = "disease_classifier.onnx"
torch.onnx.export(model, example_input, onnx_filename, input_names=["input"], output_names=["output"], opset_version=11)

convert(onnx_filename, output_folder_path="tf_classifier")

