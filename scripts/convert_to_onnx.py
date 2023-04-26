import torch
import torch.nn as nn
from pathlib import Path
import argparse
from mann_pytorch.utils import read_from_file
import numpy as np


def convert_model(model_path: Path, onnx_model_path: Path, normalization_folder: Path, opset_version: int):
    # Restore the model with the trained weights
    mann_restored = torch.load(str(model_path))

    # Set dropout and batch normalization layers to evaluation mode before running inference
    mann_restored.eval()
    input_size = next(mann_restored.parameters()).size()[1]

    # Here we create two layes for normalization and denormalization
    X_mean = read_from_file(str(normalization_folder / "X_mean.txt"))
    X_std = read_from_file(str(normalization_folder / "X_std.txt"))
    X_std[np.where(X_std <= 1e-4)] = 1

    Y_mean = read_from_file(str(normalization_folder / "Y_mean.txt"))
    Y_std = read_from_file(str(normalization_folder / "Y_std.txt"))
    Y_std[np.where(Y_std <= 1e-4)] = 1

    # the normalization is
    # x_norm = (x - x_mean) / x_std
    # it is possible to convert it in a linear layer by massaging the equation
    # x_norm = x / x_std - x_mean / x_std
    lin_normalization = nn.Linear(input_size, input_size)
    with torch.no_grad():
        lin_normalization.weight.copy_(torch.tensor(np.diag(np.reciprocal(X_std))))
        lin_normalization.bias.copy_(torch.tensor(-X_mean / X_std))

    # the denormalization is
    # y = y_std * y_norm + y_mean
    lin_output_denormalization = nn.Linear(Y_mean.size, Y_mean.size)
    with torch.no_grad():
        lin_output_denormalization.weight.copy_(torch.diag(torch.tensor(Y_std)))
        lin_output_denormalization.bias.copy_(torch.tensor(Y_mean))

    # The extended model contains the normalization and the denormalization
    extended_model = nn.Sequential(lin_normalization,
                                   mann_restored,
                                   lin_output_denormalization)

    # Input to the model
    batch_size = 1
    x = torch.randn(batch_size, input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(extended_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      str(onnx_model_path),  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=opset_version,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


def main():
    parser = argparse.ArgumentParser(description='Convert mann-pytorch model into a onnx model.')
    parser.add_argument('--output', '-o', type=lambda p: Path(p).absolute(), required=True,
                         help='Onnx model path.')
    parser.add_argument('--torch_model_path', '-i', type=lambda p: Path(p).absolute(),
                        default=Path(__file__).absolute().parent.parent /
                                "models" / "storage_20220909-131438" / "models" / "model_49.pth",
                        required=False,
                        help='Pytorch model location.')
    parser.add_argument('--normalization_path', '-n', type=lambda p: Path(p).absolute(),
                        default=Path(__file__).absolute().parent.parent /
                                "models" / "storage_20220909-131438" / "normalization",
                        required=False,
                        help='Folder containing the normalization files.')
    parser.add_argument('--onnx_opset_version', type=int, default=12, required=False,
                        help='The ONNX version to export the model to. At least 12 is required.')
    args = parser.parse_args()

    convert_model(model_path=args.torch_model_path,
                  onnx_model_path=args.output,
                  normalization_folder=args.normalization_path,
                  opset_version=args.onnx_opset_version)


if __name__ == "__main__":
    main()
