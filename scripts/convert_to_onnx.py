import torch
from pathlib import Path
import argparse


def convert_model(model_path: Path, onnx_model_path: Path, opset_version: int):
    # Restore the model with the trained weights
    mann_restored = torch.load(str(model_path))

    # Set dropout and batch normalization layers to evaluation mode before running inference
    mann_restored.eval()

    # Input to the model
    batch_size = 1
    input_size = next(mann_restored.parameters()).size()[1]
    x = torch.randn(batch_size, input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(mann_restored,  # model being run
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
    parser.add_argument('--onnx_opset_version', type=int, default=12, required=False,
                        help='The ONNX version to export the model to. At least 12 is required.')
    args = parser.parse_args()

    convert_model(model_path=args.torch_model_path, onnx_model_path=args.output, opset_version=args.onnx_opset_version)


if __name__ == "__main__":
    main()
