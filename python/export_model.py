#!/usr/bin/env python3
"""
Export a pretrained ResNet18 from PyTorch to ONNX.

Creates: ../models/resnet18.onnx
Usage:
  python3 export_model.py --output ../models/resnet18.onnx --opset 13
"""

import argparse
import torch
import torchvision.models as models
import os

def export_resnet18(output_path: str, opset_version: int = 13, dynamic_batch: bool = True):
    model = models.resnet18(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if dynamic_batch else None

    input_names = ['input']
    output_names = ['output']

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported ResNet18 to {output_path} (opset={opset_version}, dynamic_batch={dynamic_batch})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ResNet18 to ONNX")
    parser.add_argument("--output", "-o", default="../models/resnet18.onnx", help="Output ONNX model path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--no-dynamic-batch", action="store_true", dest="no_dynamic", help="Disable dynamic batch axis")
    args = parser.parse_args()

    export_resnet18(args.output, opset_version=args.opset, dynamic_batch=(not args.no_dynamic))
