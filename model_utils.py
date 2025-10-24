from pathlib import Path
from typing import Tuple, Sequence

import onnx
import torch
from onnxsim import simplify
from torch import nn


def export_onnx(
    network: nn.Module,
    out_onnx_path: Path,
    inputs: Tuple[torch.Tensor, ...],
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset_version: int = 12,
    export_fp16: bool = False,
):
    torch.onnx.export(
        network,
        inputs,
        out_onnx_path.as_posix(),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
    )
    print(f"ONNX fp32 model saved to {out_onnx_path}")

    # Simplify the ONNX model using onnx-simplifier
    save_sim_path = out_onnx_path.with_suffix(".sim.onnx")
    onnx_model = onnx.load(out_onnx_path)
    onnx.checker.check_model(onnx_model)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, save_sim_path)
    print(f"Simplified ONNX model saved to {save_sim_path}")

    # Convert the ONNX model to float16
    if export_fp16:
        from onnxconverter_common import float16

        model = onnx.load(save_sim_path.as_posix())
        model_fp16 = float16.convert_float_to_float16(model)
        save_fp16_path = out_onnx_path.with_suffix(".sim_fp16.onnx")
        onnx.save(model_fp16, save_fp16_path.as_posix())
        print(f"ONNX fp16 model saved to {save_fp16_path}")
    ...
