#!/usr/bin/env python3
"""Export a pruned TCNAttnClassifier checkpoint to fixed-shape ONNX.

Default target is the C16 checkpoint produced by prune_tcn_checkpoint.py.
The exported graph uses input shape [1, 60, 6] and output shape [1, 2].

Example:
    python3 export_pruned_tcn.py \
        --checkpoint pruned_models/lstm_fall_detector1_30hz_ft_3s_c16.pt
"""

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch

from prune_tcn_checkpoint import (
    TCNAttnClassifier,
    infer_architecture,
    load_checkpoint,
    unwrap_state_dict,
)


def parse_args():
    p = argparse.ArgumentParser(description="Export pruned TCNAttnClassifier checkpoint to ONNX.")
    p.add_argument(
        "--checkpoint",
        default="pruned_models/lstm_fall_detector1_30hz_ft_3s_c16.pt",
        help="Pruned PyTorch checkpoint to export.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output ONNX path. Defaults to the checkpoint path with .onnx suffix.",
    )
    p.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def shape_from_value_info(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output) if args.output else checkpoint_path.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    state = unwrap_state_dict(checkpoint)
    arch = infer_architecture(state)

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    dropout = float(config.get("dropout", 0.2))
    win_len = int(config.get("win_len", 60))

    model = TCNAttnClassifier(
        input_dim=arch["input_dim"],
        channels=arch["channels"],
        num_blocks=arch["num_blocks"],
        dropout=dropout,
        num_classes=arch["num_classes"],
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    expected_input_shape = (1, win_len, arch["input_dim"])
    expected_output_shape = (1, arch["num_classes"])

    example = torch.randn(*expected_input_shape, dtype=torch.float32)
    with torch.no_grad():
        torch_output = model(example).cpu().numpy()

    if tuple(torch_output.shape) != expected_output_shape:
        raise RuntimeError(
            f"Unexpected PyTorch output shape {tuple(torch_output.shape)}, "
            f"expected {expected_output_shape}."
        )

    export_kwargs = dict(
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    # Force the classic exporter where supported. It is stable for this simple
    # fixed-shape graph and avoids newer exporter-specific dependencies.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    try:
        torch.onnx.export(model, example, str(output_path), **export_kwargs)
    except ModuleNotFoundError as exc:
        if exc.name == "onnx":
            raise RuntimeError(
                "ONNX export requires the Python package 'onnx', but it is not installed in this "
                "Python environment. Do not install anything yet; report this error so the existing "
                "environment can be checked first."
            ) from exc
        raise

    try:
        import onnx
        from onnx.reference import ReferenceEvaluator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The ONNX file was exported, but verification requires the Python package 'onnx'."
        ) from exc

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    input_shape = shape_from_value_info(onnx_model.graph.input[0])
    output_shape = shape_from_value_info(onnx_model.graph.output[0])
    if input_shape != list(expected_input_shape):
        raise RuntimeError(f"ONNX input shape is {input_shape}, expected {list(expected_input_shape)}")
    if output_shape != list(expected_output_shape):
        raise RuntimeError(f"ONNX output shape is {output_shape}, expected {list(expected_output_shape)}")

    evaluator = ReferenceEvaluator(onnx_model)
    onnx_output = evaluator.run(None, {"input": example.cpu().numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    mean_abs_diff = float(np.mean(np.abs(torch_output - onnx_output)))
    same_class = int(np.argmax(torch_output, axis=1)[0]) == int(np.argmax(onnx_output, axis=1)[0])
    close = bool(np.allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-5))

    size_bytes = output_path.stat().st_size
    print(f"Checkpoint: {checkpoint_path}")
    print(
        "Architecture: "
        f"input_dim={arch['input_dim']} channels={arch['channels']} "
        f"blocks={arch['num_blocks']} classes={arch['num_classes']} win_len={win_len}"
    )
    print(f"ONNX: {output_path}")
    print(f"ONNX size: {size_bytes:,} bytes ({size_bytes / 1024:.1f} KiB)")
    print(f"Input shape:  {input_shape}")
    print(f"Output shape: {output_shape}")
    print(f"PyTorch logits: {torch_output.tolist()}")
    print(f"ONNX logits:    {np.asarray(onnx_output).tolist()}")
    print(f"max_abs_diff={max_abs_diff:.8g}")
    print(f"mean_abs_diff={mean_abs_diff:.8g}")
    print(f"same_class={same_class}")
    print(f"allclose(rtol=1e-4, atol=1e-5)={close}")

    if not same_class or not close:
        raise RuntimeError("PyTorch and ONNX verification failed; do not continue to TFLite conversion.")

    print("ONNX export and verification passed.")


if __name__ == "__main__":
    main()
