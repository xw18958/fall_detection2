#!/usr/bin/env python3
"""Convert a pruned TCNAttnClassifier checkpoint directly to float32 TFLite.

This avoids ONNX conversion dependencies. The script rebuilds the same inference
network in TensorFlow/Keras, copies the trained PyTorch weights exactly, checks
PyTorch vs TensorFlow numerics, converts to float32 TFLite, and checks TFLite
vs PyTorch numerics before saving the model.

Default target:
    pruned_models/lstm_fall_detector1_30hz_ft_3s_c16.pt

Default output:
    pruned_models/lstm_fall_detector1_30hz_ft_3s_c16_float32.tflite
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from prune_tcn_checkpoint import infer_architecture, load_checkpoint, unwrap_state_dict


def parse_args():
    p = argparse.ArgumentParser(description="Convert pruned TCNAttnClassifier to float32 TFLite.")
    p.add_argument(
        "--checkpoint",
        default="pruned_models/lstm_fall_detector1_30hz_ft_3s_c16.pt",
        help="Pruned PyTorch checkpoint.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output .tflite path. Defaults to <checkpoint_stem>_float32.tflite.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--verify-samples",
        type=int,
        default=8,
        help="Number of deterministic random normalized windows used for numeric verification.",
    )
    return p.parse_args()


def _np(t):
    return t.detach().cpu().numpy().astype(np.float32, copy=False)


def _conv1d_kernel(torch_weight):
    # PyTorch Conv1d: [out_channels, in_channels, kernel]
    # Keras Conv1D:    [kernel, in_channels, out_channels]
    return np.transpose(_np(torch_weight), (2, 1, 0))


def build_tf_model(tf, state, arch, win_len):
    """Build channels-last TensorFlow equivalent and load the PyTorch weights."""
    c = arch["channels"]
    input_dim = arch["input_dim"]
    num_blocks = arch["num_blocks"]
    num_classes = arch["num_classes"]

    inputs = tf.keras.Input(shape=(win_len, input_dim), batch_size=1, dtype=tf.float32, name="input")
    x = inputs

    layers = {}

    layers["stem_conv"] = tf.keras.layers.Conv1D(
        c, 5, padding="same", use_bias=False, name="stem_conv"
    )
    layers["stem_bn"] = tf.keras.layers.BatchNormalization(
        axis=-1, epsilon=1e-5, momentum=0.9, name="stem_bn"
    )
    x = layers["stem_conv"](x)
    x = layers["stem_bn"](x, training=False)
    x = tf.keras.layers.Activation(tf.nn.gelu, name="stem_gelu")(x)

    dilations = [1, 2, 4, 8][:num_blocks]
    for b, dilation in enumerate(dilations):
        residual = x
        conv1 = tf.keras.layers.Conv1D(
            c,
            5,
            padding="same",
            dilation_rate=dilation,
            use_bias=False,
            name=f"tcn_{b}_conv1",
        )
        bn1 = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=1e-5, momentum=0.9, name=f"tcn_{b}_bn1"
        )
        conv2 = tf.keras.layers.Conv1D(
            c,
            5,
            padding="same",
            dilation_rate=dilation,
            use_bias=False,
            name=f"tcn_{b}_conv2",
        )
        bn2 = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=1e-5, momentum=0.9, name=f"tcn_{b}_bn2"
        )
        layers[f"tcn_{b}_conv1"] = conv1
        layers[f"tcn_{b}_bn1"] = bn1
        layers[f"tcn_{b}_conv2"] = conv2
        layers[f"tcn_{b}_bn2"] = bn2

        y = conv1(x)
        y = bn1(y, training=False)
        y = tf.keras.layers.Activation(tf.nn.gelu, name=f"tcn_{b}_gelu1")(y)
        y = conv2(y)
        y = bn2(y, training=False)
        y = tf.keras.layers.Activation(tf.nn.gelu, name=f"tcn_{b}_gelu2")(y)
        x = tf.keras.layers.Add(name=f"tcn_{b}_add")([y, residual])

    attn_hidden = c // 2
    layers["attn_conv1"] = tf.keras.layers.Conv1D(
        attn_hidden, 1, padding="valid", use_bias=True, name="attn_conv1"
    )
    layers["attn_conv2"] = tf.keras.layers.Conv1D(
        1, 1, padding="valid", use_bias=True, name="attn_conv2"
    )
    a = layers["attn_conv1"](x)
    a = tf.keras.layers.Activation(tf.nn.gelu, name="attn_gelu")(a)
    a = layers["attn_conv2"](a)
    a = tf.keras.layers.Lambda(lambda z: tf.squeeze(z, axis=-1), name="attn_squeeze")(a)
    w = tf.keras.layers.Softmax(axis=-1, name="attn_softmax")(a)
    w = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z, axis=-1), name="attn_expand")(w)
    weighted = tf.keras.layers.Multiply(name="attn_mul")([x, w])
    h = tf.keras.layers.Lambda(lambda z: tf.reduce_sum(z, axis=1), name="attn_sum")(weighted)

    layers["head_ln"] = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-5, name="head_ln"
    )
    layers["head_dense"] = tf.keras.layers.Dense(num_classes, use_bias=True, name="logits")
    h = layers["head_ln"](h)
    outputs = layers["head_dense"](h)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"tcn_attn_c{c}")

    # Assign trained weights.
    layers["stem_conv"].set_weights([_conv1d_kernel(state["stem.0.weight"])])
    layers["stem_bn"].set_weights(
        [
            _np(state["stem.1.weight"]),
            _np(state["stem.1.bias"]),
            _np(state["stem.1.running_mean"]),
            _np(state["stem.1.running_var"]),
        ]
    )

    for b in range(num_blocks):
        layers[f"tcn_{b}_conv1"].set_weights(
            [_conv1d_kernel(state[f"tcn.{b}.net.0.weight"])]
        )
        layers[f"tcn_{b}_bn1"].set_weights(
            [
                _np(state[f"tcn.{b}.net.1.weight"]),
                _np(state[f"tcn.{b}.net.1.bias"]),
                _np(state[f"tcn.{b}.net.1.running_mean"]),
                _np(state[f"tcn.{b}.net.1.running_var"]),
            ]
        )
        layers[f"tcn_{b}_conv2"].set_weights(
            [_conv1d_kernel(state[f"tcn.{b}.net.4.weight"])]
        )
        layers[f"tcn_{b}_bn2"].set_weights(
            [
                _np(state[f"tcn.{b}.net.5.weight"]),
                _np(state[f"tcn.{b}.net.5.bias"]),
                _np(state[f"tcn.{b}.net.5.running_mean"]),
                _np(state[f"tcn.{b}.net.5.running_var"]),
            ]
        )

    layers["attn_conv1"].set_weights(
        [
            _conv1d_kernel(state["pool.score.0.weight"]),
            _np(state["pool.score.0.bias"]),
        ]
    )
    layers["attn_conv2"].set_weights(
        [
            _conv1d_kernel(state["pool.score.2.weight"]),
            _np(state["pool.score.2.bias"]),
        ]
    )
    layers["head_ln"].set_weights(
        [_np(state["head.0.weight"]), _np(state["head.0.bias"])]
    )
    layers["head_dense"].set_weights(
        [
            np.transpose(_np(state["head.2.weight"]), (1, 0)),
            _np(state["head.2.bias"]),
        ]
    )

    return model


def build_torch_model(checkpoint, state, arch):
    from prune_tcn_checkpoint import TCNAttnClassifier

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    dropout = float(config.get("dropout", 0.2))
    model = TCNAttnClassifier(
        input_dim=arch["input_dim"],
        channels=arch["channels"],
        num_blocks=arch["num_blocks"],
        dropout=dropout,
        num_classes=arch["num_classes"],
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def run_tflite(tf, model_bytes, samples):
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError(
            f"Expected one input and one output, got {len(input_details)} inputs and {len(output_details)} outputs."
        )

    inp = input_details[0]
    out = output_details[0]
    outputs = []
    for sample in samples:
        interpreter.set_tensor(inp["index"], sample[None, ...].astype(np.float32, copy=False))
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(out["index"])[0].copy())

    op_names = []
    if hasattr(interpreter, "_get_ops_details"):
        for item in interpreter._get_ops_details():
            name = item.get("op_name")
            if name and name not in op_names:
                op_names.append(name)

    return (
        np.asarray(outputs, dtype=np.float32),
        list(inp["shape"]),
        list(out["shape"]),
        str(inp["dtype"]),
        str(out["dtype"]),
        op_names,
    )


def compare(name, reference, candidate, rtol=2e-4, atol=2e-5):
    max_abs = float(np.max(np.abs(reference - candidate)))
    mean_abs = float(np.mean(np.abs(reference - candidate)))
    ref_cls = np.argmax(reference, axis=1)
    cand_cls = np.argmax(candidate, axis=1)
    agreement = float(np.mean(ref_cls == cand_cls))
    close = bool(np.allclose(reference, candidate, rtol=rtol, atol=atol))
    print(
        f"{name}: max_abs_diff={max_abs:.8g} mean_abs_diff={mean_abs:.8g} "
        f"class_agreement={agreement:.4f} allclose={close}"
    )
    return close and agreement == 1.0


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorFlow is required for this converter.") from exc

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.with_name(checkpoint_path.stem + "_float32.tflite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    state = unwrap_state_dict(checkpoint)
    arch = infer_architecture(state)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    win_len = int(config.get("win_len", 60))

    torch_model = build_torch_model(checkpoint, state, arch)
    tf_model = build_tf_model(tf, state, arch, win_len)

    rng = np.random.RandomState(args.seed)
    samples = rng.normal(
        0.0,
        1.0,
        size=(args.verify_samples, win_len, arch["input_dim"]),
    ).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(samples)).cpu().numpy().astype(np.float32)
    tf_out = np.asarray(tf_model(samples, training=False).numpy(), dtype=np.float32)

    if not compare("PyTorch vs TensorFlow", torch_out, tf_out):
        raise RuntimeError("PyTorch -> TensorFlow verification failed; refusing TFLite conversion.")

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = []
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    (
        tflite_out,
        input_shape,
        output_shape,
        input_dtype,
        output_dtype,
        op_names,
    ) = run_tflite(tf, tflite_model, samples)

    expected_input_shape = [1, win_len, arch["input_dim"]]
    expected_output_shape = [1, arch["num_classes"]]
    if input_shape != expected_input_shape:
        raise RuntimeError(f"TFLite input shape {input_shape} != expected {expected_input_shape}")
    if output_shape != expected_output_shape:
        raise RuntimeError(f"TFLite output shape {output_shape} != expected {expected_output_shape}")
    if "float32" not in input_dtype or "float32" not in output_dtype:
        raise RuntimeError(
            f"Expected float32 I/O, got input={input_dtype}, output={output_dtype}"
        )

    if not compare("PyTorch vs TFLite", torch_out, tflite_out, rtol=5e-4, atol=5e-5):
        raise RuntimeError("PyTorch -> TFLite verification failed; refusing to save model.")

    output_path.write_bytes(tflite_model)

    print(f"Checkpoint: {checkpoint_path}")
    print(
        "Architecture: "
        f"input_dim={arch['input_dim']} channels={arch['channels']} "
        f"blocks={arch['num_blocks']} classes={arch['num_classes']} win_len={win_len}"
    )
    print(f"TFLite: {output_path}")
    print(f"TFLite size: {len(tflite_model):,} bytes ({len(tflite_model) / 1024:.1f} KiB)")
    print(f"Input shape:  {input_shape} dtype={input_dtype}")
    print(f"Output shape: {output_shape} dtype={output_dtype}")
    if op_names:
        print("TFLite operators: " + ", ".join(op_names))
    print("Float32 TFLite conversion and verification passed.")


if __name__ == "__main__":
    main()
