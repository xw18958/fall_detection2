#!/usr/bin/env python3
"""Structured channel pruning for the trained TCNAttnClassifier checkpoint.

This does not retrain or redesign the network. It keeps the same topology and
slices trained hidden channels consistently through the stem, TCN blocks,
attention pool, and classifier head.

Example:
    python3 prune_tcn_checkpoint.py \
        --checkpoint lstm_fall_detector1_30hz_ft_3s.pt \
        --targets 64 32 16
"""

import argparse
import copy
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, dilation=1, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class AttnPool1d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(ch, ch // 2, 1),
            nn.GELU(),
            nn.Conv1d(ch // 2, 1, 1),
        )

    def forward(self, x):
        a = self.score(x).squeeze(1)
        w = torch.softmax(a, dim=-1).unsqueeze(1)
        return (x * w).sum(dim=-1)


class TCNAttnClassifier(nn.Module):
    def __init__(self, input_dim=6, channels=128, num_blocks=4, dropout=0.2, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        dilations = [1, 2, 4, 8][:num_blocks]
        self.tcn = nn.Sequential(
            *[TCNBlock(channels, channels, k=5, dilation=d, dropout=dropout) for d in dilations]
        )
        self.pool = AttnPool1d(channels)
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.tcn(x)
        x = self.pool(x)
        return self.head(x)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        state = checkpoint
    else:
        raise ValueError("Checkpoint does not contain a recognizable model_state_dict.")

    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def infer_architecture(state):
    stem_w = state["stem.0.weight"]
    head_w = state["head.2.weight"]
    block_ids = sorted({
        int(m.group(1))
        for key in state
        for m in [re.match(r"^tcn\.(\d+)\.net\.0\.weight$", key)]
        if m is not None
    })
    if not block_ids:
        raise ValueError("Could not infer TCN blocks from the checkpoint.")
    if block_ids != list(range(max(block_ids) + 1)):
        raise ValueError(f"Unexpected TCN block numbering: {block_ids}")
    return {
        "input_dim": int(stem_w.shape[1]),
        "channels": int(stem_w.shape[0]),
        "num_blocks": len(block_ids),
        "num_classes": int(head_w.shape[0]),
    }


def normalize_importance(v):
    v = v.detach().float().cpu()
    return v / v.mean().abs().clamp_min(1e-12)


def main_channel_importance(state, channels, num_blocks):
    score = torch.zeros(channels, dtype=torch.float32)
    terms = 0

    def add(v):
        nonlocal score, terms
        if v.numel() != channels:
            raise ValueError(f"Importance vector has {v.numel()} elements, expected {channels}.")
        score += normalize_importance(v)
        terms += 1

    add(state["stem.0.weight"].abs().mean(dim=(1, 2)))
    add(state["stem.1.weight"].abs())

    for b in range(num_blocks):
        for conv_idx in (0, 4):
            w = state[f"tcn.{b}.net.{conv_idx}.weight"].abs()
            add(w.mean(dim=(1, 2)))
            add(w.mean(dim=(0, 2)))
        for bn_idx in (1, 5):
            add(state[f"tcn.{b}.net.{bn_idx}.weight"].abs())

    add(state["pool.score.0.weight"].abs().mean(dim=(0, 2)))
    add(state["head.0.weight"].abs())
    add(state["head.2.weight"].abs().mean(dim=0))
    return score / max(1, terms)


def attention_hidden_importance(state):
    a = state["pool.score.0.weight"].abs().mean(dim=(1, 2))
    b = state["pool.score.2.weight"].abs().mean(dim=(0, 2))
    return 0.5 * (normalize_importance(a) + normalize_importance(b))


def top_indices(score, n):
    selected = torch.topk(score, k=n, largest=True, sorted=False).indices
    return torch.sort(selected).values.long()


def slice_1d(t, idx):
    return t.index_select(0, idx.to(t.device)).clone()


def prune_state_dict(source, main_idx, attn_idx, num_blocks):
    out = {}
    for key, value in source.items():
        if key == "stem.0.weight":
            out[key] = value.index_select(0, main_idx).clone()
        elif key in {"stem.1.weight", "stem.1.bias", "stem.1.running_mean", "stem.1.running_var"}:
            out[key] = slice_1d(value, main_idx)
        elif key == "stem.1.num_batches_tracked":
            out[key] = value.clone()
        elif re.match(r"^tcn\.\d+\.net\.(0|4)\.weight$", key):
            out[key] = value.index_select(0, main_idx).index_select(1, main_idx).clone()
        elif re.match(r"^tcn\.\d+\.net\.(1|5)\.(weight|bias|running_mean|running_var)$", key):
            out[key] = slice_1d(value, main_idx)
        elif re.match(r"^tcn\.\d+\.net\.(1|5)\.num_batches_tracked$", key):
            out[key] = value.clone()
        elif re.match(r"^tcn\.\d+\.skip\.weight$", key):
            out[key] = value.index_select(0, main_idx).index_select(1, main_idx).clone()
        elif key == "pool.score.0.weight":
            out[key] = value.index_select(0, attn_idx).index_select(1, main_idx).clone()
        elif key == "pool.score.0.bias":
            out[key] = slice_1d(value, attn_idx)
        elif key == "pool.score.2.weight":
            out[key] = value.index_select(1, attn_idx).clone()
        elif key == "pool.score.2.bias":
            out[key] = value.clone()
        elif key in {"head.0.weight", "head.0.bias"}:
            out[key] = slice_1d(value, main_idx)
        elif key == "head.2.weight":
            out[key] = value.index_select(1, main_idx).clone()
        elif key == "head.2.bias":
            out[key] = value.clone()
        else:
            raise KeyError(f"Unsupported state_dict key {key!r}; refusing partial pruning.")

    for b in range(num_blocks):
        for key in (f"tcn.{b}.net.0.weight", f"tcn.{b}.net.4.weight"):
            if key not in out:
                raise KeyError(f"Missing expected key after pruning: {key}")
    return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def approximate_macs(channels, input_dim, timesteps, num_blocks, num_classes):
    c, t = int(channels), int(timesteps)
    macs = t * c * input_dim * 5
    macs += num_blocks * 2 * t * c * c * 5
    ah = c // 2
    macs += t * ah * c
    macs += t * ah
    macs += t * c
    macs += c * num_classes
    return int(macs)


@torch.no_grad()
def model_outputs(model, windows, batch_size=128):
    model.eval()
    outputs = []
    for start in range(0, windows.shape[0], batch_size):
        x = torch.from_numpy(windows[start:start + batch_size]).float()
        outputs.append(model(x).cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 2))


def load_windows(path, win_len, input_dim, max_windows):
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (win_len, input_dim):
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (win_len, input_dim):
        raise ValueError(f"Expected [N,{win_len},{input_dim}], got {arr.shape}.")
    return arr[:max_windows]


def similarity_report(reference_logits, candidate_logits):
    ref_pred = reference_logits.argmax(dim=1)
    cand_pred = candidate_logits.argmax(dim=1)
    return {
        "agreement": float((ref_pred == cand_pred).float().mean().item()),
        "mean_abs_logit_diff": float((reference_logits - candidate_logits).abs().mean().item()),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Structured channel pruning for TCNAttnClassifier")
    p.add_argument("--checkpoint", default="lstm_fall_detector1_30hz_ft_3s.pt")
    p.add_argument("--targets", type=int, nargs="+", default=[64, 32, 16])
    p.add_argument("--output-dir", default="pruned_models")
    p.add_argument("--windows-npy", default=None,
                   help="Optional already-normalized [N,T,6] windows for output comparison")
    p.add_argument("--max-windows", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    source_state = unwrap_state_dict(checkpoint)
    arch = infer_architecture(source_state)

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    dropout = float(config.get("dropout", 0.2))
    win_len = int(config.get("win_len", 60))

    original = TCNAttnClassifier(
        input_dim=arch["input_dim"], channels=arch["channels"],
        num_blocks=arch["num_blocks"], dropout=dropout,
        num_classes=arch["num_classes"],
    )
    original.load_state_dict(source_state, strict=True)
    original.eval()

    targets = []
    for target in args.targets:
        if target >= arch["channels"]:
            raise ValueError(f"Target {target} must be smaller than source width {arch['channels']}.")
        if target < 2 or target % 2:
            raise ValueError("Target channels must be even and >= 2.")
        if target not in targets:
            targets.append(target)

    main_score = main_channel_importance(source_state, arch["channels"], arch["num_blocks"])
    attn_score = attention_hidden_importance(source_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.windows_npy:
        windows = load_windows(args.windows_npy, win_len, arch["input_dim"], args.max_windows)
        eval_name = f"real windows: {args.windows_npy}"
    else:
        windows = np.random.RandomState(args.seed).normal(
            0.0, 1.0, size=(64, win_len, arch["input_dim"])
        ).astype(np.float32)
        eval_name = "synthetic normalized windows"

    ref_logits = model_outputs(original, windows)
    original_params = count_parameters(original)
    original_macs = approximate_macs(
        arch["channels"], arch["input_dim"], win_len,
        arch["num_blocks"], arch["num_classes"]
    )

    print(f"Source: {checkpoint_path}")
    print(
        f"Architecture: input_dim={arch['input_dim']} channels={arch['channels']} "
        f"blocks={arch['num_blocks']} classes={arch['num_classes']} win_len={win_len}"
    )
    print(f"Original params={original_params:,} approx_MACs={original_macs:,}")
    print(f"Similarity input: {eval_name}")

    stem = checkpoint_path.stem
    for target in targets:
        main_idx = top_indices(main_score, target)
        attn_idx = top_indices(attn_score, target // 2)
        pruned_state = prune_state_dict(source_state, main_idx, attn_idx, arch["num_blocks"])

        model = TCNAttnClassifier(
            input_dim=arch["input_dim"], channels=target,
            num_blocks=arch["num_blocks"], dropout=dropout,
            num_classes=arch["num_classes"],
        )
        model.load_state_dict(pruned_state, strict=True)
        model.eval()

        with torch.no_grad():
            smoke = model(torch.zeros(1, win_len, arch["input_dim"], dtype=torch.float32))
        if tuple(smoke.shape) != (1, arch["num_classes"]):
            raise RuntimeError(f"Unexpected output shape for C={target}: {tuple(smoke.shape)}")

        candidate_logits = model_outputs(model, windows)
        sim = similarity_report(ref_logits, candidate_logits)
        params = count_parameters(model)
        macs = approximate_macs(
            target, arch["input_dim"], win_len,
            arch["num_blocks"], arch["num_classes"]
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            saved = copy.deepcopy(checkpoint)
            saved["model_state_dict"] = pruned_state
        else:
            saved = {"model_state_dict": pruned_state}

        saved_config = copy.deepcopy(saved.get("config", {}))
        saved_config["hidden_dim"] = target
        saved_config["num_layers"] = arch["num_blocks"]
        saved_config["win_len"] = win_len
        saved_config["dropout"] = dropout
        saved["config"] = saved_config
        saved["pruning"] = {
            "method": "structured_magnitude_channel_pruning_no_finetune",
            "source_checkpoint": str(checkpoint_path),
            "source_channels": arch["channels"],
            "target_channels": target,
            "num_blocks": arch["num_blocks"],
            "selected_main_channels": main_idx.tolist(),
            "selected_attention_channels": attn_idx.tolist(),
            "parameter_count": params,
            "approx_macs": macs,
            "source_parameter_count": original_params,
            "source_approx_macs": original_macs,
            "similarity_input": eval_name,
            "class_agreement_vs_original": sim["agreement"],
            "mean_abs_logit_diff_vs_original": sim["mean_abs_logit_diff"],
            "note": "No fine-tuning was performed; similarity is not labeled accuracy evaluation.",
        }

        out_path = output_dir / f"{stem}_c{target}.pt"
        torch.save(saved, out_path)

        print(f"\nC={target}: {out_path}")
        print(f"  params={params:,} ({params / original_params:.3f}x original)")
        print(f"  approx_MACs={macs:,} ({macs / original_macs:.3f}x original)")
        print(f"  class_agreement_vs_original={sim['agreement']:.4f}")
        print(f"  mean_abs_logit_diff_vs_original={sim['mean_abs_logit_diff']:.6f}")
        print(f"  smoke_output_shape={tuple(smoke.shape)}")


if __name__ == "__main__":
    main()
