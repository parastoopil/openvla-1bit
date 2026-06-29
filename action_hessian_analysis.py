#!/usr/bin/env python3
"""
Action-Token Hessian Divergence Analysis for OpenVLA
=====================================================
For each of the 32 Llama decoder layers, computes two Hessians:

  H_all:    accumulated from ALL token positions (image + instruction + action)
  H_action: accumulated from ONLY the 7 action-token positions

Then computes a salience divergence score per layer:
  - Salience score per column j:  s_j = mean_i(W_ij² / H_diag_j²)
  - Divergence = 1 - cosine_sim(salience_all, salience_action)

High divergence means this layer "behaves differently" on action tokens
vs. image/instruction tokens — i.e., standard GPTQ calibration (H_all)
selects different salient weights than action-specific calibration (H_action).

Those layers are where 1-bit quantization is most likely to destroy gripper
accuracy, because the salient weights identified by H_all may not be the
ones actually responsible for outputting action tokens.

Usage:
  python action_hessian_analysis.py --device cuda:0 --nsamples 128

Outputs:
  output/action_layer_analysis/layer_divergence.json
  output/action_layer_analysis/action_layer_divergence.png
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantize_utils import HessianAccumulator

HF_CACHE      = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OPENVLA_ID    = "openvla/openvla-7b"
UNNORM_KEY    = "bridge_orig"
_BRIDGE_REPO  = "nvidia/BridgeData2_LeRobot_v3"
_BRIDGE_VIDEO = "videos/observation.images.image_0/chunk-000/file-000.mp4"
_BRIDGE_META  = "data/chunk-000/file-000.parquet"
_BRIDGE_TASKS = "meta/tasks.parquet"
_BRIDGE_NFRAMES = 12282


# ── Model ─────────────────────────────────────────────────────────────────────

def load_openvla():
    from transformers import AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    processor = AutoProcessor.from_pretrained(
        OPENVLA_ID, trust_remote_code=True, cache_dir=HF_CACHE)
    model_cls = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction",
        OPENVLA_ID, cache_dir=HF_CACHE, trust_remote_code=True)
    model = model_cls.from_pretrained(
        OPENVLA_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, cache_dir=HF_CACHE, attn_implementation="eager")
    model.eval()
    return model, processor


# ── Action tokenization ───────────────────────────────────────────────────────

def get_norm_stats(model):
    """Extract bridge_orig action normalization stats from model config."""
    ns = getattr(model, "norm_stats", {})
    if UNNORM_KEY not in ns:
        raise RuntimeError(
            f"norm_stats['{UNNORM_KEY}'] not found. "
            f"Available keys: {list(ns.keys())}")
    entry = ns[UNNORM_KEY]
    # Prismatic VLMs store stats nested under 'action'
    return entry.get("action", entry)


def action_to_token_ids(action, norm_stats):
    """
    Convert a continuous 7-DOF action vector to 7 OpenVLA vocabulary token IDs.

    OpenVLA discretises each dimension into 256 bins over [q01, q99] and maps
    bin k → token (31745 + k).  Bins outside [q01, q99] are clipped.
    """
    q01 = np.asarray(norm_stats["q01"], dtype=np.float32)
    q99 = np.asarray(norm_stats["q99"], dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)

    norm = 2.0 * (action - q01) / (q99 - q01 + 1e-8) - 1.0
    norm = np.clip(norm, -1.0, 1.0)
    bins = np.round((norm + 1.0) / 2.0 * 255.0).astype(np.int32)
    bins = np.clip(bins, 0, 255)
    return (31745 + bins).tolist()


# ── BridgeData V2 sample loading (with ground-truth action tokens) ────────────

def collect_bridge_samples(processor, norm_stats, nsamples=128):
    """
    Returns list of dicts:
      'inputs':          processor output (input_ids, pixel_values)
      'action_token_ids': list of 7 ints  ← teacher-forcing targets
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import imageio as iio
    from PIL import Image

    meta_path = hf_hub_download(
        repo_id=_BRIDGE_REPO, filename=_BRIDGE_META,
        repo_type="dataset", cache_dir=HF_CACHE)
    df = pd.read_parquet(meta_path)
    df = df[df["index"] < _BRIDGE_NFRAMES].copy()

    tasks_df = pd.read_parquet(hf_hub_download(
        repo_id=_BRIDGE_REPO, filename=_BRIDGE_TASKS,
        repo_type="dataset", cache_dir=HF_CACHE))
    task_map = {int(row["task_index"]): str(text)
                for text, row in tasks_df.iterrows()}

    stride     = max(1, _BRIDGE_NFRAMES // nsamples)
    frame_idxs = list(range(0, _BRIDGE_NFRAMES, stride))[:nsamples]
    selected   = df[df["index"].isin(frame_idxs)].drop_duplicates("index")
    selected   = selected.sort_values("index")

    vid_path = hf_hub_download(
        repo_id=_BRIDGE_REPO, filename=_BRIDGE_VIDEO,
        repo_type="dataset", cache_dir=HF_CACHE)

    reader  = iio.get_reader(vid_path, "ffmpeg")
    samples = []
    for _, row in selected.iterrows():
        fidx = int(row["index"])
        try:
            frame = reader.get_data(fidx)
            img   = Image.fromarray(frame)
            task  = task_map.get(int(row["task_index"]), "pick up the object")
            instr = f"In: What action should the robot take to {task}?\nOut:"
            inputs = processor(instr, img, return_tensors="pt")
            act_ids = action_to_token_ids(list(row["action"]), norm_stats)
            samples.append({
                "inputs":           {k: v for k, v in inputs.items()},
                "action_token_ids": act_ids,
            })
        except Exception as e:
            print(f"  warning: frame {fidx} skipped — {e}")
    reader.close()
    print(f"[data] Loaded {len(samples)} samples with action tokens.")
    return samples


# ── Divergence metric ─────────────────────────────────────────────────────────

def salience_divergence(H_all: torch.Tensor, H_action: torch.Tensor,
                        W: torch.Tensor) -> tuple:
    """
    Returns:
      divergence : 1 - cosine_sim between column-wise salience vectors
                   (higher → layer behaves more differently on action tokens)
      mismatch   : fraction of top-10%-salient columns that differ between
                   H_all and H_action salience rankings
    """
    diag_all = torch.diag(H_all).clamp(min=1e-8)
    diag_act = torch.diag(H_action).clamp(min=1e-8)

    # Per-column salience: average over output rows
    sal_all = (W.float() ** 2 / diag_all.unsqueeze(0) ** 2).mean(0)  # [ic]
    sal_act = (W.float() ** 2 / diag_act.unsqueeze(0) ** 2).mean(0)

    sal_all_n = sal_all / sal_all.sum().clamp(min=1e-8)
    sal_act_n = sal_act / sal_act.sum().clamp(min=1e-8)

    cos = torch.dot(sal_all_n, sal_act_n) / (
        sal_all_n.norm() * sal_act_n.norm() + 1e-8)
    divergence = float(1.0 - cos.item())

    k = max(1, W.shape[1] // 10)
    top_all = set(sal_all_n.topk(k).indices.tolist())
    top_act = set(sal_act_n.topk(k).indices.tolist())
    mismatch = 1.0 - len(top_all & top_act) / k

    return divergence, mismatch


# ── Main analysis loop ────────────────────────────────────────────────────────

@torch.no_grad()
def analyze(model, processor, dev, nsamples=128):
    norm_stats   = get_norm_stats(model)
    samples      = collect_bridge_samples(processor, norm_stats, nsamples)
    nsamples     = len(samples)

    lm          = model.language_model
    lm.config.use_cache = False
    layers      = lm.model.layers
    hidden_size = lm.config.hidden_size
    model_dtype = next(lm.parameters()).dtype

    # ── Step 1: run vision backbone to get first-layer LLM inputs ────────────
    for comp in [model.vision_backbone, model.projector, lm.model.embed_tokens]:
        comp.to(dev)
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb.to(dev)
    layers[0].to(dev)

    # Probe sequence length
    probe = {"seqlen": None, "kwargs": {}}
    class _Prober(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, x, **kwargs):
            probe["seqlen"] = x.shape[1]; probe["kwargs"].update(kwargs)
            raise ValueError
    layers[0] = _Prober(layers[0])
    try:
        model(input_ids=samples[0]["inputs"]["input_ids"].to(dev),
              pixel_values=samples[0]["inputs"]["pixel_values"].to(dev, dtype=model_dtype))
    except ValueError: pass
    layers[0] = layers[0].module
    seqlen = probe["seqlen"]
    print(f"[analysis] Image+instruction length: {seqlen} tokens")
    print(f"[analysis] Action tokens will be appended at positions {seqlen}–{seqlen+6}")

    # Collect extended hidden states (image+instruction embedding + action embeddings)
    ext_seqlen  = seqlen + 7
    action_start = seqlen
    all_inps     = torch.zeros((nsamples, ext_seqlen, hidden_size),
                               dtype=model_dtype, device="cpu")
    ci = {"n": 0}
    layer_kwargs = dict(probe["kwargs"])

    class _Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, x, **kwargs):
            idx = ci["n"]
            if idx < nsamples:
                s       = samples[idx]
                act_ids = torch.tensor(
                    s["action_token_ids"], dtype=torch.long, device=dev).unsqueeze(0)
                act_emb = lm.model.embed_tokens(act_ids).to(x.dtype)  # [1,7,H]
                x_ext   = torch.cat([x, act_emb], dim=1)              # [1,seqlen+7,H]
                sl = min(x_ext.shape[1], ext_seqlen)
                all_inps[idx, :sl] = x_ext[0, :sl].detach().cpu()
                layer_kwargs.update(kwargs)
                ci["n"] += 1
            raise ValueError

    layers[0] = _Catcher(layers[0])
    for s in samples:
        try:
            model(input_ids=s["inputs"]["input_ids"].to(dev),
                  pixel_values=s["inputs"]["pixel_values"].to(dev, dtype=model_dtype))
        except ValueError: pass
    layers[0] = layers[0].module

    # Compute position embeddings for the EXTENDED sequence while rotary_emb is
    # still on GPU — transformers v5 requires precomputed (cos, sin) and has no
    # fallback inside attention.
    pos_ids_ext = torch.arange(ext_seqlen, dtype=torch.long, device=dev).unsqueeze(0)
    if hasattr(lm.model, "rotary_emb"):
        _dummy = torch.zeros(1, ext_seqlen, hidden_size, dtype=model_dtype, device=dev)
        pos_emb_ext = lm.model.rotary_emb(_dummy, pos_ids_ext)
    else:
        pos_emb_ext = None

    for comp in [model.vision_backbone, model.projector, lm.model.embed_tokens]:
        comp.cpu()
    if hasattr(lm.model, "rotary_emb"): lm.model.rotary_emb.cpu()
    layers[0].cpu()
    torch.cuda.empty_cache()
    print(f"[analysis] Captured {ci['n']} extended activation samples.")

    inps = all_inps.to(dev)
    outs = torch.zeros_like(inps)

    # Drop all sequence-length-dependent kwargs from the probe; replace with
    # versions sized for the extended (283-token) sequence.
    _SKIP  = {"past_key_values", "use_cache", "cache_position",
              "position_embeddings", "attention_mask", "position_ids"}
    fwd_kw = {k: v for k, v in layer_kwargs.items() if k not in _SKIP}

    fwd_kw["position_ids"] = pos_ids_ext
    if pos_emb_ext is not None:
        fwd_kw["position_embeddings"] = pos_emb_ext  # (cos, sin) for 283 tokens

    # 4D additive causal mask: 0 where attending, -inf where masked
    _causal = torch.zeros(1, 1, ext_seqlen, ext_seqlen,
                          dtype=model_dtype, device=dev)
    _causal.masked_fill_(
        torch.triu(torch.ones(ext_seqlen, ext_seqlen,
                              dtype=torch.bool, device=dev), diagonal=1),
        float("-inf"))
    fwd_kw["attention_mask"] = _causal

    # ── Step 2: layer-by-layer dual Hessian ──────────────────────────────────
    results = []

    for li in range(len(layers)):
        print(f"\n[layer {li+1:2d}/{len(layers)}]", end="  ", flush=True)
        layer  = layers[li].to(dev)
        subset = {n: m for n, m in layer.named_modules()
                  if isinstance(m, nn.Linear)}

        h_all = {n: HessianAccumulator(m.weight.shape[1], dev) for n, m in subset.items()}
        h_act = {n: HessianAccumulator(m.weight.shape[1], dev) for n, m in subset.items()}

        hooks = []
        for name, sublayer in subset.items():
            def _hook(_, inp, __, _n=name):
                x = inp[0].detach()          # [1, ext_seqlen, ic]
                h_all[_n].add_batch(x.reshape(-1, x.shape[-1]))
                # Action-token positions only
                x_act = x[:, action_start:, :]
                if x_act.shape[1] > 0:
                    h_act[_n].add_batch(x_act.reshape(-1, x_act.shape[-1]))
            hooks.append(sublayer.register_forward_hook(_hook))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kw)[0]

        for h in hooks: h.remove()

        # Compute per-sublayer divergence
        sub_divs, sub_mm = [], []
        for name, sublayer in subset.items():
            W  = sublayer.weight.data
            Ha = h_all[name].get_hessian()
            Hb = h_act[name].get_hessian()
            d, mm = salience_divergence(Ha, Hb, W)
            sub_divs.append(d)
            sub_mm.append(mm)
            print(f"{name.split('.')[-1]}({d:.3f})", end=" ", flush=True)

        mean_div = float(np.mean(sub_divs))
        mean_mm  = float(np.mean(sub_mm))
        results.append({
            "layer":            li,
            "mean_divergence":  mean_div,
            "mean_mismatch":    mean_mm,
            "per_sublayer":     {n: {"divergence": d, "mismatch": mm}
                                 for n, d, mm in zip(subset, sub_divs, sub_mm)},
        })
        print(f"→ mean div={mean_div:.4f}  mismatch={mean_mm:.1%}")

        layers[li] = layer.cpu()
        del layer, h_all, h_act
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    lm.config.use_cache = True
    return results


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_results(results, outdir):
    layers = [r["layer"] + 1 for r in results]
    divs   = [r["mean_divergence"] for r in results]
    mm     = [r["mean_mismatch"] * 100 for r in results]
    mean_d = np.mean(divs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    colors = ["tomato" if d > mean_d * 1.3 else "steelblue" for d in divs]
    ax1.bar(layers, divs, color=colors)
    ax1.axhline(mean_d, color="black", linestyle="--", linewidth=1,
                label=f"mean = {mean_d:.4f}")
    ax1.set_ylabel("Salience divergence\n(1 − cosine sim)")
    ax1.set_title("Action-Token vs. All-Token Salience Divergence per Layer\n"
                  "Red = layer treats action tokens differently → protect these")
    ax1.legend(fontsize=9)

    ax2.bar(layers, mm, color=colors)
    ax2.axhline(np.mean(mm), color="black", linestyle="--", linewidth=1,
                label=f"mean = {np.mean(mm):.1f}%")
    ax2.set_xlabel("Decoder layer index")
    ax2.set_ylabel("Top-10% weight mismatch (%)")
    ax2.set_title("Fraction of action-salient columns NOT in all-token top-10%\n"
                  "Higher → standard GPTQ picks the wrong weights in this layer")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    p = Path(outdir) / "action_layer_divergence.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {p}")
    return p


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Action-token Hessian divergence analysis for OpenVLA")
    p.add_argument("--device",   default="cuda:0")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--output",   default="output/action_layer_analysis")
    return p.parse_args()


def main():
    args   = parse_args()
    dev    = torch.device(args.device)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] Loading OpenVLA...")
    model, processor = load_openvla()

    results = analyze(model, processor, dev, nsamples=args.nsamples)

    # Save JSON
    jpath = outdir / "layer_divergence.json"
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] JSON → {jpath}")

    # Plot
    plot_results(results, outdir)

    # Summary
    ranked = sorted(results, key=lambda r: r["mean_divergence"], reverse=True)
    print("\n" + "=" * 62)
    print("  ACTION-CRITICAL LAYERS  (top Hessian divergence)")
    print("=" * 62)
    print(f"  {'Layer':>5}  {'Divergence':>11}  {'Mismatch':>10}  Interpretation")
    print(f"  {'-'*5}  {'-'*11}  {'-'*10}  {'-'*30}")
    for r in ranked[:8]:
        flag = " ← protect" if r["mean_divergence"] > np.mean(
            [x["mean_divergence"] for x in results]) * 1.3 else ""
        print(f"  {r['layer']+1:>5}  {r['mean_divergence']:>11.4f}  "
              f"{r['mean_mismatch']:>9.1%}  {flag}")
    print()
    print("  Recommendation: give these layers higher precision (e.g. 2-bit")
    print("  non-salient instead of 1-bit) in a targeted re-quantization.")
    print("=" * 62)

    # Write top layers to a simple text file for use by quant scripts
    top_layers = [r["layer"] for r in ranked
                  if r["mean_divergence"] > np.mean(
                      [x["mean_divergence"] for x in results]) * 1.3]
    (outdir / "action_critical_layers.txt").write_text(
        "\n".join(str(l) for l in sorted(top_layers)))
    print(f"\n[save] Action-critical layer indices → "
          f"{outdir / 'action_critical_layers.txt'}")
    print(f"       Layers: {sorted(top_layers)}")


if __name__ == "__main__":
    main()
