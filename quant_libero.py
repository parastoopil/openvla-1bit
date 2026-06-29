#!/usr/bin/env python3
"""
Quantize an OpenVLA LIBERO-finetuned checkpoint with LIBERO calibration data.

Pipeline:
  FP16 finetuned ckpt → GPTQ (LIBERO calib) → quantized_llm/ dir
  Calibration: images + instructions from lerobot/libero_*_image parquet

Usage:
  python quant_libero.py \
      --ckpt  /path/to/openvla-7b-finetuned-libero-spatial \
      --suite libero_spatial \
      --n_calib 128 \
      --output output/libero_spatial_quantized

Then evaluate:
  MUJOCO_GL=egl conda run -n libero-eval python run_libero_eval.py \
      --ckpt /path/to/openvla-7b-finetuned-libero-spatial \
      --suite libero_spatial \
      --quant_path output/libero_spatial_quantized/quantized_llm
"""
import argparse
import io
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("LIBERO_PATH", "."))
sys.path.insert(0, os.environ.get("OPENVLA_PATH", "."))

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# HuggingFace dataset IDs for lerobot LIBERO demos (parquet, no TF needed)
LEROBOT_SUITE = {
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_object":  "lerobot/libero_object_image",
    "libero_goal":    "lerobot/libero_goal_image",
    "libero_10":      "lerobot/libero_10_image",
}


def load_finetuned_model(ckpt_path: str):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print(f"[load] Model from {ckpt_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
    with open(os.path.join(ckpt_path, "dataset_statistics.json")) as f:
        full_stats = json.load(f)
    model.norm_stats = full_stats
    model.eval()
    return model, processor, full_stats


def collect_libero_calib(suite: str, processor, norm_stats: dict,
                          n_samples: int = 128) -> list:
    """
    Load calibration samples from lerobot parquet files.
    Returns list of {input_ids, pixel_values} dicts — same format
    expected by quantize_mixed_precision in quant_robot_calibrated.py.

    Data layout (verified):
      data/chunk-000/file-NNN.parquet
      columns: observation.images.image (dict {'bytes':..., 'path':...}),
               action (7-float array), task_index (int)
      meta/tasks.parquet: index=task_description, column=task_index
    """
    from PIL import Image
    import pandas as pd
    import glob

    hf_repo   = LEROBOT_SUITE[suite]
    snap_root = os.path.join(
        HF_CACHE, f"datasets--lerobot--{hf_repo.split('/')[-1]}", "snapshots")
    snap      = sorted(os.listdir(snap_root))[-1]
    snap_dir  = os.path.join(snap_root, snap)

    print(f"[calib] Loading from {snap_dir}")

    # tasks.parquet: task description is the DataFrame index
    tasks_df = pd.read_parquet(os.path.join(snap_dir, "meta", "tasks.parquet"))
    task_map = {int(v): str(k) for k, v in tasks_df["task_index"].items()}

    # Collect all parquet files (may be in data/chunk-NNN/ subdirs)
    all_parquets = sorted(glob.glob(os.path.join(snap_dir, "data", "**", "*.parquet"),
                                    recursive=True))
    stride    = max(1, len(all_parquets) // max(1, n_samples // 6))
    sel_files = all_parquets[::stride]

    IMG_COL = "observation.images.image"

    samples = []
    for fpath in sel_files:
        if len(samples) >= n_samples:
            break
        try:
            df = pd.read_parquet(fpath)
            if IMG_COL not in df.columns:
                continue
            row_stride = max(1, len(df) // 6)
            for _, row in df.iloc[::row_stride].iterrows():
                if len(samples) >= n_samples:
                    break
                try:
                    img_data = row[IMG_COL]
                    raw = img_data["bytes"] if isinstance(img_data, dict) else img_data
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    img = img.resize((224, 224))

                    task_idx  = int(row["task_index"])
                    task_desc = task_map.get(task_idx, "pick up the object")
                    prompt = (f"In: What action should the robot take to "
                              f"{task_desc.lower()}?\nOut:")

                    inputs = processor(prompt, img, return_tensors="pt")
                    samples.append({
                        "input_ids":    inputs["input_ids"],
                        "pixel_values": inputs["pixel_values"],
                    })
                except Exception:
                    pass
        except Exception as e:
            print(f"  [warn] {os.path.basename(fpath)}: {e}")

    print(f"[calib] Collected {len(samples)} samples")
    return samples


def find_linear_layers(module: nn.Module) -> dict:
    """Return {name: Linear} for all nn.Linear in module."""
    result = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            result[name] = child
    return result


def quantize_libero(model, calib_samples: list, dev: torch.device,
                    n_salient: int = 50,
                    blocksize: int = 128,
                    percdamp: float = 0.01) -> None:
    """
    In-place GPTQ quantization of model.language_model using LIBERO calib data.
    Replicates quantize_mixed_precision from quant_robot_calibrated.py exactly.
    """
    import time as _time
    from quantize_utils import HessianAccumulator, mixed_precision_gptq

    lm          = model.language_model
    lm.config.use_cache = False
    layers      = lm.model.layers
    hidden_size = lm.config.hidden_size
    model_dtype = next(lm.parameters()).dtype
    nsamples    = len(calib_samples)

    print(f"\n[quant] {nsamples} calib samples | 4/1-bit mixed | "
          f"n_salient={n_salient}")

    # ── Step 1: run vision+embedding to get LLM boundary activations ──────────
    model.vision_backbone.to(dev)
    model.projector.to(dev)
    lm.model.embed_tokens.to(dev)
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb.to(dev)
    layers[0].to(dev)

    # Probe to get multimodal seqlen and layer kwargs
    probe = {"seqlen": None, "kwargs": {}}

    class _Prober(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, x, **kw):
            probe["seqlen"] = x.shape[1]; probe["kwargs"].update(kw)
            raise ValueError("probe")

    layers[0] = _Prober(layers[0])
    try:
        s0 = calib_samples[0]
        model(input_ids=s0["input_ids"].to(dev),
              pixel_values=s0["pixel_values"].to(dev, dtype=model_dtype))
    except ValueError:
        pass
    layers[0] = layers[0].module
    seqlen = probe["seqlen"]
    print(f"[quant] Multimodal seqlen = {seqlen}")

    # Capture all first-layer inputs
    inps  = torch.zeros((nsamples, seqlen, hidden_size), dtype=model_dtype, device=dev)
    ci    = {"n": 0}
    layer_kwargs = dict(probe["kwargs"])

    class _Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, x, **kw):
            idx = ci["n"]
            if idx < nsamples:
                sl = min(x.shape[1], seqlen)
                inps[idx, :sl] = x[0, :sl].detach()
                layer_kwargs.update(kw); ci["n"] += 1
            raise ValueError("caught")

    layers[0] = _Catcher(layers[0])
    for s in calib_samples:
        try:
            model(input_ids=s["input_ids"].to(dev),
                  pixel_values=s["pixel_values"].to(dev, dtype=model_dtype))
        except ValueError:
            pass
    layers[0] = layers[0].module
    print(f"[quant] Captured {ci['n']} activation samples")

    # Move non-layer components back to CPU
    for comp in [model.vision_backbone, model.projector, lm.model.embed_tokens]:
        comp.cpu()
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb.cpu()
    layers[0].cpu()
    torch.cuda.empty_cache()

    outs   = torch.zeros_like(inps)
    _SKIP  = {"past_key_values", "use_cache", "cache_position"}
    fwd_kw = {k: v for k, v in layer_kwargs.items() if k not in _SKIP}

    t_total = _time.time()

    # ── Step 2: layer-by-layer quantization ──────────────────────────────────
    for li, layer in enumerate(layers):
        t0    = _time.time()
        layer = layer.to(dev)
        sub   = find_linear_layers(layer)

        # Register hooks to accumulate Hessians
        h_accum = {name: HessianAccumulator(m.weight.shape[1], dev)
                   for name, m in sub.items()}
        hooks   = []
        for name in sub:
            def _hook(_, inp, __, _n=name):
                x = inp[0].detach()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                h_accum[_n].add_batch(x)
            hooks.append(sub[name].register_forward_hook(_hook))

        # Forward all calib samples through this layer to accumulate Hessians
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kw)[0]
        for h in hooks:
            h.remove()

        # GPTQ per sublayer
        for name, m in sub.items():
            W = m.weight.data.float()
            H = h_accum[name].get_hessian()
            print(f"  [{li+1:2d}] {name} [{W.shape[0]}×{W.shape[1]}] ...",
                  end=" ", flush=True)
            W_q, elapsed, err = mixed_precision_gptq(
                W, H, blocksize=blocksize, percdamp=percdamp,
                n_salient_per_col=n_salient)
            m.weight.data = W_q.to(m.weight.dtype)
            print(f"  err={err:.3f}  ({elapsed:.0f}s)")
            del H, W, W_q, h_accum[name]

        # Forward again to get accurate outputs for next layer
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kw)[0]

        layers[li] = layer.cpu()
        inps, outs = outs, inps          # swap: next layer's input = this layer's output
        del layer; torch.cuda.empty_cache()

        print(f"  Layer {li+1:2d}/32 done in {_time.time()-t0:.1f}s")

    print(f"\n[quant] All 32 layers done in "
          f"{(_time.time()-t_total)/60:.1f} min total.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    required=True,
                    help="Local path to finetuned OpenVLA checkpoint")
    ap.add_argument("--suite",   required=True,
                    choices=list(LEROBOT_SUITE))
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--n_salient", type=int, default=50)
    ap.add_argument("--output",  required=True)
    ap.add_argument("--device",  default="cuda:0")
    args = ap.parse_args()

    dev     = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model + processor + norm stats
    model, processor, full_stats = load_finetuned_model(args.ckpt)
    norm_stats = full_stats[args.suite]
    ns = norm_stats.get("action", norm_stats)

    # Collect LIBERO calibration samples
    calib = collect_libero_calib(args.suite, processor, ns, n_samples=args.n_calib)
    if len(calib) < 16:
        raise RuntimeError(f"Only {len(calib)} calib samples — need at least 16")

    # Move model to GPU and quantize in-place
    model.to(dev)
    quantize_libero(model, calib, dev, n_salient=args.n_salient)

    # Save quantized LLM
    quant_lm_dir = out_dir / "quantized_llm"
    quant_lm_dir.mkdir(exist_ok=True)
    print(f"\n[save] Quantized LLM → {quant_lm_dir}")
    model.language_model.save_pretrained(str(quant_lm_dir))

    # Copy dataset_statistics.json so eval knows action normalization
    shutil.copy(os.path.join(args.ckpt, "dataset_statistics.json"),
                out_dir / "dataset_statistics.json")
    print(f"[save] dataset_statistics.json copied")
    print(f"\nRun eval with:")
    print(f"  MUJOCO_GL=egl conda run -n libero-eval python run_libero_eval.py \\")
    print(f"    --ckpt {args.ckpt} --suite {args.suite} \\")
    print(f"    --quant_path {quant_lm_dir}")


if __name__ == "__main__":
    main()
