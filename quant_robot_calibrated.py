#!/usr/bin/env python3
"""
Robot-Calibrated Mixed-Precision Quantization of OpenVLA-7B (Experiment D)
===========================================================================
Uses BridgeData V2 (OpenVLA's primary training dataset) for Hessian
calibration, replacing the low-diversity ALOHA data used in Experiment B.

Pipeline:
  1. Stream 128+ diverse frames from nvidia/BridgeData2_LeRobot_v3,
     sampling across different episodes / tasks / scenes.
  2. Feed each frame (image + language instruction) through the full
     OpenVLA model (vision backbone → projector → LLM) and capture
     LLM hidden-state activations at each decoder layer.
  3. Build per-weight Hessians from these activations and run
     mixed-precision PTQ: salient → 4-bit, non-salient → 1-bit BRAQ.
  4. Evaluate action accuracy vs. FP16 oracle.

Why BridgeData V2 beats ALOHA:
  - ALOHA: 64 near-identical frames, rank-1 Hessian, sign-flipped actions.
  - BridgeData V2: 60k+ episodes, diverse tasks/scenes/objects from
    OpenVLA's own training distribution → full-rank Hessian, correct salience.

Usage:
  python quant_robot_calibrated.py --device cuda:0 --eval_action

  # More calibration samples for better Hessian:
  python quant_robot_calibrated.py --nsamples 256 --eval_action
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from quantize_utils import (
    HessianAccumulator,
    mixed_precision_gptq,
    pure_1bit_gptq,
)

HF_CACHE   = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OPENVLA_ID = "openvla/openvla-7b"
ACTION_DIM = 7

_INSTR_KEY_CANDIDATES = [
    "task.description",
    "language_instruction",
    "annotation.human.task_language_instruction",
    "task",
]


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_openvla(model_id=OPENVLA_ID, lora_merged_path=None):
    from transformers import AutoProcessor, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    print("[load] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=HF_CACHE
    )
    print("[load] Resolving model class...")
    model_cls = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction",
        model_id, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    print("[load] Loading weights (bfloat16)...")
    model = model_cls.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, cache_dir=HF_CACHE, attn_implementation="eager",
    )

    if lora_merged_path is not None:
        print(f"[load] Installing LoRA-merged backbone from {lora_merged_path}")
        merged_lm = AutoModelForCausalLM.from_pretrained(
            lora_merged_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model.language_model = merged_lm

    model.eval()
    return model, processor


def find_linear_layers(module: nn.Module) -> dict:
    return {name: layer for name, layer in module.named_modules()
            if isinstance(layer, nn.Linear)}


# ══════════════════════════════════════════════════════════════════════════════
# BridgeData V2 calibration data
# ══════════════════════════════════════════════════════════════════════════════

_BRIDGE_REPO = "nvidia/BridgeData2_LeRobot_v3"
# First video chunk: 198 MB, 12,282 frames at 5 fps, 640×480 AV1
_BRIDGE_VIDEO  = "videos/observation.images.image_0/chunk-000/file-000.mp4"
_BRIDGE_META   = "data/chunk-000/file-000.parquet"
_BRIDGE_TASKS  = "meta/tasks.parquet"
_BRIDGE_NFRAMES = 12282   # frame count for file-000.mp4


def collect_bridge_inputs(processor, nsamples: int = 128) -> list:
    """
    Extract diverse calibration frames from nvidia/BridgeData2_LeRobot_v3.

    How:
      - Downloads the parquet metadata (already cached after first run).
      - Downloads the first video chunk (~198 MB, cached).
      - Reads frames at a uniform stride over the 12 k-frame video, ensuring
        samples span ~nsamples distinct episodes and tasks.
      - Uses imageio-ffmpeg (bundled AV1-capable ffmpeg) to decode frames.

    This video chunk covers ~345 distinct BridgeDataV2 episodes across many
    tasks (pour, pick, wipe, drawer, ...) — giving a well-conditioned Hessian.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import imageio as iio
    from PIL import Image

    # 1. Parquet metadata (already small, cached immediately)
    meta_path = hf_hub_download(
        repo_id=_BRIDGE_REPO, filename=_BRIDGE_META,
        repo_type="dataset", cache_dir=HF_CACHE,
    )
    df = pd.read_parquet(meta_path)
    # Keep only rows in the first video chunk (index < frame count)
    df = df[df["index"] < _BRIDGE_NFRAMES].copy()

    # 2. Task text lookup: task_index → description string
    tasks_path = hf_hub_download(
        repo_id=_BRIDGE_REPO, filename=_BRIDGE_TASKS,
        repo_type="dataset", cache_dir=HF_CACHE,
    )
    tasks_df = pd.read_parquet(tasks_path)
    # tasks_df index = task text, column task_index = integer ID
    task_map = {int(row["task_index"]): str(text)
                for text, row in tasks_df.iterrows()}

    # 3. Select nsamples frames at uniform stride (in-order → fast seek)
    stride    = max(1, _BRIDGE_NFRAMES // nsamples)
    frame_idxs = list(range(0, _BRIDGE_NFRAMES, stride))[:nsamples]
    selected  = df[df["index"].isin(frame_idxs)].drop_duplicates("index")
    selected  = selected.sort_values("index")   # sequential read → fast

    # 4. Download video chunk (198 MB, cached after first run)
    print(f"[calib] Downloading BridgeData V2 video chunk (~198 MB, cached after first run)...")
    vid_path = hf_hub_download(
        repo_id=_BRIDGE_REPO, filename=_BRIDGE_VIDEO,
        repo_type="dataset", cache_dir=HF_CACHE,
    )
    print(f"[calib] Video cached at {vid_path}")

    # 5. Open with imageio-ffmpeg (ships its own AV1-capable ffmpeg binary)
    reader  = iio.get_reader(vid_path, "ffmpeg")
    samples = []
    errors  = 0
    unique_episodes = set()

    for _, row in selected.iterrows():
        fidx = int(row["index"])
        try:
            frame = reader.get_data(fidx)   # RGB uint8 (H, W, 3)
            img   = Image.fromarray(frame)

            task_txt = task_map.get(int(row["task_index"]), "pick up the object")
            instr    = f"In: What action should the robot take to {task_txt}?\nOut:"

            inputs = processor(instr, img, return_tensors="pt")
            samples.append({k: v for k, v in inputs.items()})
            unique_episodes.add(int(row["episode_index"]))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  warning: frame {fidx} skipped — {e}")

    reader.close()
    print(f"[calib] Collected {len(samples)} BridgeData V2 frames "
          f"from {len(unique_episodes)} distinct episodes "
          f"({errors} errors).")
    return samples


def collect_aloha_inputs(processor, nsamples: int = 64, stride: int = 50) -> list:
    """Fallback: ALOHA calibration data (used only if BridgeData V2 fails)."""
    from datasets import load_dataset

    _INSTRS = [
        "In: What action should the robot take to insert the peg into the socket?\nOut:",
        "In: What action should the robot take to grasp the component?\nOut:",
        "In: What action should the robot take to move to the target position?\nOut:",
        "In: What action should the robot take to complete the manipulation task?\nOut:",
        "In: What action should the robot take to align the objects?\nOut:",
    ]
    print(f"[calib] Falling back to ALOHA ({nsamples} frames, stride={stride})...")
    ds = load_dataset("lerobot/aloha_sim_insertion_scripted_image",
                      split="train", streaming=True, cache_dir=HF_CACHE)
    samples = []
    for i, item in enumerate(ds):
        if i % stride != 0:
            continue
        if len(samples) >= nsamples:
            break
        try:
            img    = item["observation.images.top"]
            instr  = _INSTRS[i % len(_INSTRS)]
            inputs = processor(instr, img, return_tensors="pt")
            samples.append({k: v for k, v in inputs.items()})
        except Exception as e:
            print(f"  warning: ALOHA frame {i} skipped ({e})")
    print(f"[calib] Collected {len(samples)} ALOHA fallback samples.")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# Layer-by-layer quantization
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def quantize_mixed_precision(
    model,
    processor,
    dev: torch.device,
    nsamples: int = 128,
    blocksize: int = 128,
    percdamp: float = 0.01,
    n_salient: int = 50,
    mode: str = "mixed",
    use_bridge: bool = True,
) -> dict:
    """
    Layer-by-layer mixed-precision PTQ with BridgeData V2 calibration.

    Forward pass (per calibration sample):
      vision backbone → projector → embed_tokens → [layer hook] → ...
    The hook captures input hidden states to each Linear in every decoder
    block, from which we build the per-column Hessian H = Σ x xᵀ / n.

    Quantization per layer:
      salient   (~n_salient rows per 128-col block) → 4-bit (if mode='mixed')
      non-salient                                   → 1-bit BRAQ (order 1)
    """
    import time as _time

    lm          = model.language_model
    lm.config.use_cache = False
    layers      = lm.model.layers
    hidden_size = lm.config.hidden_size
    model_dtype = next(lm.parameters()).dtype
    mode_str    = ("4-bit salient / 1-bit non-salient" if mode == "mixed"
                   else "2-bit salient / 1-bit non-salient")

    print(f"\n{'='*65}")
    print("BridgeData V2 Robot-Calibrated Quantization — OpenVLA LLM")
    print(f"  device={dev}  nsamples={nsamples}  mode={mode_str}")
    print(f"  blocksize={blocksize}  percdamp={percdamp}  n_salient={n_salient}")
    print(f"{'='*65}\n")

    # ── Step 1: collect calibration inputs ───────────────────────────────────
    if use_bridge:
        try:
            calib = collect_bridge_inputs(processor, nsamples=nsamples)
            if len(calib) < 16:
                raise RuntimeError(f"Only {len(calib)} samples collected — not enough")
        except Exception as e:
            print(f"[calib] BridgeData V2 failed ({e}), falling back to ALOHA.")
            calib = collect_aloha_inputs(processor, nsamples=nsamples)
    else:
        calib = collect_aloha_inputs(processor, nsamples=nsamples)

    nsamples = len(calib)

    # ── Step 2: run vision+projector to get LLM boundary activations ─────────
    model.vision_backbone  = model.vision_backbone.to(dev)
    model.projector        = model.projector.to(dev)
    lm.model.embed_tokens  = lm.model.embed_tokens.to(dev)
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb = lm.model.rotary_emb.to(dev)
    if hasattr(lm.model, "norm"):
        lm.model.norm = lm.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    # Probe first sample to determine multimodal sequence length
    probe = {"seqlen": None, "layer_kwargs": {}}

    class _Prober(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, x, **kwargs):
            probe["seqlen"] = x.shape[1]
            probe["layer_kwargs"].update(kwargs)
            raise ValueError

    layers[0] = _Prober(layers[0])
    try:
        model(input_ids=calib[0]["input_ids"].to(dev),
              pixel_values=calib[0]["pixel_values"].to(dev, dtype=model_dtype))
    except ValueError:
        pass
    layers[0] = layers[0].module
    seqlen    = probe["seqlen"]
    print(f"[calib] Multimodal sequence length: {seqlen} tokens\n")

    # Collect all first-layer hidden states
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=model_dtype, device=dev)
    ci   = {"n": 0}
    layer_kwargs = dict(probe["layer_kwargs"])

    class _Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, x, **kwargs):
            idx = ci["n"]
            if idx < nsamples:
                sl = min(x.shape[1], seqlen)
                inps[idx, :sl] = x[0, :sl].detach()
                layer_kwargs.update(kwargs)
                ci["n"] += 1
            raise ValueError

    layers[0] = _Catcher(layers[0])
    for s in calib:
        try:
            model(input_ids=s["input_ids"].to(dev),
                  pixel_values=s["pixel_values"].to(dev, dtype=model_dtype))
        except ValueError:
            pass
    layers[0] = layers[0].module
    print(f"[calib] Captured {ci['n']} activation samples "
          f"({ci['n'] * seqlen:,} token positions for Hessian).")

    for component in [model.vision_backbone, model.projector,
                      lm.model.embed_tokens]:
        component.cpu()
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb.cpu()
    if hasattr(lm.model, "norm"):
        lm.model.norm.cpu()
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()

    outs     = torch.zeros_like(inps)
    _SKIP    = {"past_key_values", "use_cache", "cache_position"}
    fwd_kw   = {k: v for k, v in layer_kwargs.items() if k not in _SKIP}

    n_layers    = len(layers)
    layer_times = []
    t_total     = _time.time()

    # ── Step 3: layer-by-layer quantization ──────────────────────────────────
    for li in range(n_layers):
        t0    = _time.time()
        print(f"\n[quant] Layer {li+1}/{n_layers}")
        layer  = layers[li].to(dev)
        subset = find_linear_layers(layer)

        h_accum = {
            name: HessianAccumulator(sublayer.weight.shape[1], dev)
            for name, sublayer in subset.items()
        }

        hooks = []
        for name in subset:
            def _hook(_, inp, __, _n=name):
                x = inp[0].detach()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                h_accum[_n].add_batch(x)
            hooks.append(subset[name].register_forward_hook(_hook))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kw)[0]

        for h in hooks:
            h.remove()

        for name, sublayer in subset.items():
            W  = sublayer.weight.data.float()
            H  = h_accum[name].get_hessian()
            print(f"  {name} [{W.shape[0]}×{W.shape[1]}] ...", end=" ", flush=True)

            if mode == "mixed":
                W_q, elapsed, err = mixed_precision_gptq(
                    W, H, blocksize=blocksize, percdamp=percdamp,
                    n_salient_per_col=n_salient,
                )
            else:
                W_q, elapsed, err = pure_1bit_gptq(
                    W, H, blocksize=blocksize, percdamp=percdamp,
                    n_salient_per_col=n_salient,
                )

            sublayer.weight.data = W_q.to(sublayer.weight.dtype)
            print(f"done  ({elapsed:.0f}s  err={err:.2f})")
            del H, W, W_q, h_accum[name]

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kw)[0]

        layers[li] = layer.cpu()
        del layer, subset, h_accum
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        elapsed_l  = _time.time() - t0
        layer_times.append(elapsed_l)
        print(f"  Layer {li+1} done in {elapsed_l:.1f}s")

    lm.config.use_cache = True
    total_elapsed = _time.time() - t_total
    print(f"\n[quant] All {n_layers} layers done in {total_elapsed/60:.1f} min total.")
    return {"per_layer_seconds": layer_times, "total_seconds": total_elapsed, "mode": mode}


# ══════════════════════════════════════════════════════════════════════════════
# Action accuracy evaluation
# ══════════════════════════════════════════════════════════════════════════════

def load_bridge_eval_samples(n_samples: int = 50) -> list:
    """
    Load n_samples diverse (PIL.Image, instruction_str) pairs from BridgeDataV2.

    Selects the middle frame from each of n_samples distinct robot manipulation
    episodes, guaranteeing maximum visual variety across tasks (pour, pick, wipe,
    drawer, ...).  Uses the same cached video as calibration so no extra download.
    Frames are offset from calibration indices to minimise train/eval overlap.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import imageio as iio
    from PIL import Image

    meta_path = hf_hub_download(
        repo_id=_BRIDGE_REPO, filename=_BRIDGE_META,
        repo_type="dataset", cache_dir=HF_CACHE,
    )
    df = pd.read_parquet(meta_path)
    df = df[df["index"] < _BRIDGE_NFRAMES].copy()

    tasks_path = hf_hub_download(
        repo_id=_BRIDGE_REPO, filename=_BRIDGE_TASKS,
        repo_type="dataset", cache_dir=HF_CACHE,
    )
    tasks_df = pd.read_parquet(tasks_path)
    task_map = {int(row["task_index"]): str(text)
                for text, row in tasks_df.iterrows()}

    # One middle frame per episode → maximum visual diversity
    episodes = df.groupby("episode_index", sort=True)
    ep_keys  = list(episodes.groups.keys())
    # Use a fixed but different stride than calibration (which starts at 0)
    # to avoid evaluating on the same frames used for Hessian accumulation.
    step     = max(1, len(ep_keys) // n_samples)
    selected_eps = ep_keys[step // 2 :: step][:n_samples]   # offset by step//2

    vid_path = hf_hub_download(
        repo_id=_BRIDGE_REPO, filename=_BRIDGE_VIDEO,
        repo_type="dataset", cache_dir=HF_CACHE,
    )
    reader  = iio.get_reader(vid_path, "ffmpeg")
    samples, errors = [], 0
    for ep_idx in selected_eps:
        ep_rows  = episodes.get_group(ep_idx).sort_values("index")
        mid_row  = ep_rows.iloc[len(ep_rows) // 2]   # middle of episode
        fidx     = int(mid_row["index"])
        try:
            frame = reader.get_data(fidx)
            img   = Image.fromarray(frame)
            task  = task_map.get(int(mid_row["task_index"]), "pick up the object")
            instr = f"In: What action should the robot take to {task}?\nOut:"
            samples.append((img, instr))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [eval-data] frame {fidx} skipped — {e}")
    reader.close()
    print(f"[eval] Loaded {len(samples)} BridgeDataV2 eval frames "
          f"from {len(selected_eps)} distinct episodes ({errors} errors).")
    return samples


@torch.no_grad()
def eval_action_accuracy(
    fp16_model, quant_model, processor,
    dev: torch.device,
    n_samples: int = 50,
) -> dict:
    """
    Compare quantized vs. FP16 on diverse BridgeDataV2 frames via predict_action().
    One frame per distinct manipulation episode guarantees varied visual inputs.
    Models are run sequentially to avoid OOM from two 13 GB models on GPU.
    """
    samples     = load_bridge_eval_samples(n_samples)
    DIM_NAMES   = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    model_dtype = torch.bfloat16

    def _run(m, label):
        m = m.to(dev).eval()
        acts = []
        for si, (img, instr) in enumerate(samples):
            inputs = processor(instr, img, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            pv     = inputs["pixel_values"].to(dtype=model_dtype)
            try:
                a = m.predict_action(input_ids=inputs["input_ids"],
                                     unnorm_key="bridge_orig", pixel_values=pv)
                acts.append(a)
            except Exception as e:
                print(f"  [{label}] sample {si} failed: {e}")
                acts.append(None)
            if (si + 1) % 10 == 0:
                print(f"  [{label}] {si+1}/{len(samples)}")
        m.cpu()
        torch.cuda.empty_cache()
        return acts

    print("[eval] FP16 predictions...")
    fp16_preds  = _run(fp16_model,  "fp16")
    print("[eval] Quant predictions...")
    quant_preds = _run(quant_model, "quant")

    fp16_acts, quant_acts = [], []
    for f, q in zip(fp16_preds, quant_preds):
        if f is not None and q is not None:
            fp16_acts.append(f); quant_acts.append(q)

    n = len(fp16_acts)
    if n == 0:
        print("[eval] No valid samples!")
        return {}

    fp16_arr  = np.stack(fp16_acts)
    quant_arr = np.stack(quant_acts)
    l1        = np.abs(fp16_arr - quant_arr)

    results = {
        "n_valid_samples": n,
        "mean_action_l1":  float(l1.mean()),
        "mean_action_l2":  float(np.sqrt((l1 ** 2).sum(1)).mean()),
        "mean_cosine_sim": float(np.mean([
            np.dot(fp16_acts[i], quant_acts[i]) /
            (np.linalg.norm(fp16_acts[i]) * np.linalg.norm(quant_acts[i]) + 1e-8)
            for i in range(n)
        ])),
        "per_dim_l1":      l1.mean(0).tolist(),
        "per_dim_std":     l1.std(0).tolist(),
        "per_dim_labels":  DIM_NAMES,
        "fp16_actions_sample":  fp16_arr[:5].tolist(),
        "quant_actions_sample": quant_arr[:5].tolist(),
    }

    print("\n  Per-dimension L1 error vs FP16:")
    for name, v in zip(DIM_NAMES, results["per_dim_l1"]):
        bar  = "█" * max(0, int(v * 80))
        flag = " ← ⚠" if v > 0.05 else ""
        print(f"    {name:7s}: {v:.4f}  {bar}{flag}")
    print(f"\n  Mean L1          : {results['mean_action_l1']:.4f}")
    print(f"  Cosine similarity: {results['mean_cosine_sim']:.4f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Perplexity evaluation (optional)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_perplexity(lm, tokenizer, dev, seqlen=2048) -> float:
    from datasets import load_dataset
    from torch.nn import CrossEntropyLoss

    print("[ppl] Loading C4 validation data...")
    ds = load_dataset("allenai/c4", "en", split="validation",
                      streaming=True, cache_dir=HF_CACHE)
    text_parts, total = [], 0
    for s in ds:
        text_parts.append(s["text"])
        total += len(s["text"])
        if total > seqlen * 300 * 7:
            break
    testids = tokenizer("\n\n".join(text_parts), return_tensors="pt",
                        add_special_tokens=False).input_ids
    nsamples = testids.numel() // seqlen

    lm.eval(); lm.config.use_cache = False
    layers = lm.model.layers
    dtype  = next(lm.parameters()).dtype

    lm.model.embed_tokens = lm.model.embed_tokens.to(dev)
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb = lm.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    inps = torch.zeros((nsamples, seqlen, lm.config.hidden_size), dtype=dtype, device=dev)
    kw   = {}; ci = {"n": 0}

    class _C(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, x, **kwargs):
            inps[ci["n"]] = x; ci["n"] += 1; kw.update(kwargs); raise ValueError

    layers[0] = _C(layers[0])
    for i in range(nsamples):
        try: lm(testids[:, i*seqlen:(i+1)*seqlen].to(dev))
        except ValueError: pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    lm.model.embed_tokens.cpu()
    if hasattr(lm.model, "rotary_emb"): lm.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    fwd  = {k: v for k, v in kw.items()
            if k not in {"past_key_values", "use_cache", "cache_position"}}
    for i, layer in enumerate(layers):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd)[0]
        layers[i] = layer.cpu(); del layer; torch.cuda.empty_cache()
        inps, outs = outs, inps

    if hasattr(lm.model, "norm") and lm.model.norm is not None:
        lm.model.norm = lm.model.norm.to(dev)
    lm.lm_head = lm.lm_head.to(dev)

    nlls = []
    for i in range(nsamples):
        h = inps[i].unsqueeze(0)
        if hasattr(lm.model, "norm") and lm.model.norm is not None:
            h = lm.model.norm(h)
        logits = lm.lm_head(h)
        sl  = logits[:, :-1].contiguous()
        tl  = testids[:, i*seqlen:(i+1)*seqlen][:, 1:].to(dev)
        nlls.append(CrossEntropyLoss()(sl.view(-1, sl.size(-1)), tl.view(-1)).float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    lm.lm_head.cpu()
    if hasattr(lm.model, "norm") and lm.model.norm is not None:
        lm.model.norm.cpu()
    lm.config.use_cache = True
    return ppl


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="BridgeData V2 robot-calibrated PTQ for OpenVLA")
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--lora_merged",   default=None,
                   help="Path to LoRA-merged LLM (optional)")
    p.add_argument("--mode",          choices=["mixed", "pure1bit"], default="mixed",
                   help="mixed=4-bit salient + 1-bit non-salient")
    p.add_argument("--nsamples",      type=int, default=128,
                   help="Calibration frames from BridgeData V2 (diverse episodes)")
    p.add_argument("--blocksize",     type=int, default=128)
    p.add_argument("--percdamp",      type=float, default=0.01)
    p.add_argument("--n_salient",     type=int, default=50,
                   help="Salient weight rows per 128-col GPTQ block")
    p.add_argument("--output",        default="output/openvla_bridge_calib")
    p.add_argument("--use_aloha",     action="store_true",
                   help="Use ALOHA fallback instead of BridgeData V2")
    p.add_argument("--eval_action",   action="store_true")
    p.add_argument("--n_eval",        type=int, default=50)
    p.add_argument("--eval_ppl",      action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    dev    = torch.device(args.device)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {"config": vars(args)}

    model, processor = load_openvla(lora_merged_path=args.lora_merged)

    timing = quantize_mixed_precision(
        model, processor, dev,
        nsamples=args.nsamples,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        n_salient=args.n_salient,
        mode=args.mode,
        use_bridge=not args.use_aloha,
    )
    results["quant_timing"] = timing

    save_path = outdir / "quantized_llm"
    print(f"\n[save] Saving quantized LLM → {save_path}")
    model.language_model.save_pretrained(save_path)
    processor.tokenizer.save_pretrained(save_path)

    if args.eval_ppl:
        ppl = eval_perplexity(model.language_model, processor.tokenizer, dev)
        print(f"[ppl] C4 perplexity: {ppl:.2f}")
        results["ppl"] = ppl

    if args.eval_action:
        print("\n[eval] Loading FP16 reference model...")
        fp16_model, _ = load_openvla()
        action_res = eval_action_accuracy(
            fp16_model, model, processor, dev, n_samples=args.n_eval
        )
        results["action_accuracy"] = action_res
        del fp16_model
        torch.cuda.empty_cache()

    res_path = outdir / "results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] Results → {res_path}")

    print("\n" + "="*60)
    print("  EXPERIMENT D — BridgeData V2 Robot-Calibrated 4-bit/1-bit")
    print("="*60)
    print(f"  Mode             : {timing['mode']}")
    print(f"  Calibration data : BridgeData V2 ({args.nsamples} diverse frames)")
    print(f"  Total quant time : {timing['total_seconds']/60:.1f} min")
    if "ppl" in results:
        print(f"  C4 perplexity    : {results['ppl']:.2f}")
    if "action_accuracy" in results:
        aa = results["action_accuracy"]
        c4_baseline = [0.006, 0.010, 0.011, 0.036, 0.040, 0.118, 0.552]
        exp_c_mixed = [0.005, 0.008, 0.009, 0.026, 0.031, 0.083, 0.393]
        print(f"\n  {'Dim':7s}  {'BridgeD2':>9s}  {'C4-4/1bit':>10s}  {'C4-1bit':>9s}")
        for lbl, v, c, b in zip(aa["per_dim_labels"], aa["per_dim_l1"],
                                  exp_c_mixed, c4_baseline):
            flag = " ⚠" if v > 0.05 else ""
            print(f"  {lbl:7s}  {v:9.4f}  {c:10.4f}  {b:9.4f}{flag}")
        print(f"\n  Mean L1          : {aa['mean_action_l1']:.4f}")
        print(f"  Cosine similarity: {aa['mean_cosine_sim']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
