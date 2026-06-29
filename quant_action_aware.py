#!/usr/bin/env python3
"""
Experiment E: Action-Aware Mixed 4/1-bit Quantization
======================================================
For layers identified as action-critical (high Hessian divergence between
all-token and action-token calibration), this script uses H_action — built
from ONLY the 7 action-token positions — to decide WHICH weights to protect
in 4-bit (salience detection).  The actual GPTQ weight updates still use
H_all (all 283 tokens) because it is better conditioned.

For non-critical layers, standard calibration (H_all for both) is used.

Critical layers (0-indexed): loaded from
  output/action_layer_analysis/action_critical_layers.txt

Usage:
  python quant_action_aware.py --device cuda:0 --nsamples 128

Output:
  output/openvla_exp_e_action_aware/  — quantized model weights
  logs/exp_e_action_aware.log
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from quantize_utils import HessianAccumulator, mixed_precision_gptq

HF_CACHE      = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OPENVLA_ID    = "openvla/openvla-7b"
UNNORM_KEY    = "bridge_orig"
_BRIDGE_REPO  = "nvidia/BridgeData2_LeRobot_v3"
_BRIDGE_VIDEO = "videos/observation.images.image_0/chunk-000/file-000.mp4"
_BRIDGE_META  = "data/chunk-000/file-000.parquet"
_BRIDGE_TASKS = "meta/tasks.parquet"
_BRIDGE_NFRAMES = 12282
CRITICAL_LAYERS_FILE = "output/action_layer_analysis/action_critical_layers.txt"


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def get_norm_stats(model):
    ns = getattr(model, "norm_stats", {})
    if UNNORM_KEY not in ns:
        raise RuntimeError(f"norm_stats['{UNNORM_KEY}'] not found")
    entry = ns[UNNORM_KEY]
    return entry.get("action", entry)


def action_to_token_ids(action, norm_stats):
    q01 = np.asarray(norm_stats["q01"], dtype=np.float32)
    q99 = np.asarray(norm_stats["q99"], dtype=np.float32)
    a   = np.asarray(action, dtype=np.float32)
    norm = np.clip(2.0 * (a - q01) / (q99 - q01 + 1e-8) - 1.0, -1.0, 1.0)
    bins = np.clip(np.round((norm + 1.0) / 2.0 * 255.0).astype(np.int32), 0, 255)
    return (31745 + bins).tolist()


def collect_bridge_samples(processor, norm_stats, nsamples=128):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import imageio as iio
    from PIL import Image

    meta_path  = hf_hub_download(repo_id=_BRIDGE_REPO, filename=_BRIDGE_META,
                                  repo_type="dataset", cache_dir=HF_CACHE)
    tasks_path = hf_hub_download(repo_id=_BRIDGE_REPO, filename=_BRIDGE_TASKS,
                                  repo_type="dataset", cache_dir=HF_CACHE)
    df       = pd.read_parquet(meta_path)
    df       = df[df["index"] < _BRIDGE_NFRAMES].copy()
    tasks_df = pd.read_parquet(tasks_path)
    task_map = {int(row["task_index"]): str(text)
                for text, row in tasks_df.iterrows()}

    stride     = max(1, _BRIDGE_NFRAMES // nsamples)
    frame_idxs = list(range(0, _BRIDGE_NFRAMES, stride))[:nsamples]
    selected   = df[df["index"].isin(frame_idxs)].drop_duplicates("index").sort_values("index")

    vid_path = hf_hub_download(repo_id=_BRIDGE_REPO, filename=_BRIDGE_VIDEO,
                                repo_type="dataset", cache_dir=HF_CACHE)
    reader   = iio.get_reader(vid_path, "ffmpeg")
    samples  = []
    for _, row in selected.iterrows():
        try:
            fidx  = int(row["index"])
            frame = reader.get_data(fidx)
            img   = Image.fromarray(frame)
            task  = task_map.get(int(row["task_index"]), "pick up the object")
            instr = f"In: What action should the robot take to {task}?\nOut:"
            inp   = processor(instr, img, return_tensors="pt")
            aids  = action_to_token_ids(list(row["action"]), norm_stats)
            samples.append({"inputs": {k: v for k, v in inp.items()},
                            "action_token_ids": aids})
        except Exception as e:
            print(f"  warning: frame {fidx} skipped — {e}")
    reader.close()
    print(f"[data] Loaded {len(samples)} samples with action tokens.")
    return samples


# ── Quantization ───────────────────────────────────────────────────────────────

@torch.no_grad()
def quantize(model, processor, dev, nsamples=128, n_salient=50,
             critical_layers=None, outdir=None):
    norm_stats = get_norm_stats(model)
    samples    = collect_bridge_samples(processor, norm_stats, nsamples)
    nsamples   = len(samples)

    lm          = model.language_model
    lm.config.use_cache = False
    layers      = lm.model.layers
    hidden_size = lm.config.hidden_size
    model_dtype = next(lm.parameters()).dtype

    # ── Collect first-layer inputs (with action tokens appended) ─────────────
    for comp in [model.vision_backbone, model.projector, lm.model.embed_tokens]:
        comp.to(dev)
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb.to(dev)
    layers[0].to(dev)

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

    seqlen       = probe["seqlen"]
    ext_seqlen   = seqlen + 7
    action_start = seqlen
    print(f"[quant] Sequence: {seqlen} → {ext_seqlen} tokens (action at [{action_start}:{action_start+7}])")

    # Capture extended hidden states
    all_inps = torch.zeros((nsamples, ext_seqlen, hidden_size),
                           dtype=model_dtype, device="cpu")
    ci           = {"n": 0}
    layer_kwargs = dict(probe["kwargs"])

    class _Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, x, **kwargs):
            idx = ci["n"]
            if idx < nsamples:
                aids    = torch.tensor(samples[idx]["action_token_ids"],
                                       dtype=torch.long, device=dev).unsqueeze(0)
                act_emb = lm.model.embed_tokens(aids).to(x.dtype)
                x_ext   = torch.cat([x, act_emb], dim=1)
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

    # Compute position embeddings for extended sequence while rotary_emb is on GPU
    pos_ids_ext = torch.arange(ext_seqlen, dtype=torch.long, device=dev).unsqueeze(0)
    if hasattr(lm.model, "rotary_emb"):
        _dummy    = torch.zeros(1, ext_seqlen, hidden_size, dtype=model_dtype, device=dev)
        pos_emb_ext = lm.model.rotary_emb(_dummy, pos_ids_ext)
    else:
        pos_emb_ext = None

    for comp in [model.vision_backbone, model.projector, lm.model.embed_tokens]:
        comp.cpu()
    if hasattr(lm.model, "rotary_emb"): lm.model.rotary_emb.cpu()
    layers[0].cpu()
    torch.cuda.empty_cache()
    print(f"[quant] Captured {ci['n']} extended calibration samples.")

    inps = all_inps.to(dev)
    outs = torch.zeros_like(inps)

    _SKIP  = {"past_key_values", "use_cache", "cache_position",
              "position_embeddings", "attention_mask", "position_ids"}
    fwd_kw = {k: v for k, v in layer_kwargs.items() if k not in _SKIP}
    fwd_kw["position_ids"] = pos_ids_ext
    if pos_emb_ext is not None:
        fwd_kw["position_embeddings"] = pos_emb_ext
    _causal = torch.zeros(1, 1, ext_seqlen, ext_seqlen, dtype=model_dtype, device=dev)
    _causal.masked_fill_(torch.triu(torch.ones(ext_seqlen, ext_seqlen,
                                               dtype=torch.bool, device=dev), diagonal=1),
                         float("-inf"))
    fwd_kw["attention_mask"] = _causal

    # ── Layer-by-layer quantization ───────────────────────────────────────────
    n_critical, n_standard = 0, 0

    for li in range(len(layers)):
        is_critical = (li in critical_layers)
        mode_tag    = "ACTION-AWARE" if is_critical else "standard  "
        print(f"\n[layer {li+1:2d}/{len(layers)}] {mode_tag}")

        layer  = layers[li].to(dev)
        subset = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}

        # Build Hessian accumulators — dual for critical layers, single for others
        h_all = {n: HessianAccumulator(m.weight.shape[1], dev) for n, m in subset.items()}
        h_act = {n: HessianAccumulator(m.weight.shape[1], dev) for n, m in subset.items()} \
                if is_critical else None

        hooks = []
        for name, sublayer in subset.items():
            def _hook(_, inp, __, _n=name):
                x = inp[0].detach()
                h_all[_n].add_batch(x)
                if h_act is not None:
                    x_a = x[:, action_start:, :]
                    if x_a.shape[1] > 0:
                        h_act[_n].add_batch(x_a)
            hooks.append(sublayer.register_forward_hook(_hook))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kw)[0]

        for h in hooks: h.remove()

        # Quantize each sublayer
        for name, sublayer in subset.items():
            W  = sublayer.weight.data.float()
            Ha = h_all[name].get_hessian()
            Hs = h_act[name].get_hessian() if is_critical else None

            t0 = time.time()
            W_q, elapsed, loss = mixed_precision_gptq(
                W, Ha,
                n_salient_per_col=n_salient,
                H_salience=Hs,   # None for non-critical → standard behaviour
            )
            sublayer.weight.data = W_q.to(model_dtype)
            tag = "Hs=H_action" if is_critical else "Hs=H_all  "
            print(f"  {name:20s} [{tag}]  loss={loss:.4f}  {elapsed:.1f}s")

        if is_critical: n_critical += 1
        else:           n_standard += 1

        layers[li] = layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    print(f"\n[quant] Done — {n_critical} action-aware layers, {n_standard} standard layers.")
    lm.config.use_cache = True

    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        lm.save_pretrained(str(outdir))
        print(f"[quant] LLM saved → {outdir}")

    return model


# ── Evaluation — reuse Exp D's eval_action_accuracy ───────────────────────────

def run_evaluation(quant_model, processor, dev, n_eval=50, outdir=None):
    import json
    from quant_robot_calibrated import eval_action_accuracy, load_openvla as load_fp16
    print("[eval] Loading fresh FP16 model for reference...")
    fp16_model, _ = load_fp16()
    results = eval_action_accuracy(fp16_model, quant_model, processor, dev, n_samples=n_eval)
    del fp16_model
    torch.cuda.empty_cache()
    if outdir and results:
        p = Path(outdir) / "eval_results.json"
        with open(p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] Results saved → {p}")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device",   default="cuda:0")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--n_salient", type=int, default=50)
    p.add_argument("--critical_layers_file", default=CRITICAL_LAYERS_FILE)
    p.add_argument("--outdir", default="output/openvla_exp_e_action_aware")
    return p.parse_args()


def main():
    args = parse_args()
    dev  = torch.device(args.device)

    # Load action-critical layer indices from analysis
    crit_path = Path(args.critical_layers_file)
    if crit_path.exists():
        critical_layers = set(int(l.strip()) for l in crit_path.read_text().splitlines() if l.strip())
        print(f"[exp-e] Action-critical layers (0-indexed): {sorted(critical_layers)}")
    else:
        # Fallback to hardcoded result from the analysis run
        critical_layers = {8, 9, 11, 12, 13, 14, 15, 16, 31}
        print(f"[exp-e] Using hardcoded critical layers: {sorted(critical_layers)}")

    print("[load] Loading OpenVLA...")
    model, processor = load_openvla()

    model = quantize(model, processor, dev,
                     nsamples=args.nsamples,
                     n_salient=args.n_salient,
                     critical_layers=critical_layers,
                     outdir=args.outdir)

    print("\n[eval] Running evaluation...")
    run_evaluation(model, processor, dev, n_eval=50, outdir=args.outdir)


if __name__ == "__main__":
    main()
