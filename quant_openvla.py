#!/usr/bin/env python3
"""
OpenVLA 1-bit Quantization — Experiment 1 (C4-calibrated BRAQ)
===============================================================
Applies Binary Residual Approximation Quantization (BRAQ) to OpenVLA-7B's
LLM backbone (Llama transformer layers), keeping vision encoders in FP16.

Quantization is done post-training (PTQ) using C4 text calibration data.
All quantization logic lives in quantize_utils.py (self-contained, no BiLLM).
Evaluation: C4 perplexity (layer-by-layer, memory-efficient) + optional
action token accuracy (1-bit predictions vs. FP16 reference).

Usage:
  # Quantize and evaluate perplexity:
  python quant_openvla.py --device cuda:0

  # Also evaluate action token accuracy:
  python quant_openvla.py --device cuda:0 --eval_action

  # Load a previously saved quantized backbone and evaluate only:
  python quant_openvla.py --load_quantized output/openvla_1bit_llm

  # Evaluate FP16 baseline only:
  python quant_openvla.py --skip_quant
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Self-contained quantization utilities (no external BiLLM dependency)
from quantize_utils import HessianAccumulator, pure_1bit_gptq, mixed_precision_gptq  # noqa: E402


def find_layers(module: nn.Module) -> dict:
    """Return {name: layer} for all nn.Linear children of a decoder block."""
    return {name: layer for name, layer in module.named_modules()
            if isinstance(layer, nn.Linear)}

# ── Constants ──────────────────────────────────────────────────────────────
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OPENVLA_ID = "openvla/openvla-7b"
SEQLEN = 2048


# ══════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════

def load_openvla(model_id: str = OPENVLA_ID, dtype=torch.float16):
    """Load OpenVLA-7B (processor + model) from HuggingFace or local cache.

    Uses the dynamic module loader directly because transformers 5.x removed
    the AutoModelForVision2Seq alias that OpenVLA's config auto_map references.
    """
    from transformers import AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    print(f"[load] Loading processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=HF_CACHE,
    )

    print(f"[load] Resolving model class via dynamic module loader...")
    model_cls = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction",
        model_id,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )

    print(f"[load] Loading model weights from {model_id} (CPU, low_cpu_mem_usage)...")
    model = model_cls.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=HF_CACHE,
        attn_implementation="eager",   # custom class lacks _supports_sdpa (transformers 5.x)
    )
    model.eval()
    print("[load] Model loaded.")
    return model, processor


def extract_llm(model) -> nn.Module:
    """Return the LLM backbone from an OpenVLA model object."""
    # OpenVLA wraps a standard Llama/Mistral under .language_model
    if hasattr(model, "language_model"):
        return model.language_model
    raise AttributeError("Cannot find .language_model on the OpenVLA model.")


# ══════════════════════════════════════════════════════════════════════════
# Calibration + test data (C4)
# ══════════════════════════════════════════════════════════════════════════

def _load_c4_text(split: str, max_chars: int) -> str:
    from datasets import load_dataset

    ds = load_dataset(
        "allenai/c4", "en",
        split=split,
        streaming=True,
        cache_dir=HF_CACHE,
    )
    parts, total = [], 0
    for sample in ds:
        parts.append(sample["text"])
        total += len(sample["text"])
        if total >= max_chars:
            break
    return "\n\n".join(parts)


def get_calibration_data(tokenizer, nsamples: int = 128, seqlen: int = SEQLEN, seed: int = 0):
    """Return list of (input_ids, labels) pairs from C4 train split."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[calib] Loading {nsamples} calibration samples from C4...")
    text = _load_c4_text("train", max_chars=nsamples * seqlen * 7)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    tokens = enc.input_ids  # (1, N)

    n = tokens.shape[1]
    data = []
    for _ in range(nsamples):
        i = random.randint(0, n - seqlen - 1)
        inp = tokens[:, i : i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        data.append((inp, tar))
    print(f"[calib] Done: {nsamples} sequences of length {seqlen}.")
    return data


def get_c4_test(tokenizer, seqlen: int = SEQLEN):
    """Return tokenised C4 validation text for perplexity evaluation."""
    print("[eval] Loading C4 validation data...")
    text = _load_c4_text("validation", max_chars=seqlen * 300)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    print(f"[eval] Test set: {enc.input_ids.numel()} tokens → "
          f"{enc.input_ids.numel() // seqlen} complete sequences.")
    return enc


# ══════════════════════════════════════════════════════════════════════════
# 1-bit quantization (BiLLM BRAQ)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def quantize_llm_backbone(
    lm: nn.Module,
    calib_data: list,
    dev: torch.device,
    blocksize: int = 128,
    percdamp: float = 0.01,
    salient_metric: str = "hessian",
    disable_gptq: bool = False,
    mode: str = "pure1bit",   # "pure1bit" or "mixed" (salient→4bit, rest→1bit)
) -> dict:
    """
    Apply PTQ to all nn.Linear layers inside lm.model.layers (Llama decoder stack).
    mode="pure1bit" : salient→2-bit BRAQ, non-salient→1-bit BRAQ  (Experiment A)
    mode="mixed"    : salient→4-bit int,  non-salient→1-bit BRAQ  (Experiment C)

    Follows the sequential layer-by-layer approach from BiLLM:
      1. Capture hidden states at layer-0 boundary from calibration data.
      2. For each layer: run forward to collect H (Hessian), quantize weights,
         forward again with quantized weights to produce corrected activations.
      3. Offload finished layers back to CPU to keep peak GPU memory small.
    """
    print("\n" + "=" * 60)
    print("BiLLM BRAQ 1-bit Quantization — OpenVLA LLM Backbone")
    print(f"  device={dev}  blocksize={blocksize}  metric={salient_metric}")
    print("=" * 60)

    lm.config.use_cache = False
    layers = lm.model.layers
    nsamples = len(calib_data)
    seqlen = calib_data[0][0].shape[1]
    hidden_size = lm.config.hidden_size
    dtype = next(lm.parameters()).dtype

    # Move embedding + rotary_emb + first layer to device.
    # transformers ≥5.x computes position_embeddings (RoPE cos/sin) at the
    # model level and passes them as kwargs to every decoder layer, so
    # rotary_emb must be on the same device as embed_tokens.
    lm.model.embed_tokens = lm.model.embed_tokens.to(dev)
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb = lm.model.rotary_emb.to(dev)
    if hasattr(lm.model, "norm"):
        lm.model.norm = lm.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=dev)
    # Capture ALL layer kwargs (attention_mask, position_embeddings, cache_position, …)
    # so the code is robust to any transformers version.
    layer_kwargs: dict = {}
    cache = {"i": 0}

    class _Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            layer_kwargs.update(kwargs)   # grab everything the model passes
            raise ValueError              # abort forward after capturing

    layers[0] = _Catcher(layers[0])
    for inp_ids, _ in calib_data:
        try:
            lm(inp_ids.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # Offload embeddings / rotary_emb after capture
    layers[0] = layers[0].cpu()
    lm.model.embed_tokens = lm.model.embed_tokens.cpu()
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb = lm.model.rotary_emb.cpu()
    if hasattr(lm.model, "norm"):
        lm.model.norm = lm.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    # Pass all captured kwargs except KV-cache objects; force use_cache=False.
    _SKIP = {"past_key_values", "use_cache", "cache_position"}
    fwd_kwargs = {k: v for k, v in layer_kwargs.items() if k not in _SKIP}

    n_layers = len(layers)
    layer_times = []
    t_total_start = time.time()

    for i in range(n_layers):
        t0 = time.time()
        print(f"\n[quant] Layer {i+1}/{n_layers}")
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        # Build Hessian accumulators for every Linear in this layer
        h_accum = {
            name: HessianAccumulator(subset[name].weight.shape[1], dev)
            for name in subset
        }

        # Register hooks to collect activation statistics (Hessian)
        handles = []
        for name in subset:
            def _hook(_, inp, out, _n=name):
                x = inp[0].detach()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                h_accum[_n].add_batch(x)
            handles.append(subset[name].register_forward_hook(_hook))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kwargs)[0]

        for h in handles:
            h.remove()

        # Quantize each sub-layer
        quant_fn = mixed_precision_gptq if mode == "mixed" else pure_1bit_gptq
        for name in subset:
            print(f"  quantizing {name} ...", end=" ", flush=True)
            W = subset[name].weight.data.float()
            H = h_accum[name].get_hessian()
            W_q, elapsed, err = quant_fn(W, H, blocksize=blocksize, percdamp=percdamp)
            subset[name].weight.data = W_q.to(subset[name].weight.dtype)
            del H, W, W_q, h_accum[name]
            print(f"done  ({elapsed:.0f}s  err={err:.2f})")

        # Re-run with binarized weights to propagate corrected activations
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kwargs)[0]

        layers[i] = layer.cpu()
        del layer, subset, h_accum
        torch.cuda.empty_cache()

        inps, outs = outs, inps  # next layer reads from this layer's outputs
        elapsed = time.time() - t0
        layer_times.append(elapsed)
        print(f"  Layer {i+1} done in {elapsed:.1f}s")

    lm.config.use_cache = True
    total = time.time() - t_total_start
    print(f"\n[quant] All {n_layers} layers quantized in {total:.1f}s total.\n")
    return {"per_layer_seconds": layer_times, "total_seconds": total}


# ══════════════════════════════════════════════════════════════════════════
# Perplexity evaluation (memory-efficient, layer-by-layer)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_perplexity(lm: nn.Module, testenc, dev: torch.device, seqlen: int = SEQLEN) -> float:
    """
    Compute C4 perplexity on the LLM backbone using the same layer-by-layer
    technique as BiLLM's llama_eval — only one decoder layer lives on GPU
    at a time.
    """
    print("[ppl] Computing perplexity (layer-by-layer, memory-efficient)...")
    lm.eval()
    lm.config.use_cache = False

    testids = testenc.input_ids
    nsamples = testids.numel() // seqlen
    layers = lm.model.layers
    hidden_size = lm.config.hidden_size
    dtype = next(lm.parameters()).dtype

    lm.model.embed_tokens = lm.model.embed_tokens.to(dev)
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb = lm.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=dev)
    layer_kwargs: dict = {}
    cache = {"i": 0}

    class _Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            layer_kwargs.update(kwargs)
            raise ValueError

    layers[0] = _Catcher(layers[0])
    for i in range(nsamples):
        batch = testids[:, i * seqlen : (i + 1) * seqlen].to(dev)
        try:
            lm(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    lm.model.embed_tokens = lm.model.embed_tokens.cpu()
    if hasattr(lm.model, "rotary_emb"):
        lm.model.rotary_emb = lm.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    _SKIP = {"past_key_values", "use_cache", "cache_position"}
    fwd_kw = {k: v for k, v in layer_kwargs.items() if k not in _SKIP}

    for i, _ in enumerate(layers):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **fwd_kw)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        if (i + 1) % 8 == 0 or (i + 1) == len(layers):
            print(f"  [ppl] {i+1}/{len(layers)} layers done")

    # Final norm + lm_head on GPU
    if hasattr(lm.model, "norm") and lm.model.norm is not None:
        lm.model.norm = lm.model.norm.to(dev)
    lm.lm_head = lm.lm_head.to(dev)

    testids_dev = testids.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden = inps[i].unsqueeze(0)
        if hasattr(lm.model, "norm") and lm.model.norm is not None:
            hidden = lm.model.norm(hidden)
        logits = lm.lm_head(hidden)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = testids_dev[:, i * seqlen : (i + 1) * seqlen][:, 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()

    # Offload back to CPU
    if hasattr(lm.model, "norm") and lm.model.norm is not None:
        lm.model.norm = lm.model.norm.cpu()
    lm.lm_head = lm.lm_head.cpu()
    torch.cuda.empty_cache()

    lm.config.use_cache = True
    return ppl


# ══════════════════════════════════════════════════════════════════════════
# Action token accuracy evaluation
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_action_accuracy(
    fp16_model,
    quant_model,
    processor,
    dev: torch.device,
    n_samples: int = 20,
    seed: int = 42,
) -> dict:
    """
    Compare 1-bit model action predictions against FP16 model predictions
    on synthetic robot workspace images.

    Metrics reported:
      - overall_action_token_accuracy : % of the 7 action dimensions where
                                        the 1-bit model matches FP16 exactly.
      - per_dim_accuracy              : accuracy per joint/dimension.
      - overall_action_l1_error       : mean absolute token-index difference
                                        (scale 0–255, since OpenVLA discretises
                                        each action dim into 256 bins).
      - per_dim_l1_error              : per-dimension mean absolute error.
    """
    from PIL import Image

    rng = np.random.RandomState(seed)

    instruction_pool = [
        "In: What action should the robot take to pick up the red block?\nOut:",
        "In: Move the gripper to grasp the blue cube on the table.\nOut:",
        "In: What action should the robot take to push the object to the target?\nOut:",
        "In: Open the drawer using the robot arm.\nOut:",
        "In: What action should the robot take to stack the blocks?\nOut:",
        "In: Grasp the yellow object and move it to the right.\nOut:",
        "In: What action should the robot take to pour water into the cup?\nOut:",
    ]

    fp16_model = fp16_model.to(dev).eval()
    quant_model = quant_model.to(dev).eval()

    all_matches, all_l1 = [], []
    print(f"\n[action-acc] Evaluating {n_samples} samples...")

    for s in range(n_samples):
        prompt = instruction_pool[s % len(instruction_pool)]

        # Synthetic 224×224 robot scene: coloured background + random circular object
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :, 1] = 60   # greenish table background
        cx, cy = rng.randint(60, 164, size=2)
        r = rng.randint(18, 38)
        color = rng.randint(80, 255, 3).tolist()
        ys, xs = np.ogrid[:224, :224]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
        img[mask] = color
        image = Image.fromarray(img)

        inputs = processor(prompt, image, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        fp16_out = fp16_model.generate(
            **inputs, max_new_tokens=7, do_sample=False
        )
        fp16_tok = fp16_out[0, -7:].cpu()

        quant_out = quant_model.generate(
            **inputs, max_new_tokens=7, do_sample=False
        )
        quant_tok = quant_out[0, -7:].cpu()

        all_matches.append((fp16_tok == quant_tok).float())
        all_l1.append((fp16_tok.float() - quant_tok.float()).abs())

        if (s + 1) % 5 == 0:
            print(f"  {s+1}/{n_samples} samples done")

    fp16_model.cpu()
    quant_model.cpu()
    torch.cuda.empty_cache()

    matches = torch.stack(all_matches)   # (N, 7)
    l1_err = torch.stack(all_l1)        # (N, 7)

    return {
        "overall_action_token_accuracy": matches.mean().item(),
        "per_dim_accuracy": matches.mean(dim=0).tolist(),
        "overall_action_l1_error": l1_err.mean().item(),
        "per_dim_l1_error": l1_err.mean(dim=0).tolist(),
        "n_samples": n_samples,
    }


# ══════════════════════════════════════════════════════════════════════════
# GPU / memory helpers
# ══════════════════════════════════════════════════════════════════════════

def _gpu_free_gb(dev_idx: int) -> float:
    props = torch.cuda.mem_get_info(dev_idx)
    return props[0] / 1e9  # free bytes → GB


def pick_device(requested: str) -> torch.device:
    if not torch.cuda.is_available():
        print("[device] No CUDA found — using CPU (will be slow).")
        return torch.device("cpu")
    dev_idx = int(requested.split(":")[-1]) if ":" in requested else 0
    free_gb = _gpu_free_gb(dev_idx)
    print(f"[device] {requested}: {free_gb:.1f} GB free")
    if free_gb < 1.5:
        # Try another GPU
        for i in range(torch.cuda.device_count()):
            f = _gpu_free_gb(i)
            if f >= 1.5:
                print(f"[device] Switching to cuda:{i} ({f:.1f} GB free).")
                return torch.device(f"cuda:{i}")
        print("[device] Warning: all GPUs have <1.5 GB free — using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA 1-bit PTQ (BiLLM BRAQ)")
    p.add_argument("--device", default="cuda:0",
                   help="CUDA device for quantization/eval (default: cuda:0)")
    p.add_argument("--blocksize", type=int, default=128,
                   help="GPTQ block size / group size for binarization")
    p.add_argument("--percdamp", type=float, default=0.01,
                   help="GPTQ Hessian dampening fraction")
    p.add_argument("--nsamples", type=int, default=128,
                   help="Number of C4 calibration sequences")
    p.add_argument("--seqlen", type=int, default=2048,
                   help="Sequence length for calibration")
    p.add_argument("--salient_metric", choices=["hessian", "magnitude"],
                   default="hessian",
                   help="Metric to identify salient (outlier) weights")
    p.add_argument("--mode", choices=["pure1bit", "mixed"], default="pure1bit",
                   help="pure1bit: salient→2-bit BRAQ (Exp A)  "
                        "mixed: salient→4-bit int (Exp C)")
    p.add_argument("--disable_gptq", action="store_true",
                   help="Skip GPTQ error-correction step (faster, lower quality)")
    p.add_argument("--save", default="output/openvla_1bit_llm",
                   help="Directory to save the quantized LLM backbone")
    p.add_argument("--load_quantized", default=None,
                   help="Load a previously saved quantized LLM backbone "
                        "(skips re-quantization)")
    p.add_argument("--skip_quant", action="store_true",
                   help="Skip quantization; evaluate FP16 baseline only")
    p.add_argument("--eval_action", action="store_true",
                   help="Also evaluate action token accuracy (needs extra GPU RAM)")
    p.add_argument("--n_action_samples", type=int, default=20,
                   help="Number of synthetic action samples for accuracy eval")
    return p.parse_args()


def print_summary(results: dict):
    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    fp16_p = results.get('fp16_ppl', 'N/A')
    fp16_str = f"{fp16_p:.3f}" if isinstance(fp16_p, float) else str(fp16_p)
    print(f"  FP16 C4 perplexity     : {fp16_str}")
    print(f"  (Note: high absolute PPL expected — OpenVLA LLM is action-tuned,")
    print(f"   causing catastrophic text-forgetting. Relative ratio is meaningful.)")
    if "quant_1bit_ppl" in results:
        print(f"  1-bit C4 perplexity    : {results['quant_1bit_ppl']:.3f}")
        print(f"  PPL degradation (abs)  : +{results['ppl_degradation']:.3f}")
        print(f"  PPL degradation (rel)  : +{results.get('ppl_relative_increase_pct', 0):.2f}%")
    if "action_accuracy" in results:
        aa = results["action_accuracy"]
        print(f"  Action token accuracy  : {aa['overall_action_token_accuracy']*100:.1f}%  "
              f"(vs. FP16 reference)")
        print(f"  Mean L1 action error   : {aa['overall_action_l1_error']:.2f} / 255 tokens")
        dims = [f"{x*100:.0f}%" for x in aa["per_dim_accuracy"]]
        print(f"  Per-dim accuracy       : {dims}")
    print("=" * 55)


def main():
    args = parse_args()
    dev = pick_device(args.device)

    save_dir = Path(args.save)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir.parent, exist_ok=True)

    results: dict = {}

    # ── Load OpenVLA ───────────────────────────────────────────────────────
    model, processor = load_openvla()
    tokenizer = processor.tokenizer
    lm = extract_llm(model)

    # ── Perplexity baseline (FP16) ─────────────────────────────────────────
    test_enc = get_c4_test(tokenizer, seqlen=args.seqlen)
    print("\n--- FP16 Baseline ---")
    fp16_ppl = eval_perplexity(lm, test_enc, dev, seqlen=args.seqlen)
    print(f"FP16 C4 Perplexity: {fp16_ppl:.3f}")
    results["fp16_ppl"] = fp16_ppl

    if args.skip_quant:
        results_path = save_dir.parent / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print_summary(results)
        print(f"\nResults saved to {results_path}")
        return

    # ── Quantize (or load previously quantized) ────────────────────────────
    if args.load_quantized:
        print(f"\n[load] Loading quantized LLM from {args.load_quantized}...")
        from transformers import AutoModelForCausalLM
        quant_lm = AutoModelForCausalLM.from_pretrained(
            args.load_quantized,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.language_model = quant_lm
        lm = quant_lm
    else:
        calib_data = get_calibration_data(
            tokenizer,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
        )
        timing = quantize_llm_backbone(
            lm, calib_data, dev,
            blocksize=args.blocksize,
            percdamp=args.percdamp,
            salient_metric=args.salient_metric,
            disable_gptq=args.disable_gptq,
            mode=args.mode,
        )
        results["quant_timing"] = timing

        # Save quantized LLM backbone (weights stored as FP16 with binary values)
        print(f"\n[save] Saving quantized LLM backbone to {save_dir}...")
        lm.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("[save] Done.")

    # ── Perplexity after 1-bit quantization ────────────────────────────────
    print("\n--- 1-bit Quantized ---")
    quant_ppl = eval_perplexity(lm, test_enc, dev, seqlen=args.seqlen)
    print(f"1-bit C4 Perplexity: {quant_ppl:.3f}")
    results["quant_1bit_ppl"] = quant_ppl
    results["ppl_degradation"] = quant_ppl - fp16_ppl
    # Relative PPL increase from quantization (meaningful even when absolute PPLs are high
    # due to catastrophic forgetting of text in OpenVLA's action-tuned LLM backbone)
    results["ppl_relative_increase_pct"] = 100.0 * (quant_ppl - fp16_ppl) / max(fp16_ppl, 1e-6)

    # ── Optional: action token accuracy ───────────────────────────────────
    if args.eval_action:
        print("\n--- Action Token Accuracy ---")
        print("[action-acc] Loading fresh FP16 model as reference...")
        fp16_ref, _ = load_openvla()
        action_results = eval_action_accuracy(
            fp16_ref, model, processor, dev,
            n_samples=args.n_action_samples,
        )
        results["action_accuracy"] = action_results
        del fp16_ref
        torch.cuda.empty_cache()
        print(f"Action token accuracy : "
              f"{action_results['overall_action_token_accuracy']*100:.1f}%")

    # ── Save results ───────────────────────────────────────────────────────
    results_path = save_dir.parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] Results written to {results_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
