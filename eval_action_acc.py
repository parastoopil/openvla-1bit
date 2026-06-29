#!/usr/bin/env python3
"""
OpenVLA Action Accuracy — FP16 vs 1-bit Quantized
===================================================
Uses predict_action() (OpenVLA's native interface) to compare continuous
7-DoF robot action predictions between the FP16 model and the 1-bit
quantized variant.

Real robot images are loaded from the BridgeDataV2 subset on HuggingFace
(openvla/modified_bridge_orig). Falls back to synthetic images if the
dataset is unavailable.

Metrics (all compare 1-bit against FP16 oracle):
  action_l1            : mean |fp16_action - quant_action| per dimension
  action_l2            : mean Euclidean distance between action vectors
  cosine_similarity    : mean cosine similarity of action vectors
  per_dim_l1           : per-joint mean absolute error
  max_joint_error      : worst-case joint error across all samples
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OPENVLA_ID = "openvla/openvla-7b"
ACTION_DIMS = 7
UNNORM_KEY = "bridge_orig"

DIM_LABELS = ["x (m)", "y (m)", "z (m)", "roll (rad)", "pitch (rad)", "yaw (rad)", "gripper"]


# ══════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════

def _load_base(dtype=torch.float16):
    from transformers import AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    processor = AutoProcessor.from_pretrained(
        OPENVLA_ID, trust_remote_code=True, cache_dir=HF_CACHE
    )
    model_cls = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction",
        OPENVLA_ID, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    return processor, model_cls


def load_fp16_model():
    processor, model_cls = _load_base()
    # Use bfloat16 — the LLM backbone checkpoint is bfloat16; loading as float16
    # causes silent dtype mismatch between vision backbone and LLM embeddings.
    model = model_cls.from_pretrained(
        OPENVLA_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, cache_dir=HF_CACHE,
        attn_implementation="eager",
    )
    return model.eval(), processor


def load_quant_model(quant_llm_path: str):
    from transformers import AutoModelForCausalLM
    processor, model_cls = _load_base()
    model = model_cls.from_pretrained(
        OPENVLA_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, cache_dir=HF_CACHE,
        attn_implementation="eager",
    )
    print(f"[load] Swapping in 1-bit LLM backbone from {quant_llm_path}...")
    quant_lm = AutoModelForCausalLM.from_pretrained(
        quant_llm_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.language_model = quant_lm
    return model.eval(), processor


# ══════════════════════════════════════════════════════════════════════════
# Evaluation data
# ══════════════════════════════════════════════════════════════════════════

def load_robot_samples(n: int, seed: int = 42):
    """
    Load n BridgeDataV2 frames for evaluation.
    Returns list of (PIL.Image, instruction_str) pairs, one per distinct episode,
    guaranteeing visual diversity across tasks (pour, pick, wipe, drawer, ...).
    Falls back to synthetic images if the dataset is unavailable.
    """
    try:
        from quant_robot_calibrated import load_bridge_eval_samples
        return load_bridge_eval_samples(n)
    except Exception as e:
        print(f"[data] BridgeDataV2 load failed ({e}); falling back to synthetic.")
        return _synthetic_samples(n, seed=seed)


def _synthetic_samples(n: int, seed: int = 99):
    """Fallback: synthetic robot-scene images with realistic-looking structure."""
    from PIL import Image, ImageDraw

    rng = np.random.RandomState(seed)
    instructions = [
        "pick up the red block and place it in the bowl",
        "move the robot arm to grasp the blue object",
        "push the yellow cube to the right side of the table",
        "lift the green bottle off the surface",
        "slide the small object toward the target zone",
    ]
    samples = []
    for i in range(n):
        # More realistic: grey tabletop, arm silhouette, coloured object
        img_arr = np.full((224, 224, 3), [120, 110, 100], dtype=np.uint8)
        # table surface
        img_arr[140:, :] = [180, 165, 150]
        # robot arm (dark rectangle)
        img_arr[60:140, 100:120] = [50, 50, 60]
        # object
        cx, cy = rng.randint(60, 160), rng.randint(150, 200)
        r = rng.randint(10, 20)
        color = rng.randint(80, 220, 3).tolist()
        ys, xs = np.ogrid[:224, :224]
        img_arr[(xs - cx)**2 + (ys - cy)**2 <= r**2] = color
        image = Image.fromarray(img_arr)
        instr = instructions[i % len(instructions)]
        samples.append((image, instr))
    return samples


# ══════════════════════════════════════════════════════════════════════════
# Core evaluation
# ══════════════════════════════════════════════════════════════════════════

def format_prompt(instruction: str) -> str:
    """Format instruction into the OpenVLA prompt template."""
    return f"In: What action should the robot take to {instruction.lower().rstrip('.')}?\nOut:"


@torch.no_grad()
def run_eval(fp16_model, quant_model, processor, samples, dev) -> dict:
    fp16_model = fp16_model.to(dev).eval()
    quant_model = quant_model.to(dev).eval()

    fp16_actions, quant_actions = [], []
    print(f"\n[eval] Running predict_action on {len(samples)} samples...")

    for i, (image, instruction) in enumerate(samples):
        prompt = format_prompt(instruction)
        inputs = processor(prompt, image, return_tensors="pt")
        # Move all inputs to device (input_ids + pixel_values)
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        fp16_act = fp16_model.predict_action(
            **inputs, unnorm_key=UNNORM_KEY, do_sample=False
        )
        quant_act = quant_model.predict_action(
            **inputs, unnorm_key=UNNORM_KEY, do_sample=False
        )

        fp16_actions.append(fp16_act)
        quant_actions.append(quant_act)

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            # Running L1
            diffs = np.abs(np.array(fp16_actions) - np.array(quant_actions))
            print(f"  [{i+1}/{len(samples)}]  running mean L1: {diffs.mean():.5f}")

    fp16_model.cpu()
    quant_model.cpu()
    torch.cuda.empty_cache()

    fp16_arr  = np.array(fp16_actions)   # (N, 7)
    quant_arr = np.array(quant_actions)  # (N, 7)

    diff = fp16_arr - quant_arr
    l1   = np.abs(diff)
    l2   = np.linalg.norm(diff, axis=1)

    # Cosine similarity per sample
    cos_sims = []
    for f, q in zip(fp16_arr, quant_arr):
        denom = (np.linalg.norm(f) * np.linalg.norm(q)) + 1e-9
        cos_sims.append(float(np.dot(f, q) / denom))

    return {
        "n_samples": len(samples),
        "mean_action_l1": float(l1.mean()),
        "mean_action_l2": float(l2.mean()),
        "mean_cosine_similarity": float(np.mean(cos_sims)),
        "per_dim_l1": l1.mean(axis=0).tolist(),
        "per_dim_std": l1.std(axis=0).tolist(),
        "max_joint_error": float(l1.max()),
        "exact_match_rate": float(np.all(diff == 0, axis=1).mean()),
        "fp16_actions_sample": fp16_arr[:5].tolist(),
        "quant_actions_sample": quant_arr[:5].tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════
# Pretty print
# ══════════════════════════════════════════════════════════════════════════

def print_results(results: dict):
    print(f"\n{'='*62}")
    print(f"  OpenVLA Action Accuracy  [1-bit quantized vs FP16 oracle]")
    print(f"{'='*62}")
    print(f"  Samples evaluated        : {results['n_samples']}")
    print(f"  Mean action L1 error     : {results['mean_action_l1']:.5f}  (continuous, mixed units)")
    print(f"  Mean action L2 error     : {results['mean_action_l2']:.5f}")
    print(f"  Mean cosine similarity   : {results['mean_cosine_similarity']:.4f}  (1.0 = identical)")
    print(f"  Exact match rate         : {results['exact_match_rate']*100:.1f}%")
    print(f"  Max single-joint error   : {results['max_joint_error']:.5f}")
    print()
    print(f"  Per-dimension L1 errors:")
    print(f"  {'Joint':>10}  {'Mean L1':>10}  {'Std':>8}")
    print(f"  {'-'*34}")
    for label, l1, std in zip(DIM_LABELS, results['per_dim_l1'], results['per_dim_std']):
        print(f"  {label:>10}  {l1:>10.5f}  {std:>8.5f}")
    print()
    print(f"  FP16 sample actions (first 3):")
    for row in results['fp16_actions_sample'][:3]:
        print(f"    {[f'{v:.4f}' for v in row]}")
    print(f"  1-bit sample actions (first 3):")
    for row in results['quant_actions_sample'][:3]:
        print(f"    {[f'{v:.4f}' for v in row]}")
    print(f"{'='*62}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quant_llm", default="output/openvla_1bit_llm")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_synthetic", action="store_true",
                   help="Skip HuggingFace download; use synthetic images only")
    p.add_argument("--output", default="output/action_accuracy.json")
    return p.parse_args()


def main():
    args = parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load evaluation data
    if args.use_synthetic:
        samples = _synthetic_samples(args.n_samples, seed=args.seed)
        print(f"[data] Using {len(samples)} synthetic images.")
    else:
        try:
            samples = load_robot_samples(args.n_samples, seed=args.seed)
        except Exception as e:
            print(f"[data] Robot data load failed ({e}); falling back to synthetic.")
            samples = _synthetic_samples(args.n_samples, seed=args.seed)

    # Load models
    print("[load] Loading FP16 model...")
    fp16_model, processor = load_fp16_model()
    print("[load] Loading 1-bit quantized model...")
    quant_model, _ = load_quant_model(args.quant_llm)

    # Run evaluation
    results = run_eval(fp16_model, quant_model, processor, samples, dev)

    print_results(results)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] Results saved to {args.output}")


if __name__ == "__main__":
    main()
