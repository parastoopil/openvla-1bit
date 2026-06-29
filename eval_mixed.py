#!/usr/bin/env python3
"""
Evaluate the saved mixed-precision quantized model against FP16 baseline.
Runs predict_action() on 50 ALOHA robot frames, one model at a time.
Usage: python eval_mixed.py --quant_path output/openvla_mixed_4bit1bit/quantized_llm
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

HF_CACHE   = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OPENVLA_ID = "openvla/openvla-7b"
INSTRUCTIONS = [
    "In: What action should the robot take to insert the peg into the socket?\nOut:",
    "In: What action should the robot take to grasp the component?\nOut:",
    "In: What action should the robot take to move to the target position?\nOut:",
    "In: What action should the robot take to complete the manipulation task?\nOut:",
    "In: What action should the robot take to align the objects?\nOut:",
]

def load_full_model(model_id=OPENVLA_ID, llm_path=None):
    from transformers import AutoProcessor, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=HF_CACHE)
    model_cls = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction",
        model_id, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    model = model_cls.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, cache_dir=HF_CACHE, attn_implementation="eager",
    )
    if llm_path is not None:
        lm = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model.language_model = lm
    model.eval()
    return model, processor

@torch.no_grad()
def run_predictions(model, processor, frames, dev, label):
    model = model.to(dev).eval()
    preds = []
    for si, (img, instr) in enumerate(frames):
        inputs = processor(instr, img, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        pv     = inputs["pixel_values"].to(dtype=torch.bfloat16)
        try:
            a = model.predict_action(input_ids=inputs["input_ids"],
                                     unnorm_key="bridge_orig", pixel_values=pv)
            preds.append(a)
        except Exception as e:
            print(f"  [{label}] sample {si} failed: {e}")
            preds.append(None)
        if (si + 1) % 10 == 0:
            print(f"  [{label}] {si+1}/{len(frames)} done")
    model.cpu()
    torch.cuda.empty_cache()
    return preds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quant_path", default="output/openvla_mixed_4bit1bit/quantized_llm")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_eval", type=int, default=50)
    p.add_argument("--output", default="results/action_accuracy_mixed.json")
    args = p.parse_args()

    dev = torch.device(args.device)

    from quant_robot_calibrated import load_bridge_eval_samples
    print(f"[data] Loading {args.n_eval} BridgeDataV2 eval frames (one per episode)...")
    raw_samples = load_bridge_eval_samples(args.n_eval)
    # raw_samples is a list of (PIL.Image, instruction_str) pairs

    print("\n[fp16] Loading FP16 model...")
    fp16_model, processor = load_full_model()
    fp16_preds = run_predictions(fp16_model, processor, raw_samples, dev, "fp16")
    del fp16_model; torch.cuda.empty_cache()

    print(f"\n[quant] Loading quantized model from {args.quant_path}...")
    quant_model, _ = load_full_model(llm_path=args.quant_path)
    quant_preds = run_predictions(quant_model, processor, raw_samples, dev, "quant")
    del quant_model; torch.cuda.empty_cache()

    fp16_acts, quant_acts = [], []
    for f, q in zip(fp16_preds, quant_preds):
        if f is not None and q is not None:
            fp16_acts.append(f); quant_acts.append(q)

    n = len(fp16_acts)
    print(f"\n[results] {n} valid sample pairs")
    fp16_arr  = np.stack(fp16_acts)
    quant_arr = np.stack(quant_acts)
    l1        = np.abs(fp16_arr - quant_arr)
    DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    results = {
        "n_valid": n,
        "mean_action_l1": float(l1.mean()),
        "mean_action_l2": float(np.sqrt((l1**2).sum(1)).mean()),
        "mean_cosine_sim": float(np.mean([
            np.dot(fp16_acts[i], quant_acts[i]) /
            (np.linalg.norm(fp16_acts[i]) * np.linalg.norm(quant_acts[i]) + 1e-8)
            for i in range(n)])),
        "per_dim_l1":  l1.mean(0).tolist(),
        "per_dim_std": l1.std(0).tolist(),
        "per_dim_labels": DIM_NAMES,
        "fp16_actions_sample":  fp16_arr[:5].tolist(),
        "quant_actions_sample": quant_arr[:5].tolist(),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] Results → {args.output}")

    print("\n" + "="*55)
    print("  EXPERIMENT B RESULTS  (robot-calibrated 4-bit/1-bit)")
    print("="*55)
    print(f"  Mean action L1   : {results['mean_action_l1']:.4f}")
    print(f"  Cosine similarity: {results['mean_cosine_sim']:.4f}")
    print()
    print(f"  {'Dim':7s}  {'Mixed 4/1-bit':>14s}  {'1-bit C4 baseline':>18s}")
    baseline = [0.006, 0.010, 0.011, 0.036, 0.040, 0.118, 0.552]
    for name, v, b in zip(DIM_NAMES, results["per_dim_l1"], baseline):
        delta = v - b
        sign  = "+" if delta > 0 else ""
        flag  = " ⚠" if v > 0.05 else " ✓"
        print(f"  {name:7s}  {v:14.4f}  {b:18.4f}  ({sign}{delta:.4f}){flag}")
    print("="*55)

if __name__ == "__main__":
    main()
