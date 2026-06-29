#!/usr/bin/env python3
"""
LIBERO simulation evaluation for quantized OpenVLA.

Runs the policy autoregressively in LIBERO-Spatial (or other suite),
measures task success rate over N episodes per task.

Usage:
  # FP16 baseline:
  MUJOCO_GL=egl python eval_libero.py --suite libero_spatial --episodes 20

  # Quantized model:
  MUJOCO_GL=egl python eval_libero.py --suite libero_spatial --episodes 20 \
      --quant_path output/openvla_exp_f_fixed_salience/quantized_llm

  # Quick smoke test (1 task, 3 episodes):
  MUJOCO_GL=egl python eval_libero.py --suite libero_spatial --n_tasks 1 --episodes 3
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# LIBERO repo on path
sys.path.insert(0, os.environ.get("LIBERO_PATH", "."))

HF_CACHE   = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OPENVLA_ID = "openvla/openvla-7b"
MAX_STEPS  = 300   # per episode (LIBERO horizon)
UNNORM_KEY = "libero_spatial"   # will be overridden per suite if available


LIBERO_UNNORM_KEYS = {
    "libero_spatial": "libero_spatial",
    "libero_object":  "libero_object",
    "libero_goal":    "libero_goal",
    "libero_10":      "libero_10",
}


def load_openvla(quant_path=None):
    from transformers import AutoProcessor, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    processor = AutoProcessor.from_pretrained(
        OPENVLA_ID, trust_remote_code=True, cache_dir=HF_CACHE)
    model_cls = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction",
        OPENVLA_ID, cache_dir=HF_CACHE, trust_remote_code=True)
    model = model_cls.from_pretrained(
        OPENVLA_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, cache_dir=HF_CACHE, attn_implementation="eager")

    if quant_path is not None:
        print(f"[load] Replacing LM with quantized weights from {quant_path}")
        quant_lm = AutoModelForCausalLM.from_pretrained(
            quant_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model.language_model = quant_lm

    model.eval()
    return model, processor


def predict_action(model, processor, image: Image.Image, instruction: str,
                   unnorm_key: str, dev: torch.device) -> np.ndarray:
    """
    Autoregressive action prediction — 7 tokens generated one at a time.
    Returns continuous 7-DOF action.
    """
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    base_ids = inputs["input_ids"]

    # Generate 7 action tokens one at a time (true autoregressive)
    cur_ids = base_ids
    pv      = inputs["pixel_values"]
    pred_ids = []

    with torch.no_grad():
        for _ in range(7):
            out      = model(input_ids=cur_ids, pixel_values=pv, return_dict=True)
            next_tok = out.logits[0, -1, :].argmax().item()
            pred_ids.append(next_tok)
            cur_ids = torch.cat([cur_ids,
                                 torch.tensor([[next_tok]], device=dev)], dim=1)
            pv = pv  # vision features unchanged

    # Decode token IDs → continuous actions
    norm_stats = model.norm_stats.get(unnorm_key, None)
    if norm_stats is None:
        # Fall back to bridge_orig if suite key not in model
        norm_stats = model.norm_stats.get("bridge_orig", None)
        if norm_stats is None:
            # Last resort: find any key
            norm_stats = next(iter(model.norm_stats.values()))
    action_ns = norm_stats.get("action", norm_stats)

    bins = np.array(pred_ids, dtype=np.float32) - 31745  # 0–255
    norm = bins / 255.0 * 2.0 - 1.0
    q01  = np.asarray(action_ns["q01"], dtype=np.float32)
    q99  = np.asarray(action_ns["q99"], dtype=np.float32)
    return (norm + 1.0) / 2.0 * (q99 - q01) + q01


def eval_task(model, processor, task, suite_name: str,
              n_episodes: int, dev: torch.device, max_steps: int = MAX_STEPS):
    """Run N episodes of one LIBERO task. Returns (n_success, n_episodes)."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_path = os.path.join(
        get_libero_path("bddl_files"), suite_name, task.bddl_file)
    unnorm_key = LIBERO_UNNORM_KEYS.get(suite_name, "bridge_orig")

    n_success = 0
    for ep in range(n_episodes):
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=224,
            camera_widths=224,
        )
        obs      = env.reset()
        done     = False
        success  = False

        for step in range(max_steps):
            img   = Image.fromarray(obs["agentview_image"][::-1])  # flip vertically
            action = predict_action(
                model, processor, img, task.language, unnorm_key, dev)

            obs, reward, done, info = env.step(action)
            if done or info.get("success", False):
                success = True
                break

        env.close()
        if success:
            n_success += 1

    return n_success, n_episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite",      default="libero_spatial",
                   choices=["libero_spatial", "libero_object",
                            "libero_goal", "libero_10"])
    p.add_argument("--quant_path", default=None,
                   help="Path to quantized LLM dir. Omit for FP16 baseline.")
    p.add_argument("--episodes",   type=int, default=20,
                   help="Episodes per task")
    p.add_argument("--n_tasks",    type=int, default=None,
                   help="Evaluate first N tasks (default: all 10)")
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--output",     default=None)
    args = p.parse_args()

    dev   = torch.device(args.device)
    label = "quantized" if args.quant_path else "fp16"

    print(f"[eval] Suite: {args.suite} | Model: {label} | "
          f"{args.episodes} episodes/task")

    from libero.libero import benchmark
    bm     = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = args.n_tasks or bm.n_tasks
    tasks   = [bm.get_task(i) for i in range(n_tasks)]

    print(f"[load] Loading OpenVLA model...")
    model, processor = load_openvla(args.quant_path)
    model = model.to(dev)

    results_per_task = {}
    total_success    = 0
    total_episodes   = 0

    for ti, task in enumerate(tasks):
        print(f"\n[task {ti+1}/{n_tasks}] {task.name}")
        ns, ne = eval_task(
            model, processor, task, args.suite,
            args.episodes, dev)
        sr = ns / ne * 100
        results_per_task[task.name] = {"success": ns, "total": ne, "rate": sr}
        total_success   += ns
        total_episodes  += ne
        print(f"  → {ns}/{ne} = {sr:.1f}%")

    overall_sr = total_success / total_episodes * 100

    print("\n" + "="*55)
    print(f"  LIBERO RESULTS — {label.upper()} — {args.suite}")
    print("="*55)
    for name, r in results_per_task.items():
        bar = "█" * int(r["rate"] / 5)
        print(f"  {r['rate']:5.1f}%  {bar}  {name[:55]}")
    print(f"\n  Overall success rate: {overall_sr:.1f}%  "
          f"({total_success}/{total_episodes})")
    print("="*55)

    out = {
        "suite":       args.suite,
        "model":       label,
        "quant_path":  args.quant_path,
        "episodes":    args.episodes,
        "overall_sr":  overall_sr,
        "per_task":    results_per_task,
    }
    out_path = (args.output or
                f"output/libero_{args.suite}_{label}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] Results → {out_path}")


if __name__ == "__main__":
    main()
