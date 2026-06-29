#!/usr/bin/env python3
"""
LIBERO evaluation for OpenVLA — FP16 and quantized.
Mirrors the official experiments/robot/libero/run_libero_eval.py
but drops TensorFlow (uses PIL) and flash_attention_2 (uses eager).

Correct pipeline:
  1. FP16 baseline:    python run_libero_eval.py --ckpt <hf_ckpt> --suite libero_spatial
  2. Quantized:        python run_libero_eval.py --ckpt <hf_ckpt> --suite libero_spatial
                           --quant_path output/libero_spatial_quantized/quantized_llm

Use MUJOCO_GL=egl for headless GPU rendering.

Official paper settings:
  --num_trials_per_task 50 --seed 7 --center_crop
  (gives 500 rollouts per suite = 10 tasks × 50 trials)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tqdm
import torch

sys.path.insert(0, os.environ.get("LIBERO_PATH", "."))
sys.path.insert(0, os.environ.get("OPENVLA_PATH", "."))

from libero.libero import benchmark
from libero_eval_utils import (
    get_libero_env,
    get_libero_image,
    get_libero_dummy_action,
    get_vla_action,
    load_openvla_for_libero,
    normalize_gripper_action,
    invert_gripper_action,
    set_seed_everywhere,
)

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Max steps per suite (from official script)
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
    "libero_90":      400,
}
NUM_STEPS_WAIT = 10   # let objects settle before acting


def resolve_checkpoint(ckpt: str) -> str:
    """
    Accept either a HF model id (openvla/openvla-7b-finetuned-libero-spatial)
    or a local path. Returns a local path.
    """
    if os.path.isdir(ckpt):
        return ckpt
    # Treat as HF model id — find in HF cache
    from huggingface_hub import snapshot_download
    print(f"[load] Downloading {ckpt} → {HF_CACHE} ...")
    path = snapshot_download(
        ckpt, cache_dir=HF_CACHE,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     required=True,
                   help="HF model id OR local path to finetuned checkpoint")
    p.add_argument("--suite",    default="libero_spatial",
                   choices=["libero_spatial", "libero_object",
                            "libero_goal", "libero_10", "libero_90"])
    p.add_argument("--quant_path", default=None,
                   help="Path to quantized LLM dir. Omit for FP16 baseline.")
    p.add_argument("--num_trials_per_task", type=int, default=50)
    p.add_argument("--n_tasks",  type=int, default=None,
                   help="Limit to first N tasks (default: all)")
    p.add_argument("--center_crop", action="store_true", default=True,
                   help="Apply center crop (True for all official finetuned ckpts)")
    p.add_argument("--no_center_crop", dest="center_crop", action="store_false")
    p.add_argument("--seed",     type=int, default=7)
    p.add_argument("--device",   default="cuda:0")
    p.add_argument("--output",   default=None)
    args = p.parse_args()

    set_seed_everywhere(args.seed)
    dev   = torch.device(args.device)
    label = "quantized" if args.quant_path else "fp16"

    # ── Resolve checkpoint path ──────────────────────────────────────────────
    ckpt_path = resolve_checkpoint(args.ckpt)
    print(f"[config] ckpt: {ckpt_path}")
    print(f"[config] suite: {args.suite} | model: {label} | "
          f"trials/task: {args.num_trials_per_task} | seed: {args.seed}")

    # ── Load model ───────────────────────────────────────────────────────────
    model, processor, norm_stats = load_openvla_for_libero(
        ckpt_path, quant_lm_path=args.quant_path, device=dev)

    unnorm_key = args.suite
    if unnorm_key not in model.norm_stats:
        # Sometimes key has _no_noops suffix
        alt = f"{unnorm_key}_no_noops"
        if alt in model.norm_stats:
            unnorm_key = alt
        else:
            print(f"[WARN] unnorm_key '{unnorm_key}' not in norm_stats. "
                  f"Available: {list(model.norm_stats.keys())}")

    # ── Init LIBERO ──────────────────────────────────────────────────────────
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks    = args.n_tasks or task_suite.n_tasks
    max_steps  = MAX_STEPS.get(args.suite, 300)

    # ── Logging ──────────────────────────────────────────────────────────────
    os.makedirs("experiments/logs", exist_ok=True)
    run_id   = f"EVAL-{args.suite}-{label}-{time.strftime('%Y%m%d-%H%M%S')}"
    log_path = f"experiments/logs/{run_id}.txt"
    log_file = open(log_path, "w")
    print(f"[log] {log_path}")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    results_per_task  = {}
    total_success     = 0
    total_episodes    = 0

    for task_id in tqdm.tqdm(range(n_tasks), desc="tasks"):
        task              = task_suite.get_task(task_id)
        initial_states    = task_suite.get_task_init_states(task_id)
        env, task_desc    = get_libero_env(task, resolution=256)

        task_success  = 0
        task_episodes = 0

        for ep_idx in tqdm.tqdm(range(args.num_trials_per_task),
                                 desc=f"  {task.name[:40]}", leave=False):
            env.reset()
            obs      = env.set_init_state(initial_states[ep_idx])
            done     = False
            t        = 0

            while t < max_steps + NUM_STEPS_WAIT:
                # Wait for sim to stabilize
                if t < NUM_STEPS_WAIT:
                    obs, _, done, info = env.step(get_libero_dummy_action())
                    t += 1
                    continue

                try:
                    img    = get_libero_image(obs, resize_size=224)
                    action = get_vla_action(
                        model, processor, img,
                        task_desc, unnorm_key,
                        center_crop_flag=args.center_crop,
                        device=dev,
                    )
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)
                    obs, _, done, info = env.step(action.tolist())
                    if done:
                        break
                except Exception as e:
                    log_file.write(f"[ep {ep_idx}] exception: {e}\n")
                    break
                t += 1

            if done:
                task_success  += 1
                total_success += 1
            task_episodes  += 1
            total_episodes += 1

        env.close()
        sr = task_success / task_episodes * 100
        results_per_task[task.name] = {
            "success": task_success, "total": task_episodes, "rate": sr}
        log_file.write(
            f"[task {task_id}] {task.name}: {task_success}/{task_episodes} = {sr:.1f}%\n")
        log_file.flush()
        print(f"  {sr:.1f}%  {task.name}")

    log_file.close()
    overall_sr = total_success / total_episodes * 100

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  LIBERO {args.suite.upper()} — {label.upper()}")
    print("=" * 60)
    for name, r in results_per_task.items():
        bar = "█" * int(r["rate"] / 5)
        print(f"  {r['rate']:5.1f}%  {bar}  {name[:50]}")
    print(f"\n  OVERALL SUCCESS RATE: {overall_sr:.1f}%  ({total_success}/{total_episodes})")
    print("=" * 60)

    out = {
        "suite":       args.suite,
        "model":       label,
        "ckpt":        ckpt_path,
        "quant_path":  args.quant_path,
        "num_trials":  args.num_trials_per_task,
        "seed":        args.seed,
        "center_crop": args.center_crop,
        "overall_sr":  overall_sr,
        "per_task":    results_per_task,
    }
    out_path = (args.output or
                f"output/libero_{args.suite}_{label}_seed{args.seed}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
