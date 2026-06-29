#!/usr/bin/env python3
"""
BridgeDataV2-based action evaluation for quantized OpenVLA.
Uses teacher-forced forward pass (no generate), so works with transformers 5.x.
Loads the quantized model from a saved checkpoint directory.

Usage:
  python eval_bridge.py --quant_path output/openvla_exp_f_fixed_salience/quantized_llm
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

HF_CACHE   = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OPENVLA_ID = "openvla/openvla-7b"
UNNORM_KEY = "bridge_orig"
_BRIDGE_REPO   = "nvidia/BridgeData2_LeRobot_v3"
_BRIDGE_VIDEO  = "videos/observation.images.image_0/chunk-000/file-000.mp4"
_BRIDGE_META   = "data/chunk-000/file-000.parquet"
_BRIDGE_TASKS  = "meta/tasks.parquet"
_BRIDGE_NFRAMES = 12282


def load_openvla(lm_path=None):
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
    if lm_path is not None:
        print(f"[load] Replacing LM with quantized weights from {lm_path}")
        quant_lm = AutoModelForCausalLM.from_pretrained(
            lm_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model.language_model = quant_lm
    model.eval()
    return model, processor


def get_norm_stats(model):
    ns = getattr(model, "norm_stats", {})
    entry = ns[UNNORM_KEY]
    return entry.get("action", entry)


def action_to_token_ids(action, norm_stats):
    q01 = np.asarray(norm_stats["q01"], dtype=np.float32)
    q99 = np.asarray(norm_stats["q99"], dtype=np.float32)
    a   = np.asarray(action, dtype=np.float32)
    norm = np.clip(2.0 * (a - q01) / (q99 - q01 + 1e-8) - 1.0, -1.0, 1.0)
    bins = np.clip(np.round((norm + 1.0) / 2.0 * 255.0).astype(np.int32), 0, 255)
    return (31745 + bins).tolist()


def token_ids_to_action(token_ids, norm_stats):
    bins = np.array(token_ids, dtype=np.float32) - 31745   # 0-255
    norm = bins / 255.0 * 2.0 - 1.0                         # -1 to 1
    q01  = np.asarray(norm_stats["q01"], dtype=np.float32)
    q99  = np.asarray(norm_stats["q99"], dtype=np.float32)
    return (norm + 1.0) / 2.0 * (q99 - q01) + q01


def collect_eval_samples(processor, norm_stats, nsamples=50, offset=5000):
    """
    Load eval frames from BridgeDataV2 starting at `offset` (different from calib).
    All data is already cached so this is fast.
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

    # Sample at uniform stride starting from offset (avoids overlap with calib)
    stride    = max(1, (_BRIDGE_NFRAMES - offset) // nsamples)
    frame_idxs = list(range(offset, _BRIDGE_NFRAMES, stride))[:nsamples]
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
            gt_act_ids = action_to_token_ids(list(row["action"]), norm_stats)
            samples.append({
                "inputs":       {k: v for k, v in inputs.items()},
                "gt_act_ids":   gt_act_ids,
                "gt_action":    list(row["action"]),
            })
        except Exception as e:
            print(f"  warning: frame {fidx} skipped — {e}")
    reader.close()
    print(f"[eval] Loaded {len(samples)} evaluation samples from BridgeDataV2.")
    return samples


@torch.no_grad()
def run_inference(model, samples, dev, norm_stats, label="model"):
    """
    Single-pass teacher-forced evaluation.
    Feeds [vision + instruction + gt_action_tokens] as one 283-token sequence.
    Logit at position (seq_len-1+i) predicts action token i.
    Takes argmax → token IDs → continuous 7-DOF actions.

    Valid for FP16 vs quantized comparison: both use identical GT token context,
    so any difference in predicted action reflects quantization error alone.
    """
    model_dtype = torch.bfloat16
    actions     = []

    model = model.to(dev).eval()

    for si, s in enumerate(samples):
        try:
            inps    = {k: v.to(dev) for k, v in s["inputs"].items()}
            pv      = inps["pixel_values"].to(dtype=model_dtype)
            base_ids = inps["input_ids"]                              # [1, seq_len]
            seq_len  = base_ids.shape[1]

            # Append ground-truth action token IDs to form 283-token sequence
            gt_ids  = torch.tensor([s["gt_act_ids"]], dtype=torch.long, device=dev)
            ext_ids = torch.cat([base_ids, gt_ids], dim=1)           # [1, seq_len+7]

            out    = model(input_ids=ext_ids, pixel_values=pv, return_dict=True)
            logits = out.logits  # [1, N+7, vocab] where N = text+vision expansion

            # The model expands one image placeholder into 256 vision tokens internally.
            # The 7 GT action tokens are the LAST 7 positions of ext_ids.
            # logits[:, -8+i, :] predicts action token i (i=0..6), so slice [-8:-1].
            action_logits = logits[0, -8:-1, :]                     # [7, vocab]
            pred_ids = action_logits.argmax(dim=-1).cpu().tolist()
            actions.append(token_ids_to_action(pred_ids, norm_stats))

        except Exception as e:
            print(f"  [{label}] sample {si} error: {e}")
            actions.append(None)

        if (si + 1) % 10 == 0:
            print(f"  [{label}] {si+1}/{len(samples)}")

    model.cpu()
    torch.cuda.empty_cache()
    return actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quant_path",  required=True,
                   help="Path to saved quantized LLM (e.g. output/openvla_exp_f.../quantized_llm)")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--n_eval",      type=int, default=50)
    p.add_argument("--output",      default=None)
    args = p.parse_args()

    dev = torch.device(args.device)

    print("[load] Loading FP16 model...")
    fp16_model, processor = load_openvla()
    norm_stats = get_norm_stats(fp16_model)

    print("[load] Loading quantized model...")
    quant_model, _ = load_openvla(lm_path=args.quant_path)

    print(f"[eval] Loading {args.n_eval} BridgeDataV2 eval frames...")
    samples = collect_eval_samples(processor, norm_stats, nsamples=args.n_eval)

    print("\n[eval] Running FP16 model predictions...")
    fp16_acts = run_inference(fp16_model, samples, dev, norm_stats, label="fp16")

    print("\n[eval] Running quantized model predictions...")
    quant_acts = run_inference(quant_model, samples, dev, norm_stats, label="quant")

    # Compute metrics
    valid = [(f, q) for f, q in zip(fp16_acts, quant_acts) if f is not None and q is not None]
    n = len(valid)
    if n == 0:
        print("[eval] No valid samples!")
        return

    fp16_arr  = np.stack([v[0] for v in valid])
    quant_arr = np.stack([v[1] for v in valid])
    l1 = np.abs(fp16_arr - quant_arr)

    dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    cosines = [
        np.dot(fp16_arr[i], quant_arr[i]) /
        (np.linalg.norm(fp16_arr[i]) * np.linalg.norm(quant_arr[i]) + 1e-8)
        for i in range(n)
    ]

    results = {
        "quant_path":      args.quant_path,
        "n_valid_samples": n,
        "mean_action_l1":  float(l1.mean()),
        "mean_cosine_sim": float(np.mean(cosines)),
        "per_dim_l1":      l1.mean(0).tolist(),
        "per_dim_labels":  dim_names,
    }

    print("\n" + "="*55)
    print("  RESULTS — Fixed BiLLM Salience (Exp F)")
    print("="*55)
    print(f"\n  Per-dim L1 error vs FP16:")
    for name, v in zip(dim_names, results["per_dim_l1"]):
        bar  = "█" * max(0, int(v * 80))
        flag = " ⚠" if v > 0.05 else ""
        print(f"    {name:7s}: {v:.4f}  {bar}{flag}")

    print(f"\n  Mean L1          : {results['mean_action_l1']:.4f}  (Exp D: 0.043)")
    print(f"  Cosine similarity: {results['mean_cosine_sim']:.4f}  (Exp D: 1.000)")
    print(f"  Gripper L1       : {results['per_dim_l1'][6]:.4f}  (Exp D: 0.204)")
    print("="*55)

    out_path = args.output or str(Path(args.quant_path).parent / "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] Results → {out_path}")


if __name__ == "__main__":
    main()
