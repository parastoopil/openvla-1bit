#!/usr/bin/env python3
"""
LoRA Fine-Tuning of OpenVLA on Robot Manipulation Data
=======================================================
Domain-adapts the OpenVLA-7B LLM backbone to robot-specific visual patterns
using Parameter-Efficient Fine-Tuning (LoRA) on ALOHA simulation data.

The fine-tuned model is then used as the starting point for robot-calibrated
mixed-precision quantization (salient=4-bit, non-salient=1-bit).

Dataset:  lerobot/aloha_sim_insertion_scripted_image
Robot:    ALOHA bimanual sim — we use the first 7 DOF (left arm + gripper)
Loss:     Cross-entropy on the 7 action token positions
LoRA:     r=16, targets q/v/o/gate/up/down projections in the LLM backbone

Usage:
  python finetune_lora.py --device cuda:0 --steps 500 --output output/openvla_lora
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from peft import LoraConfig, get_peft_model, TaskType

HF_CACHE   = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OPENVLA_ID = "openvla/openvla-7b"
VOCAB_SIZE = 32000   # model.vocab_size (action tokens live below this)
N_BINS     = 256
ACTION_DIM = 7       # OpenVLA always predicts 7-DOF
ALOHA_DOF  = 7       # use first 7 DOF of ALOHA's 14-DOF action

INSTRUCTIONS = [
    "insert the peg into the socket",
    "grasp the small component and place it in position",
    "pick up the object and align it with the target",
    "move the robot arm to complete the insertion task",
    "manipulate the object to reach the goal configuration",
]

# ── Action tokenization helpers ──────────────────────────────────────────────

def compute_action_stats(actions: np.ndarray) -> dict:
    """Compute per-dimension q01/q99 normalization stats from an action array."""
    q01 = np.percentile(actions, 1, axis=0)
    q99 = np.percentile(actions, 99, axis=0)
    # Avoid degenerate ranges
    rng = q99 - q01
    rng[rng < 1e-6] = 1.0
    return {"q01": q01.tolist(), "q99": q99.tolist()}


def actions_to_tokens(actions: np.ndarray, stats: dict) -> np.ndarray:
    """
    Convert continuous 7-DOF actions to OpenVLA token IDs.

    OpenVLA decoding (predict_action):
        discretized = vocab_size - token_id
        bin_idx      = clip(discretized - 1, 0, 254)
        normalized   = bin_centers[bin_idx]
        action       = 0.5*(normalized+1)*(q99-q01) + q01

    Inverse (encoding):
        normalized = 2*(action - q01)/(q99 - q01) - 1, clipped to [-1, 1]
        bin_idx    = searchsorted(bin_centers, normalized), clipped to [0, 254]
        token_id   = vocab_size - (bin_idx + 1)
    """
    bins        = np.linspace(-1, 1, N_BINS)        # 256 edges
    bin_centers = (bins[:-1] + bins[1:]) / 2.0      # 255 centers

    q01 = np.array(stats["q01"])
    q99 = np.array(stats["q99"])
    rng = q99 - q01
    rng[rng < 1e-6] = 1.0

    normalized = 2.0 * (actions - q01) / rng - 1.0
    normalized = np.clip(normalized, -1.0, 1.0)

    bin_idx  = np.searchsorted(bin_centers, normalized)   # 0..255
    bin_idx  = np.clip(bin_idx, 0, 254)                   # 0..254 (255 centers)
    token_ids = VOCAB_SIZE - (bin_idx + 1)                # 31745..31999
    return token_ids.astype(np.int64)


# ── Dataset ──────────────────────────────────────────────────────────────────

class ALOHADataset(IterableDataset):
    """
    Streams (image, instruction, action_token_ids) tuples from the ALOHA
    simulation dataset.  Uses the first 7 of ALOHA's 14 DOF as the action.
    """

    def __init__(self, processor, action_stats: dict, seed: int = 42,
                 step: int = 5):
        from datasets import load_dataset
        self.processor    = processor
        self.action_stats = action_stats
        self.seed         = seed
        self.step         = step  # stride to avoid redundant frames

        self.ds = load_dataset(
            "lerobot/aloha_sim_insertion_scripted_image",
            split="train",
            streaming=True,
            cache_dir=HF_CACHE,
        )

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        for i, item in enumerate(self.ds):
            if i % self.step != 0:
                continue
            try:
                img    = item["observation.images.top"]
                action = np.array(item["action"])[:ALOHA_DOF].astype(np.float32)
                instr  = INSTRUCTIONS[i % len(INSTRUCTIONS)]
                prompt = f"In: What action should the robot take to {instr}?\nOut:"

                # Tokenise image + prompt
                inputs = self.processor(prompt, img, return_tensors="pt")

                # Encode action as 7 token IDs
                tok_ids = actions_to_tokens(action[None, :], self.action_stats)[0]  # (7,)

                yield {
                    "input_ids":    inputs["input_ids"].squeeze(0),
                    "pixel_values": inputs["pixel_values"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0) if "attention_mask" in inputs else None,
                    "action_tokens": torch.tensor(tok_ids, dtype=torch.long),
                }
            except Exception:
                continue


def collate_fn(batch):
    input_ids    = torch.stack([b["input_ids"] for b in batch])
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    action_tokens = torch.stack([b["action_tokens"] for b in batch])
    attention_mask = None
    if batch[0]["attention_mask"] is not None:
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "action_tokens": action_tokens,
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor():
    from transformers import AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    processor = AutoProcessor.from_pretrained(
        OPENVLA_ID, trust_remote_code=True, cache_dir=HF_CACHE
    )
    model_cls = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction",
        OPENVLA_ID, cache_dir=HF_CACHE, trust_remote_code=True,
    )
    model = model_cls.from_pretrained(
        OPENVLA_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, cache_dir=HF_CACHE, attn_implementation="eager",
    )
    return model, processor


def apply_lora(model, rank: int = 16, alpha: int = 32):
    """Apply LoRA to the LLM backbone's attention and MLP projections."""
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    # LoRA is applied to the inner language model
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    model.language_model.print_trainable_parameters()
    return model


# ── Training ─────────────────────────────────────────────────────────────────

def compute_action_stats_from_dataset(n_samples: int = 2000) -> dict:
    """Scan the ALOHA dataset to compute action normalization statistics."""
    from datasets import load_dataset

    print(f"[stats] Collecting action stats from {n_samples} ALOHA frames...")
    ds = load_dataset(
        "lerobot/aloha_sim_insertion_scripted_image",
        split="train", streaming=True, cache_dir=HF_CACHE,
    )
    actions = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        actions.append(np.array(item["action"])[:ALOHA_DOF])

    actions = np.stack(actions)   # (N, 7)
    stats   = compute_action_stats(actions)
    print(f"[stats] q01={[f'{v:.3f}' for v in stats['q01']]}")
    print(f"[stats] q99={[f'{v:.3f}' for v in stats['q99']]}")
    return stats


def train(args):
    dev = torch.device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Action stats ──────────────────────────────────────────────────────────
    stats_path = out / "aloha_action_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            action_stats = json.load(f)
        print("[stats] Loaded cached action stats.")
    else:
        action_stats = compute_action_stats_from_dataset(n_samples=2000)
        with open(stats_path, "w") as f:
            json.dump(action_stats, f, indent=2)

    # ── Model ────────────────────────────────────────────────────────────────
    print("[load] Loading OpenVLA-7B...")
    model, processor = load_model_and_processor()
    model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model = model.to(dev)
    model.train()

    # Only LoRA parameters get gradients; freeze vision backbone + projector
    for p in model.vision_backbone.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = False

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = ALOHADataset(processor, action_stats, step=3)
    # Use a simple iterator (streaming, no DataLoader needed)
    data_iter = iter(dataset)

    # ── Training loop ─────────────────────────────────────────────────────────
    GRAD_ACCUM = args.grad_accum
    loss_window = []
    model_dtype = torch.bfloat16

    print(f"\n[train] Starting LoRA fine-tuning for {args.steps} steps "
          f"(grad_accum={GRAD_ACCUM}, lr={args.lr})...")

    global_step = 0
    optimizer.zero_grad()

    for micro_step in range(args.steps * GRAD_ACCUM):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            batch = next(data_iter)

        input_ids     = batch["input_ids"].unsqueeze(0).to(dev)
        pixel_values  = batch["pixel_values"].unsqueeze(0).to(dev, dtype=model_dtype)
        action_tokens = batch["action_tokens"].to(dev)  # (7,)

        # Append the OpenVLA training-time separator token (29871) + action tokens
        # to form the full label sequence
        sep   = torch.tensor([[29871]], device=dev)
        full_ids = torch.cat([input_ids, sep, action_tokens.unsqueeze(0)], dim=1)

        # Labels: -100 everywhere except the 7 action token positions
        labels = torch.full_like(full_ids, -100)
        labels[:, -ACTION_DIM:] = action_tokens.unsqueeze(0)

        # Forward pass through the full VLM
        out = model(
            input_ids=full_ids,
            pixel_values=pixel_values,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        loss_window.append(loss.item() * GRAD_ACCUM)

        if (micro_step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 50 == 0 or global_step == 1:
                avg_loss = np.mean(loss_window[-50:])
                lr_now   = scheduler.get_last_lr()[0]
                print(f"  step {global_step:4d}/{args.steps}  loss={avg_loss:.4f}  lr={lr_now:.2e}")

    # ── Save LoRA adapter + merged model ─────────────────────────────────────
    print(f"\n[save] Saving LoRA adapter to {out}/lora_adapter/")
    model.language_model.save_pretrained(out / "lora_adapter")

    print("[save] Merging LoRA weights into base model...")
    model.language_model = model.language_model.merge_and_unload()

    print(f"[save] Saving merged language model to {out}/merged_llm/")
    model.language_model.save_pretrained(out / "merged_llm")
    processor.tokenizer.save_pretrained(out / "merged_llm")

    with open(out / "action_stats.json", "w") as f:
        json.dump(action_stats, f, indent=2)

    print("[done] Fine-tuning complete.")
    return out / "merged_llm"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--steps",     type=int, default=500)
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha",type=int, default=32)
    p.add_argument("--grad_accum",type=int, default=4)
    p.add_argument("--output",    default="output/openvla_lora")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
