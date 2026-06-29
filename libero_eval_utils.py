"""
Patched LIBERO+OpenVLA evaluation utilities.
Drops TensorFlow entirely — uses PIL/numpy for image ops.
Drops flash_attention_2 — uses eager attention.
Supports loading a quantized LLM in place of the FP16 LLM.

PIL equivalents validated to match TF behavior:
  - resize_image: JPEG round-trip + Lanczos resize (matches RLDS dataloader)
  - center_crop:  sqrt(0.9) center crop + bilinear resize (matches crop_and_resize)
  - get_libero_image: 180° rotate + resize (matches official libero_utils)
"""
import io
import json
import math
import os
import random

import imageio
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# ─── Image utilities ──────────────────────────────────────────────────────────

def resize_image(img: np.ndarray, resize_size: tuple) -> np.ndarray:
    """
    PIL equivalent of the TF JPEG encode/decode + Lanczos resize in libero_utils.
    Matches the RLDS dataloader preprocessing used during OpenVLA training.
    """
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    pil = Image.open(buf).convert("RGB")
    pil = pil.resize(resize_size, Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


def get_libero_image(obs: dict, resize_size) -> np.ndarray:
    """Extract + preprocess agentview image from LIBERO obs dict."""
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]          # 180° rotation to match train preprocessing
    return resize_image(img, resize_size)


def center_crop(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    """
    PIL equivalent of TF crop_and_resize with sqrt(crop_scale) center crop.
    Matches get_vla_action center_crop logic in official openvla_utils.py.
    """
    w, h = image.size
    new_w = int(math.sqrt(crop_scale) * w)
    new_h = int(math.sqrt(crop_scale) * h)
    left  = (w - new_w) // 2
    top   = (h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


# ─── Model loading ────────────────────────────────────────────────────────────

def load_openvla_for_libero(checkpoint_path: str,
                             quant_lm_path: str | None = None,
                             device: torch.device | None = None) -> tuple:
    """
    Load OpenVLA finetuned checkpoint and optionally swap in quantized LLM weights.
    Uses trust_remote_code so the checkpoint's bundled model code is used directly —
    no prismatic package import needed (avoids dlimp/TF dependency).

    Args:
        checkpoint_path: Local path to finetuned OpenVLA checkpoint.
        quant_lm_path:   If set, replace language_model with weights from this dir.
        device:          Target device (default: cuda:0).

    Returns:
        (model, processor, norm_stats)
    """
    from transformers import AutoModelForCausalLM

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[load] Loading FP16 model from {checkpoint_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        attn_implementation="eager",        # no flash_attn needed
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if quant_lm_path is not None:
        print(f"[load] Swapping in quantized LLM from {quant_lm_path}")
        quant_lm = AutoModelForCausalLM.from_pretrained(
            quant_lm_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model.language_model = quant_lm

    model = model.to(device).eval()

    # Load dataset statistics for action un-normalization
    stats_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            norm_stats = json.load(f)
        model.norm_stats = norm_stats
    else:
        norm_stats = getattr(model, "norm_stats", {})
        print("[warn] No dataset_statistics.json found at checkpoint path.")

    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    return model, processor, norm_stats


# ─── Inference ────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_vla_action(model, processor, obs_image: np.ndarray,
                   task_label: str, unnorm_key: str,
                   center_crop_flag: bool = True,
                   device: torch.device | None = None) -> np.ndarray:
    """
    Run one autoregressive predict_action call.
    Matches the official get_vla_action logic exactly.
    """
    if device is None:
        device = DEVICE

    image = Image.fromarray(obs_image).convert("RGB")
    if center_crop_flag:
        image = center_crop(image, crop_scale=0.9)

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action


# ─── LIBERO environment ────────────────────────────────────────────────────────

def get_libero_env(task, resolution: int = 256):
    """Create OffScreenRenderEnv for a LIBERO task."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env, task.language


def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Map gripper action [0,1] → [-1,+1], optionally binarize."""
    action[-1] = action[-1] * 2.0 - 1.0
    if binarize:
        action[-1] = 1.0 if action[-1] > 0 else -1.0
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip gripper sign to match LIBERO convention (OpenVLA flips during data loading)."""
    action[-1] = -action[-1]
    return action


def set_seed_everywhere(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
