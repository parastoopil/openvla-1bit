"""
Self-Contained Weight Quantization Utilities
=============================================
Implements the core algorithms needed for 1-bit and mixed-precision PTQ
of LLMs, following the GPTQ + BiLLM framework without any external
quantization library dependency.

Algorithms implemented:
  - GPTQ Hessian accumulation and inversion
  - Per-row salience scoring (Hessian-weighted magnitude)
  - 4-bit integer quantization (per-row scale/zero-point)
  - 1-bit binary residual approximation (BiLLM BRAQ, order 1 & 2)
  - Mixed-precision GPTQ column-block loop

Reference papers:
  GPTQ:  Frantar et al., 2022 (arXiv:2210.17323)
  BiLLM: Huang et al., 2024  (arXiv:2402.04291)
"""

import math
import time
import torch
import torch.nn as nn


# ── 4-bit quantization ────────────────────────────────────────────────────────

@torch.no_grad()
def quantize_4bit(W: torch.Tensor) -> torch.Tensor:
    """
    Symmetric 4-bit integer quantization, per output-channel (row).

    scale[i] = (max(|W[i,:]|)) / 7   (signed range [-7, 7])
    q[i,j]   = clamp(round(W[i,j] / scale[i]), -7, 7)
    W_q[i,j] = scale[i] * q[i,j]

    Args:
        W: float32 weight matrix [oc, ic]
    Returns:
        W_q: float32 dequantized weights (same shape)
    """
    assert W.ndim == 2
    bits  = 4
    maxq  = 2 ** (bits - 1) - 1   # 7  (signed: [-7, 7])

    # Per-row absmax scale
    scale = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scale = scale / maxq

    q  = torch.clamp(torch.round(W / scale), -maxq, maxq)
    return scale * q


# ── 1-bit binary residual approximation ───────────────────────────────────────

@torch.no_grad()
def binarize_braq(W: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Binary Residual Approximation Quantization (BiLLM BRAQ).
    Approximates each row with `order` binary bases:
      order=1: q = alpha * sign(W)           (1-bit)
      order=2: q = alpha1 * sign(W) + alpha2 * sign(W - q1)  (2-bit equivalent)

    The scale alpha is the mean absolute value of the non-zero elements.

    Args:
        W:     float32 weight slice [oc, ic]
        order: number of binary residual passes (1 or 2)
    Returns:
        W_q: float32 approximation (same shape)
    """
    result = torch.zeros_like(W)
    residual = W.clone()

    for _ in range(order):
        # Per-row scale = mean(|residual|), ignoring near-zero rows
        alpha = residual.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        binary = torch.sign(residual)          # -1, 0, +1
        binary[binary == 0] = 1               # treat exact zeros as +1
        layer_q = alpha * binary
        result  = result + layer_q
        residual = residual - layer_q

    return result


# ── Hessian utilities ─────────────────────────────────────────────────────────

class HessianAccumulator:
    """
    Online Hessian accumulator for a single Linear layer.

    Call .add_batch(inp) with each calibration batch's input activations.
    Call .get_hessian() to retrieve the accumulated H matrix.

    Uses the GPTQ formula:  H += 2 * X @ X.T / n_total
    where X is the [ic, batch_tokens] input matrix.
    """

    def __init__(self, n_inputs: int, device: torch.device):
        self.H        = torch.zeros(n_inputs, n_inputs, device=device)
        self.n_total  = 0
        self.device   = device

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """
        Args:
            inp: activation tensor for this layer — may be [seq, ic] or [bsz, seq, ic]
        """
        if inp.ndim == 3:
            inp = inp.reshape(-1, inp.shape[-1])   # [seq*bsz, ic]
        inp  = inp.t().float().to(self.device)     # [ic, seq*bsz]
        n    = inp.shape[1]
        # Running weighted average: H = n_prev/(n_prev+n)*H + 2*X@X.T/(n_prev+n)
        self.H      *= self.n_total / (self.n_total + n)
        self.n_total += n
        self.H      += (math.sqrt(2 / self.n_total) * inp).matmul(
                        (math.sqrt(2 / self.n_total) * inp).t())

    def get_hessian(self) -> torch.Tensor:
        return self.H.clone()


# ── Salience mask ─────────────────────────────────────────────────────────────

@torch.no_grad()
def get_salience_masks(
    W: torch.Tensor,
    H_inv_diag: torch.Tensor,
    n_salient_per_col: int = 50,
) -> tuple:
    """
    Split weight columns into three partitions using Hessian-weighted salience.

    Salience score (BiLLM Eq.3):  s[i,j] = W[i,j]² / [H⁻¹]_{jj}²
    where [H⁻¹]_{jj} is the j-th diagonal of the INVERSE Hessian.
    Small [H⁻¹]_{jj} (= large H_{jj}, high curvature) → large salience → salient.

    The top-`n_salient_per_col` entries per column (by salience) form mask3
    (salient, gets 4-bit).  The rest are split by sign for 1-bit BRAQ.

    Args:
        W:                  [oc, n_cols] float32 weight block
        H_inv_diag:         [n_cols] diagonal of H⁻¹ (NOT of H)
        n_salient_per_col:  number of salient rows to select per column
    Returns:
        (mask1, mask2, mask3): boolean tensors of shape [oc, n_cols]
    """
    oc, nc = W.shape
    h_inv_diag = H_inv_diag.clamp(min=1e-8)    # [nc]

    # BiLLM Eq.3: s_i = w_i² / [H⁻¹]_{ii}²
    # High H_{ii} → small [H⁻¹]_{ii} → large s_i → salient (protect at 4-bit)
    salience = W ** 2 / (h_inv_diag.unsqueeze(0) ** 2)   # [oc, nc]

    # mask3: top-k most salient rows per column
    k        = min(n_salient_per_col, oc)
    _, topk  = salience.topk(k, dim=0)
    mask3    = torch.zeros_like(W, dtype=torch.bool)
    mask3.scatter_(0, topk, True)

    # mask1 / mask2: split the remaining weights by sign
    non_salient = ~mask3
    pos_mask    = (W > 0) & non_salient
    neg_mask    = (W <= 0) & non_salient

    # Ensure every non-salient position is in exactly one of mask1/mask2
    mask1 = pos_mask
    mask2 = neg_mask

    return mask1, mask2, mask3


# ── GPTQ inversion ────────────────────────────────────────────────────────────

@torch.no_grad()
def invert_hessian(H: torch.Tensor, percdamp: float = 0.01) -> torch.Tensor:
    """
    Compute the Cholesky-based inverse of H (GPTQ preprocessing).
    Returns the upper-triangular Cholesky factor of H^{-1}.
    """
    n    = H.shape[0]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0

    damp = percdamp * torch.mean(torch.diag(H))
    idx  = torch.arange(n, device=H.device)
    H[idx, idx] += damp

    H    = torch.linalg.cholesky(H)
    H    = torch.cholesky_inverse(H)
    H    = torch.linalg.cholesky(H, upper=True)
    return H   # this is H^{-1}'s upper Cholesky factor (= Hinv)


# ── Mixed-precision GPTQ loop ─────────────────────────────────────────────────

@torch.no_grad()
def mixed_precision_gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    blocksize: int = 128,
    percdamp: float = 0.01,
    n_salient_per_col: int = 50,
    braq_order_non_salient: int = 1,
    H_salience: torch.Tensor = None,
) -> torch.Tensor:
    """
    GPTQ column-block quantization with mixed 4-bit (salient) / 1-bit (rest).

    For each block of `blocksize` columns:
      1. Detect salient weights using Hessian-weighted magnitude (mask3).
      2. Quantize salient positions → 4-bit integer.
      3. Quantize non-salient positions → 1-bit BRAQ.
      4. Compute and propagate GPTQ quantization error to remaining columns.

    Args:
        W:        [oc, ic] float32 weight matrix
        H:        [ic, ic] Hessian accumulated from calibration activations
        blocksize: number of columns processed per GPTQ block
        percdamp: Hessian damping fraction
        n_salient_per_col: salient weight count per block column

    Returns:
        W_quant: [oc, ic] mixed-precision quantized weight matrix (float32)
    """
    oc, ic = W.shape
    W      = W.clone().float()
    H      = H.clone()

    # Zero out dead columns (inputs with zero activation variance)
    dead        = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead]  = 0.0

    # Invert H for GPTQ error propagation.
    # If H_salience is provided, also invert it separately for salience detection
    # (so we use action-focused salience but well-conditioned H for optimization).
    Hinv = invert_hessian(H, percdamp=percdamp)

    if H_salience is not None:
        # Invert the salience Hessian separately to get its [H_sal⁻¹]_diag
        Hinv_sal   = invert_hessian(H_salience.clone(), percdamp=percdamp)
        H_inv_diag = (Hinv_sal ** 2).sum(0).clamp(min=1e-8)   # [ic] — diag of H_sal⁻¹
        del Hinv_sal
    else:
        # [H⁻¹]_{ii} from the upper Cholesky factor U: (H⁻¹)_{ii} = Σ_k U_{ki}²
        H_inv_diag = (Hinv ** 2).sum(0).clamp(min=1e-8)   # [ic]

    Losses = torch.zeros(oc, device=W.device)

    t0 = time.time()
    for col_st in range(0, ic, blocksize):
        col_ed = min(col_st + blocksize, ic)
        n_cols = col_ed - col_st

        W1    = W[:, col_st:col_ed].clone()   # [oc, n_cols]
        Q1    = torch.zeros_like(W1)
        Err1  = torch.zeros_like(W1)
        Loss1 = torch.zeros_like(W1)
        Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

        # Salience (BiLLM Eq.3): s = w² / [H⁻¹]_{ii}²
        # High curvature (large H_ii) → small [H⁻¹]_ii → large salience → 4-bit
        H_inv_diag_block = H_inv_diag[col_st:col_ed]   # [n_cols]
        mask1, mask2, mask3 = get_salience_masks(W1, H_inv_diag_block, n_salient_per_col)

        # Pre-quantize all three partitions for this block
        q_4bit  = quantize_4bit(W1)              # 4-bit for salient positions
        q_1bit1 = binarize_braq(W1, order=braq_order_non_salient)  # 1-bit partition 1
        q_1bit2 = binarize_braq(W1, order=braq_order_non_salient)  # 1-bit partition 2

        # Column-wise GPTQ error propagation
        for i in range(n_cols):
            w = W1[:, i]
            d = Hinv1[i, i]

            # Select quantized value per row based on salience mask
            q = (q_1bit1[:, i] * mask1[:, i].float() +
                 q_1bit2[:, i] * mask2[:, i].float() +
                 q_4bit[:, i]  * mask3[:, i].float())

            Q1[:, i]   = q
            Loss1[:, i] = (w - q) ** 2 / d ** 2

            err          = (w - q) / d
            Err1[:, i]  = err

        W[:, col_st:col_ed] = Q1
        Losses              += Loss1.sum(1) / 2
        # Propagate error to remaining unquantized columns
        if col_ed < ic:
            W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

    elapsed = time.time() - t0
    return W, elapsed, Losses.sum().item()


@torch.no_grad()
def pure_1bit_gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    blocksize: int = 128,
    percdamp: float = 0.01,
    n_salient_per_col: int = 50,
) -> tuple:
    """
    Pure 1-bit GPTQ quantization (BiLLM BRAQ approach).
    Salient weights get 2nd-order BRAQ (~2-bit), rest get 1st-order (~1-bit).
    This matches Experiment 1's C4-calibrated quantization.

    Returns: (W_quant, elapsed_sec, total_gptq_loss)
    """
    oc, ic = W.shape
    W      = W.clone().float()
    H      = H.clone()

    dead        = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead]  = 0.0

    Hinv       = invert_hessian(H, percdamp=percdamp)
    H_inv_diag = (Hinv ** 2).sum(0).clamp(min=1e-8)   # diag of H⁻¹ from Cholesky factor
    Losses     = torch.zeros(oc, device=W.device)

    t0 = time.time()
    for col_st in range(0, ic, blocksize):
        col_ed = min(col_st + blocksize, ic)
        n_cols = col_ed - col_st

        W1    = W[:, col_st:col_ed].clone()
        Q1    = torch.zeros_like(W1)
        Err1  = torch.zeros_like(W1)
        Loss1 = torch.zeros_like(W1)
        Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

        mask1, mask2, mask3 = get_salience_masks(W1, H_inv_diag[col_st:col_ed], n_salient_per_col)

        q_1bit    = binarize_braq(W1, order=1)   # non-salient: 1st order
        q_2bit    = binarize_braq(W1, order=2)   # salient: 2nd order residual (~2-bit)

        for i in range(n_cols):
            w = W1[:, i]
            d = Hinv1[i, i]

            q = (q_1bit[:, i] * mask1[:, i].float() +
                 q_1bit[:, i] * mask2[:, i].float() +
                 q_2bit[:, i] * mask3[:, i].float())

            Q1[:, i]    = q
            Loss1[:, i] = (w - q) ** 2 / d ** 2
            err          = (w - q) / d
            Err1[:, i]  = err

        W[:, col_st:col_ed] = Q1
        Losses              += Loss1.sum(1) / 2
        if col_ed < ic:
            W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

    elapsed = time.time() - t0
    return W, elapsed, Losses.sum().item()
