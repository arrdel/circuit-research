"""Observational & correlational head/layer localization baselines from recent DiT/diffusion literature.

Implements:
1. Attention Contribution (Zarei et al., 2025, arXiv:2505.18832) & Norm-Based Output (Maliha & Hougen, ACL 2026)
2. Head-Resolved Distributional Divergence / KL Spike (Maliha & Hougen, ACL 2026)
3. Spatial Gradient Synopsis (Wang et al., 2026, arXiv:2601.06338)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from diffusers import PixArtTransformer2DModel

from instrumentation.hooks import ActivationCache


@dataclass
class HeadObservationalScores:
    layer: int
    head: int
    attn_contribution: float
    kl_spike: float
    spatial_gradient_abs: float


def compute_spatial_gradient_score(
    attn_probs: torch.Tensor,
    token_indices: list[int],
) -> np.ndarray:
    """Wang et al. (2026) style spatial gradient correlation per head.

    attn_probs: shape (1, H, N_img, N_txt)
    Returns array of shape (H,) with |corr(map_h, x_coord)| averaged over token_indices.
    """
    probs = attn_probs[0].float()  # (H, N_img, N_txt)
    num_heads, n_img, _ = probs.shape
    side = int(math.isqrt(n_img))
    if side * side != n_img:
        raise ValueError(f"Expected square image token grid, got {n_img}")

    x_1d = torch.linspace(-1.0, 1.0, side, device=probs.device)
    x_grid = x_1d.view(1, side).expand(side, side).reshape(n_img)  # (N_img,)
    x_centered = x_grid - x_grid.mean()
    x_norm = x_centered.norm().clamp_min(1e-8)

    scores = np.zeros(num_heads, dtype=np.float64)
    valid_toks = [t for t in token_indices if t < probs.shape[-1]]
    if not valid_toks:
        return scores

    for t_idx in valid_toks:
        m = probs[:, :, t_idx]  # (H, N_img)
        m_c = m - m.mean(dim=-1, keepdim=True)
        m_norm = m_c.norm(dim=-1).clamp_min(1e-8)
        corr = (m_c * x_centered.unsqueeze(0)).sum(dim=-1) / (m_norm * x_norm)
        scores += corr.abs().cpu().numpy()
    return scores / len(valid_toks)


def compute_head_norm_distribution(
    transformer: PixArtTransformer2DModel,
    layer_idx: int,
    attn_probs: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Compute per-head projected token-contribution strength G_h(k) = E_q ||A_{h,q,k} V_{h,k} W_O^{(h)}||_2.

    Returns tensor of shape (H, N_txt).
    """
    attn = transformer.transformer_blocks[layer_idx].attn2
    assert attn is not None
    device = next(attn.parameters()).device
    dtype = next(attn.parameters()).dtype

    probs = attn_probs[0].to(device=device, dtype=torch.float32)  # (H, N_img, N_txt)
    enc = encoder_hidden_states[0].to(device=device, dtype=dtype)  # (N_txt, D_in)

    with torch.no_grad():
        val = attn.to_v(enc)  # (N_txt, H * D_h)
        num_heads = attn.heads
        head_dim = val.shape[-1] // num_heads
        val_h = val.view(-1, num_heads, head_dim).permute(1, 0, 2).float()  # (H, N_txt, D_h)

        # Output projection slice per head: W_O has shape (D_out, H * D_h)
        w_out = attn.to_out[0].weight.detach().float()  # (D_out, H * D_h)
        w_out_h = w_out.view(w_out.shape[0], num_heads, head_dim).permute(1, 2, 0)  # (H, D_h, D_out)

        # Projected value vector norm per head & token: ||V_{h,k} W_O^{(h)}||_2 -> (H, N_txt)
        proj_val = torch.bmm(val_h, w_out_h)  # (H, N_txt, D_out)
        proj_norm = proj_val.norm(dim=-1)  # (H, N_txt)

        # Since A_{h,q,k} >= 0, E_q[||A_{h,q,k} * v_{h,k}||_2] = (mean_q A_{h,q,k}) * ||v_{h,k}||_2
        mean_attn = probs.mean(dim=1)  # (H, N_txt)
        g_hk = mean_attn * proj_norm
    return g_hk.cpu()


def score_heads_from_pair_caches(
    transformer: PixArtTransformer2DModel,
    clean_cache: ActivationCache,
    corrupt_cache: ActivationCache,
    layers: list[int],
    steps: list[int],
    target_token_indices: list[int],
    eps: float = 1e-6,
) -> list[HeadObservationalScores]:
    """Compute Zarei (2025), Maliha & Hougen (2026), and Wang (2026) head scores."""
    num_heads = int(transformer.config.num_attention_heads)
    results: list[HeadObservationalScores] = []

    for l_idx in layers:
        contrib_acc = np.zeros(num_heads, dtype=np.float64)
        kl_acc = np.zeros(num_heads, dtype=np.float64)
        grad_acc = np.zeros(num_heads, dtype=np.float64)
        n_steps = 0

        for s_idx in steps:
            if (s_idx, l_idx) not in clean_cache.attn_probs or (s_idx, l_idx) not in corrupt_cache.attn_probs:
                continue
            p_clean = clean_cache.attn_probs[(s_idx, l_idx)]
            p_corrupt = corrupt_cache.attn_probs[(s_idx, l_idx)]
            enc_clean = clean_cache.get(s_idx, l_idx, "attn2_in_txt")
            enc_corrupt = corrupt_cache.get(s_idx, l_idx, "attn2_in_txt")

            g_clean = compute_head_norm_distribution(transformer, l_idx, p_clean, enc_clean)
            g_corrupt = compute_head_norm_distribution(transformer, l_idx, p_corrupt, enc_corrupt)

            # 1. Attention contribution on target relational tokens (Zarei 2025 / Maliha 2026)
            valid_t = [t for t in target_token_indices if t < g_clean.shape[1]]
            if valid_t:
                contrib_acc += g_clean[:, valid_t].sum(dim=-1).numpy()

            # 2. KL spike between clean and corrupt token contribution distributions (Maliha 2026)
            dist_c = (g_clean + eps)
            dist_c = dist_c / dist_c.sum(dim=-1, keepdim=True)
            dist_r = (g_corrupt + eps)
            dist_r = dist_r / dist_r.sum(dim=-1, keepdim=True)
            kl = (dist_c * (dist_c.log() - dist_r.log())).sum(dim=-1).numpy()
            kl_acc += kl

            # 3. Spatial gradient correlation on target tokens (Wang 2026)
            grad_acc += compute_spatial_gradient_score(p_clean, valid_t)
            n_steps += 1

        denom = max(1, n_steps)
        for h in range(num_heads):
            results.append(
                HeadObservationalScores(
                    layer=l_idx,
                    head=h,
                    attn_contribution=float(contrib_acc[h] / denom),
                    kl_spike=float(kl_acc[h] / denom),
                    spatial_gradient_abs=float(grad_acc[h] / denom),
                )
            )
    return results
