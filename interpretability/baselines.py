"""Observational, probing, and attention-intervention baselines for CIRCUIT.

Implements baselines grounded in prior diffusion interpretability literature:
1. Attention Contribution (Zarei et al., 2025, arXiv:2505.18832) & Norm-Based Output (Maliha & Hougen, ACL 2026)
2. Head-Resolved Distributional Divergence / KL Spike (Maliha & Hougen, ACL 2026)
3. Spatial Gradient Correlation (CIRCUIT's own diagnostic heuristic, inspired by the two-stage spatial-relation head prior in Wang, Fan, & Pan, 2026, arXiv:2601.06338, which studied two cross-attention heads in from-scratch DiTs)
4. DAAM (Tang et al., 2022, arXiv:2210.04885): upscaled cross-attention word-pixel attribution maps and inside-box grounding mass per (layer, head)
5. Roy (2025, arXiv:2506.17237): destructive ablation impact per (layer, head)
6. Revelio Diff-C Linear Probe (Kim et al., 2025, arXiv:2411.16725): linear probe on spatially-pooled per-head activations
7. Prompt-to-Prompt (Hertz et al., 2022, arXiv:2208.01626): cross-attention map injection via AttnMapInjectionController
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import PixArtTransformer2DModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from evaluation.spatial_eval import DetectionBox
from instrumentation.hooks import ActivationCache


@dataclass
class HeadObservationalScores:
    layer: int
    head: int
    attn_contribution: float
    kl_spike: float
    spatial_gradient_abs: float
    daam_box_mass: float = 0.0
    revelio_probe_auc: float = 0.5
    revelio_probe_acc: float = 0.5


def compute_spatial_gradient_score(
    attn_probs: torch.Tensor,
    token_indices: list[int],
) -> np.ndarray:
    """CIRCUIT diagnostic heuristic: horizontal spatial gradient correlation per head.

    Inspired by Wang, Fan, & Pan (2026, arXiv:2601.06338), which uncovered a two-stage
    circuit with dedicated relation and object cross-attention heads in from-scratch DiTs.
    This per-head |corr(attn_map, x_coord)| metric is CIRCUIT's own diagnostic rather than
    a method proposed by Wang et al.

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


def compute_daam_upscaled_map(
    attn_probs: torch.Tensor,
    token_indices: list[int],
    upsample_size: int = 64,
) -> torch.Tensor:
    """DAAM (Tang et al., 2022) bicubic-upscaled per-head attention attribution maps.

    attn_probs: shape (1, H, N_img, N_txt)
    Returns tensor of shape (H, upsample_size, upsample_size) summed over token_indices.
    """
    probs = attn_probs[0].float()  # (H, N_img, N_txt)
    num_heads, n_img, n_txt = probs.shape
    side = int(math.isqrt(n_img))
    if side * side != n_img:
        raise ValueError(f"Expected square image token grid, got {n_img}")

    valid_toks = [t for t in token_indices if t < n_txt]
    if not valid_toks:
        return torch.zeros((num_heads, upsample_size, upsample_size), dtype=torch.float32)

    tok_map = probs[:, :, valid_toks].sum(dim=-1)  # (H, N_img)
    grid = tok_map.view(1, num_heads, side, side)
    upscaled = F.interpolate(
        grid, size=(upsample_size, upsample_size), mode="bicubic", align_corners=False
    )[0]
    return upscaled.clamp_min(0.0)


def compute_daam_head_box_scores(
    attn_probs: torch.Tensor,
    token_indices: list[int],
    boxes: list[DetectionBox | None],
    upsample_size: int = 64,
) -> np.ndarray:
    """Compute per-head DAAM grounding mass inside target object bounding boxes."""
    up_map = compute_daam_upscaled_map(attn_probs, token_indices, upsample_size=upsample_size)
    num_heads = up_map.shape[0]
    valid_boxes = [b for b in boxes if b is not None]
    if not valid_boxes:
        return np.zeros(num_heads, dtype=np.float64)

    mask = torch.zeros((upsample_size, upsample_size), dtype=torch.float32, device=up_map.device)
    for b in valid_boxes:
        x1 = max(0, min(upsample_size - 1, int(math.floor(b.x1 * upsample_size))))
        x2 = max(x1 + 1, min(upsample_size, int(math.ceil(b.x2 * upsample_size))))
        y1 = max(0, min(upsample_size - 1, int(math.floor(b.y1 * upsample_size))))
        y2 = max(y1 + 1, min(upsample_size, int(math.ceil(b.y2 * upsample_size))))
        mask[y1:y2, x1:x2] = 1.0

    total = up_map.sum(dim=(-2, -1)).clamp_min(1e-8)
    inside = (up_map * mask.unsqueeze(0)).sum(dim=(-2, -1))
    return (inside / total).cpu().numpy().astype(np.float64)


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

        w_out = attn.to_out[0].weight.detach().float()  # (D_out, H * D_h)
        w_out_h = w_out.view(w_out.shape[0], num_heads, head_dim).permute(1, 2, 0)  # (H, D_h, D_out)

        proj_val = torch.bmm(val_h, w_out_h)  # (H, N_txt, D_out)
        proj_norm = proj_val.norm(dim=-1)  # (H, N_txt)

        mean_attn = probs.mean(dim=1)  # (H, N_txt)
        g_hk = mean_attn * proj_norm
    return g_hk.cpu()


def compute_roy_ablation_impact(
    baseline_rel_acc: float,
    ablated_rel_acc: float,
    baseline_dx: float | None = None,
    ablated_dx: float | None = None,
) -> dict[str, float]:
    """Roy (2025, arXiv:2506.17237) destructive ablation impact metric adapted to relation score R."""
    rel_drop = float(baseline_rel_acc - ablated_rel_acc)
    rel_impact_ratio = float(rel_drop / max(baseline_rel_acc, 1e-6))
    dx_drop = 0.0
    if baseline_dx is not None and ablated_dx is not None:
        dx_drop = float(baseline_dx - ablated_dx)
    elif baseline_dx is not None and ablated_dx is None:
        dx_drop = float(baseline_dx)
    return {
        "rel_acc_drop": rel_drop,
        "rel_impact_ratio": rel_impact_ratio,
        "dx_drop": dx_drop,
    }


def score_heads_revelio_probe(
    clean_caches: list[ActivationCache],
    corrupt_caches: list[ActivationCache],
    layers: list[int],
    steps: list[int],
    num_heads: int,
    hook_point: str = "attn2_head_pre_proj",
    seed: int = 42,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Revelio (Kim et al., 2025, arXiv:2411.16725) Diff-C-style linear probe on pooled activations.

    Trains an L2-regularized logistic probe per (layer, head) on spatially-pooled
    representations (averaged across steps) predicting clean (left) vs corrupt (right).
    Returns {(layer, head): (cv_roc_auc, cv_acc)}.
    """
    n_pairs = min(len(clean_caches), len(corrupt_caches))
    if n_pairs < 2:
        return {(l, h): (0.5, 0.5) for l in layers for h in range(num_heads)}

    out: dict[tuple[int, int], tuple[float, float]] = {}
    y = np.array([1] * n_pairs + [0] * n_pairs, dtype=np.int64)
    n_splits = min(5, n_pairs)

    for l_idx in layers:
        for h_idx in range(num_heads):
            feats_c = []
            feats_r = []
            for cache_c, cache_r in zip(clean_caches[:n_pairs], corrupt_caches[:n_pairs]):
                step_vecs_c = []
                step_vecs_r = []
                for s_idx in steps:
                    if cache_c.has(s_idx, l_idx, hook_point) and cache_r.has(s_idx, l_idx, hook_point):
                        tc = cache_c.get(s_idx, l_idx, hook_point)  # (1, N_img, H*Dh)
                        tr = cache_r.get(s_idx, l_idx, hook_point)
                        head_dim = tc.shape[-1] // num_heads
                        vc = tc.view(tc.shape[0], tc.shape[1], num_heads, head_dim)[0, :, h_idx, :].float().mean(dim=0)
                        vr = tr.view(tr.shape[0], tr.shape[1], num_heads, head_dim)[0, :, h_idx, :].float().mean(dim=0)
                        step_vecs_c.append(vc.numpy())
                        step_vecs_r.append(vr.numpy())
                if step_vecs_c and step_vecs_r:
                    feats_c.append(np.mean(step_vecs_c, axis=0))
                    feats_r.append(np.mean(step_vecs_r, axis=0))

            if len(feats_c) < 2:
                out[(l_idx, h_idx)] = (0.5, 0.5)
                continue

            x = np.vstack([np.stack(feats_c), np.stack(feats_r)])
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            oof_probs = np.zeros(len(y), dtype=np.float64)
            oof_preds = np.zeros(len(y), dtype=np.int64)
            for tr_idx, va_idx in skf.split(x, y):
                clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
                clf.fit(x[tr_idx], y[tr_idx])
                oof_probs[va_idx] = clf.predict_proba(x[va_idx])[:, 1]
                oof_preds[va_idx] = clf.predict(x[va_idx])

            auc = float(roc_auc_score(y, oof_probs))
            acc = float(accuracy_score(y, oof_preds))
            out[(l_idx, h_idx)] = (auc, acc)

    return out


def score_heads_from_pair_caches(
    transformer: PixArtTransformer2DModel,
    clean_cache: ActivationCache,
    corrupt_cache: ActivationCache,
    layers: list[int],
    steps: list[int],
    target_token_indices: list[int],
    boxes: list[DetectionBox | None] | None = None,
    eps: float = 1e-6,
) -> list[HeadObservationalScores]:
    """Compute Zarei (2025), Maliha & Hougen (2026), CIRCUIT spatial-grad, and DAAM (2022) head scores."""
    num_heads = int(transformer.config.num_attention_heads)
    results: list[HeadObservationalScores] = []

    for l_idx in layers:
        contrib_acc = np.zeros(num_heads, dtype=np.float64)
        kl_acc = np.zeros(num_heads, dtype=np.float64)
        grad_acc = np.zeros(num_heads, dtype=np.float64)
        daam_acc = np.zeros(num_heads, dtype=np.float64)
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

            valid_t = [t for t in target_token_indices if t < g_clean.shape[1]]
            if valid_t:
                contrib_acc += g_clean[:, valid_t].sum(dim=-1).numpy()

            dist_c = g_clean + eps
            dist_c = dist_c / dist_c.sum(dim=-1, keepdim=True)
            dist_r = g_corrupt + eps
            dist_r = dist_r / dist_r.sum(dim=-1, keepdim=True)
            kl = (dist_c * (dist_c.log() - dist_r.log())).sum(dim=-1).numpy()
            kl_acc += kl

            grad_acc += compute_spatial_gradient_score(p_clean, valid_t)
            if boxes is not None:
                daam_acc += compute_daam_head_box_scores(p_clean, valid_t, boxes)
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
                    daam_box_mass=float(daam_acc[h] / denom),
                )
            )
    return results
