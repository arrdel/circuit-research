"""Unit tests for observational baseline head scorers."""

import torch
from diffusers import PixArtTransformer2DModel

from instrumentation.hooks import ActivationCache
from interpretability.baselines import (
    compute_head_norm_distribution,
    compute_spatial_gradient_score,
    score_heads_from_pair_caches,
)


def test_spatial_gradient_and_norm_distribution() -> None:
    model = PixArtTransformer2DModel(
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=4,
        out_channels=8,
        num_layers=2,
        cross_attention_dim=32,
        sample_size=8,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=32,
        use_additional_conditions=False,
    )
    model.eval()

    # Build a synthetic attention map where head 2 tracks horizontal x-coordinate on token 1
    n_img = 16  # 4x4
    n_txt = 5
    probs = torch.full((1, 4, n_img, n_txt), 1.0 / n_txt)
    x_vals = torch.linspace(0.05, 0.95, 4).repeat(4)
    probs[0, 2, :, 1] = x_vals
    probs[0, 2] = probs[0, 2] / probs[0, 2].sum(dim=-1, keepdim=True)

    grads = compute_spatial_gradient_score(probs, [1])
    assert grads.shape == (4,)
    assert grads[2] > 0.9

    enc_c = torch.randn(1, n_txt, 32)
    enc_r = torch.randn(1, n_txt, 32)
    g = compute_head_norm_distribution(model, 0, probs, enc_c)
    assert g.shape == (4, n_txt)
    assert torch.all(g >= 0)

    c_cache = ActivationCache()
    r_cache = ActivationCache()
    c_cache.attn_probs[(0, 0)] = probs
    r_cache.attn_probs[(0, 0)] = probs
    c_cache.set(0, 0, "attn2_in_txt", enc_c)
    r_cache.set(0, 0, "attn2_in_txt", enc_r)

    scores = score_heads_from_pair_caches(model, c_cache, r_cache, [0], [0], [1])
    assert len(scores) == 4
    assert scores[2].spatial_gradient_abs > 0.9
