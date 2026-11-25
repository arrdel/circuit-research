"""Unit tests for all 7 observational, probing, ablation, and P2P baselines."""

import torch
from diffusers import PixArtTransformer2DModel

from evaluation.spatial_eval import DetectionBox
from instrumentation.hooks import ActivationCache, DiTHookManager
from interpretability.baselines import (
    compute_daam_head_box_scores,
    compute_head_norm_distribution,
    compute_roy_ablation_impact,
    compute_spatial_gradient_score,
    score_heads_from_pair_caches,
    score_heads_revelio_probe,
)
from interventions.patching import AttnMapInjectionController


def _tiny_model() -> PixArtTransformer2DModel:
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
    return model


def test_spatial_gradient_daam_and_norm_distribution() -> None:
    model = _tiny_model()
    n_img = 16  # 4x4
    n_txt = 5
    probs = torch.full((1, 4, n_img, n_txt), 1.0 / n_txt)
    x_vals = torch.linspace(0.05, 0.95, 4).repeat(4)
    probs[0, 2, :, 1] = x_vals
    probs[0, 2] = probs[0, 2] / probs[0, 2].sum(dim=-1, keepdim=True)

    # Head 3 focuses exclusively on top-left corner (for DAAM box mass test)
    probs[0, 3, :, :] = 1e-4
    probs[0, 3, 0, 1] = 1.0
    probs[0, 3] = probs[0, 3] / probs[0, 3].sum(dim=-1, keepdim=True)

    grads = compute_spatial_gradient_score(probs, [1])
    assert grads.shape == (4,)
    assert grads[2] > 0.9

    box = DetectionBox("apple", 0.9, 0.0, 0.0, 0.35, 0.35)
    daam_scores = compute_daam_head_box_scores(probs, [1], [box], upsample_size=32)
    assert daam_scores.shape == (4,)
    assert daam_scores[3] > daam_scores[0]

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

    scores = score_heads_from_pair_caches(model, c_cache, r_cache, [0], [0], [1], boxes=[box])
    assert len(scores) == 4
    assert scores[2].spatial_gradient_abs > 0.9
    assert scores[3].daam_box_mass > scores[0].daam_box_mass


def test_roy_ablation_and_revelio_probe() -> None:
    impact = compute_roy_ablation_impact(1.0, 0.25, 0.4, -0.1)
    assert impact["rel_acc_drop"] == 0.75
    assert impact["rel_impact_ratio"] == 0.75
    assert abs(impact["dx_drop"] - 0.5) < 1e-6

    clean_caches = []
    corrupt_caches = []
    for i in range(6):
        c = ActivationCache()
        r = ActivationCache()
        tc = torch.zeros(1, 16, 32)
        tr = torch.zeros(1, 16, 32)
        # Separable signal in head 1 (channels 8:16)
        tc[:, :, 8:16] = 2.0 + 0.05 * i
        tr[:, :, 8:16] = -2.0 - 0.05 * i
        c.set(0, 0, "attn2_head_pre_proj", tc)
        r.set(0, 0, "attn2_head_pre_proj", tr)
        clean_caches.append(c)
        corrupt_caches.append(r)

    probe_scores = score_heads_revelio_probe(clean_caches, corrupt_caches, layers=[0], steps=[0], num_heads=4)
    auc_h1, acc_h1 = probe_scores[(0, 1)]
    assert auc_h1 > 0.95
    assert acc_h1 > 0.95


def test_p2p_attn_map_injection_controller() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([500, 500])
    enc_c = torch.cat([torch.zeros(1, 6, 32), torch.randn(1, 6, 32)], dim=0)
    enc_r = torch.cat([torch.zeros(1, 6, 32), torch.randn(1, 6, 32)], dim=0)
    mask = torch.ones(2, 6)
    added = {"resolution": None, "aspect_ratio": None}

    mgr = DiTHookManager(model, layers=[0, 1], hook_points=[], steps=[0], capture_attn2_probs=True)
    with torch.no_grad(), mgr.active() as c_cache:
        mgr.on_step_start(0, t[0])
        _ = model(x, encoder_hidden_states=enc_c, encoder_attention_mask=mask, timestep=t, added_cond_kwargs=added, return_dict=False)
        mgr.on_step_end(0, t[0], x)

    with torch.no_grad():
        out_corrupt = model(x, encoder_hidden_states=enc_r, encoder_attention_mask=mask, timestep=t, added_cond_kwargs=added, return_dict=False)[0]

    p2p = AttnMapInjectionController(model, layers=[0], steps=[0], source_cache=c_cache, branch="cond")
    with torch.no_grad(), p2p.active():
        p2p.on_step_start(0, t[0])
        out_p2p = model(x, encoder_hidden_states=enc_r, encoder_attention_mask=mask, timestep=t, added_cond_kwargs=added, return_dict=False)[0]
        p2p.on_step_end(0, t[0], x)

    assert torch.allclose(out_p2p[0], out_corrupt[0], atol=1e-5)
    assert not torch.allclose(out_p2p[1], out_corrupt[1], atol=1e-4)
