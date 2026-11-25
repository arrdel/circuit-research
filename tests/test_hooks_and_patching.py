"""Unit tests for DiTHookManager and InterventionController on a tiny PixArtTransformer2DModel."""

import torch
from diffusers import PixArtTransformer2DModel

from instrumentation.hooks import ActivationCache, DiTHookManager
from interventions.patching import InterventionController, InterventionSpec


def _tiny_pixart() -> PixArtTransformer2DModel:
    model = PixArtTransformer2DModel(
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=4,
        out_channels=8,
        num_layers=3,
        cross_attention_dim=32,
        sample_size=8,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=32,
        use_additional_conditions=False,
    )
    model.eval()
    return model


def test_capture_and_selective_cond_branch_patching() -> None:
    torch.manual_seed(0)
    model = _tiny_pixart()

    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([500, 500])
    enc_clean = torch.cat([torch.zeros(1, 6, 32), torch.randn(1, 6, 32)], dim=0)
    enc_corrupt = torch.cat([torch.zeros(1, 6, 32), torch.randn(1, 6, 32)], dim=0)
    mask = torch.ones(2, 6)
    added = {"resolution": None, "aspect_ratio": None}

    hook_mgr = DiTHookManager(
        model,
        layers=[0, 1, 2],
        hook_points=[
            "block_out",
            "attn1_out",
            "attn2_out",
            "ff_out",
            "attn2_head_pre_proj",
            "attn2_in_txt",
        ],
        steps=[0],
        branch="cond",
        store_cpu=True,
        capture_attn2_probs=True,
    )

    with torch.no_grad(), hook_mgr.active() as clean_cache:
        hook_mgr.on_step_start(0, t[0])
        out_clean = model(
            x,
            encoder_hidden_states=enc_clean,
            encoder_attention_mask=mask,
            timestep=t,
            added_cond_kwargs=added,
            return_dict=False,
        )[0]
        hook_mgr.on_step_end(0, t[0], x)

    assert clean_cache.has(0, 1, "block_out")
    assert clean_cache.has(0, 1, "attn2_head_pre_proj")
    assert clean_cache.has(0, 1, "attn2_in_txt")
    assert (0, 1) in clean_cache.attn_probs
    assert clean_cache.attn_probs[(0, 1)].shape == (1, 4, 16, 6)

    with torch.no_grad():
        out_corrupt = model(
            x,
            encoder_hidden_states=enc_corrupt,
            encoder_attention_mask=mask,
            timestep=t,
            added_cond_kwargs=added,
            return_dict=False,
        )[0]

    # Patching block_out at layer 1 on cond branch must leave uncond branch untouched and change cond branch
    spec = InterventionSpec(
        kind="patch",
        hook_point="block_out",
        layers=[1],
        steps=[0],
        source_cache=clean_cache,
        branch="cond",
    )
    ctrl = InterventionController(model, [spec])
    with torch.no_grad(), ctrl.active():
        ctrl.on_step_start(0, t[0])
        out_patched = model(
            x,
            encoder_hidden_states=enc_corrupt,
            encoder_attention_mask=mask,
            timestep=t,
            added_cond_kwargs=added,
            return_dict=False,
        )[0]
        ctrl.on_step_end(0, t[0], x)

    assert torch.allclose(out_patched[0], out_corrupt[0], atol=1e-5)
    assert not torch.allclose(out_patched[1], out_corrupt[1], atol=1e-4)

    # Head-level patching & random norm-matched control should run cleanly
    head_spec = InterventionSpec(
        kind="patch",
        hook_point="attn2_head_pre_proj",
        layers=[1],
        steps=[0],
        heads=[0, 2],
        source_cache=clean_cache,
        branch="cond",
    )
    rand_spec = InterventionSpec(
        kind="random_norm_matched",
        hook_point="block_out",
        layers=[2],
        steps=[0],
        branch="cond",
        seed=123,
    )
    ctrl2 = InterventionController(model, [head_spec, rand_spec])
    with torch.no_grad(), ctrl2.active():
        ctrl2.on_step_start(0, t[0])
        out_ctrl2 = model(
            x,
            encoder_hidden_states=enc_corrupt,
            encoder_attention_mask=mask,
            timestep=t,
            added_cond_kwargs=added,
            return_dict=False,
        )[0]
        ctrl2.on_step_end(0, t[0], x)
    assert out_ctrl2.shape == out_clean.shape

    # Cache averaging for mean ablation
    mean_cache = ActivationCache.average([clean_cache, clean_cache])
    assert torch.allclose(
        mean_cache.get(0, 1, "block_out").float(),
        clean_cache.get(0, 1, "block_out").float(),
        atol=1e-5,
    )
