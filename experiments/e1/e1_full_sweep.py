"""Day 3 Task 1 & 3: Full E1 Spatial Left/Right Sweep on Test Split (512-MS)."""

from __future__ import annotations

import argparse
import json
import platform
import random
import socket
import time
from pathlib import Path

import diffusers
import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image

from benchmark.counterfactual_pairs import CounterfactualPair, build_spatial_lr_pairs
from evaluation.causal_metrics import (
    PairCausalOutcome,
    compute_pair_causal_outcome,
    summarize_causal_outcomes,
)
from evaluation.spatial_eval import ImageEvalResult, SpatialColorEvaluator
from instrumentation.hooks import ActivationCache, DiTHookManager, HookPoint
from interventions.patching import InterventionController, InterventionSpec
from models.pixart_sigma import DEFAULT_MODEL_ID, PixArtSigmaRunner
from visualization.heatmaps import plot_layer_timestep_heatmap, save_counterfactual_grid


WINDOWS: list[tuple[str, list[int]]] = [
    ("W0[0-3]", [0, 1, 2, 3]),
    ("W1[4-7]", [4, 5, 6, 7]),
    ("W2[8-11]", [8, 9, 10, 11]),
    ("W3[12-15]", [12, 13, 14, 15]),
    ("W4[16-19]", [16, 17, 18, 19]),
]

SEEDS = [42, 123, 456]


def _select_test_instances(
    runner: PixArtSigmaRunner,
    encoded: dict,
    evaluator: SpatialColorEvaluator,
    candidates: list[CounterfactualPair],
    target_pairs: int = 6,
    steps: int = 20,
    guidance: float = 4.5,
) -> list[tuple[CounterfactualPair, int]]:
    neg_entry = encoded[""]
    chosen: list[CounterfactualPair] = []
    for pair in candidates:
        if len(chosen) >= target_pairs:
            break
        valid_seeds = 0
        for sd in SEEDS:
            res_c = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=sd,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            ev_c = evaluator.evaluate_image(
                res_c.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_clean,
                color_b=pair.color_b_clean,
                expected_relation=pair.relation_clean,
                compute_embedding=False,
            )
            res_r = runner.generate(
                prompt_entry=encoded[pair.corrupt_prompt],
                negative_entry=neg_entry,
                seed=sd,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            ev_r = evaluator.evaluate_image(
                res_r.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_corrupt,
                color_b=pair.color_b_corrupt,
                expected_relation=pair.relation_corrupt,
                compute_embedding=False,
            )
            if (
                ev_c.dx is not None
                and ev_r.dx is not None
                and ev_c.relation_pred == pair.relation_clean
                and ev_r.relation_pred == pair.relation_corrupt
                and (ev_c.dx - ev_r.dx) > 0.04
            ):
                valid_seeds += 1
        if valid_seeds >= 2:
            chosen.append(pair)
            print(f"[E1 Full] Selected test pair {pair.pair_id} ({valid_seeds}/3 valid seeds)")
    return [(p, sd) for p in chosen for sd in SEEDS]


def run_e1_full(
    model_id: str = DEFAULT_MODEL_ID,
    hook_point: HookPoint = "attn2_out",
    num_pairs: int = 6,
    layer_group_size: int = 2,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
    out_dir: str = "experiments/e1",
) -> dict:
    t0 = time.time()
    out_path = Path(out_dir)
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    all_explicit = build_spatial_lr_pairs(
        num_dev=24,
        num_test=36,
        num_ood=12,
        templates=("on_the_side_explicit",),
        base_seed=3000,
    )
    test_candidates = [p for p in all_explicit if p.split == "test"]

    prompts = []
    for p in test_candidates:
        prompts.extend([p.clean_prompt, p.corrupt_prompt])

    runner = PixArtSigmaRunner(model_id=model_id, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    instances = _select_test_instances(
        runner,
        encoded,
        evaluator,
        test_candidates,
        target_pairs=num_pairs,
        steps=num_inference_steps,
        guidance=guidance_scale,
    )
    if not instances:
        raise RuntimeError("No valid test pairs selected.")

    num_layers = runner.num_layers
    num_heads = runner.num_heads
    layer_groups = [
        (
            f"L{s}-{min(num_layers - 1, s + layer_group_size - 1)}",
            list(range(s, min(num_layers, s + layer_group_size))),
        )
        for s in range(0, num_layers, layer_group_size)
    ]

    clean_caches: dict[str, ActivationCache] = {}
    corrupt_caches: dict[str, ActivationCache] = {}
    shifted_clean_caches: dict[str, ActivationCache] = {}
    clean_evals: dict[str, ImageEvalResult] = {}
    corrupt_evals: dict[str, ImageEvalResult] = {}
    clean_imgs: dict[str, Image.Image] = {}
    corrupt_imgs: dict[str, Image.Image] = {}

    print(f"[E1 Full] Caching activations for {len(instances)} test (pair, seed) instances...")
    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        mgr_c = DiTHookManager(
            runner.transformer,
            layers=list(range(num_layers)),
            hook_points=[hook_point, "attn2_head_pre_proj"],
            steps=list(range(num_inference_steps)),
            branch="cond",
            store_cpu=True,
        )
        with mgr_c.active() as c_cache:
            res_c = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=sd,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                step_callbacks=[mgr_c],
            )
            mgr_c.cache = ActivationCache()

        mgr_r = DiTHookManager(
            runner.transformer,
            layers=list(range(num_layers)),
            hook_points=[hook_point],
            steps=list(range(num_inference_steps)),
            branch="cond",
            store_cpu=True,
        )
        with mgr_r.active() as r_cache:
            res_r = runner.generate(
                prompt_entry=encoded[pair.corrupt_prompt],
                negative_entry=neg_entry,
                seed=sd,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                step_callbacks=[mgr_r],
            )
            mgr_r.cache = ActivationCache()

        mgr_s = DiTHookManager(
            runner.transformer,
            layers=list(range(num_layers)),
            hook_points=[hook_point],
            steps=list(range(num_inference_steps)),
            branch="cond",
            store_cpu=True,
        )
        with mgr_s.active() as s_cache:
            _ = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=sd + 999,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                step_callbacks=[mgr_s],
                decode_image=False,
            )
            mgr_s.cache = ActivationCache()

        clean_caches[inst_id] = c_cache
        corrupt_caches[inst_id] = r_cache
        shifted_clean_caches[inst_id] = s_cache
        clean_imgs[inst_id] = res_c.image
        corrupt_imgs[inst_id] = res_r.image
        clean_evals[inst_id] = evaluator.evaluate_image(
            res_c.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )
        corrupt_evals[inst_id] = evaluator.evaluate_image(
            res_r.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_corrupt,
            color_b=pair.color_b_corrupt,
            expected_relation=pair.relation_corrupt,
        )

    # 1. Layer-group x Timestep-window sweep
    tau_mat = np.zeros((len(layer_groups), len(WINDOWS)), dtype=np.float64)
    sigma_mat = np.zeros((len(layer_groups), len(WINDOWS)), dtype=np.float64)
    flip_mat = np.zeros((len(layer_groups), len(WINDOWS)), dtype=np.float64)
    det_mat = np.zeros((len(layer_groups), len(WINDOWS)), dtype=np.float64)
    cell_records: list[dict] = []

    best_score = -1.0
    best_li, best_wi = 0, 0
    best_imgs: dict[str, Image.Image] = {}

    for li, (l_lbl, l_idxs) in enumerate(layer_groups):
        for wi, (w_lbl, s_idxs) in enumerate(WINDOWS):
            outcomes: list[PairCausalOutcome] = []
            cell_imgs: dict[str, Image.Image] = {}
            for pair, sd in instances:
                inst_id = f"{pair.pair_id}_s{sd}"
                spec = InterventionSpec(
                    kind="patch",
                    hook_point=hook_point,
                    layers=l_idxs,
                    steps=s_idxs,
                    source_cache=clean_caches[inst_id],
                    branch="cond",
                )
                ctrl = InterventionController(runner.transformer, [spec])
                with ctrl.active():
                    res_p = runner.generate(
                        prompt_entry=encoded[pair.corrupt_prompt],
                        negative_entry=neg_entry,
                        seed=sd,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        step_callbacks=[ctrl],
                    )
                cell_imgs[inst_id] = res_p.image
                ev_p = evaluator.evaluate_image(
                    res_p.image,
                    obj_a=pair.obj_a,
                    obj_b=pair.obj_b,
                    color_a=pair.color_a_corrupt,
                    color_b=pair.color_b_corrupt,
                    expected_relation=pair.relation_clean,
                )
                outcomes.append(
                    compute_pair_causal_outcome(
                        pair_id=inst_id,
                        task="spatial_lr",
                        clean=clean_evals[inst_id],
                        corrupt=corrupt_evals[inst_id],
                        patched=ev_p,
                        clean_relation=pair.relation_clean,
                        corrupt_relation=pair.relation_corrupt,
                    )
                )

            summ = summarize_causal_outcomes(outcomes, valid_only=True)
            tau_v = summ.tau_mean if summ.tau_mean is not None else 0.0
            sig_v = summ.sigma_mean if summ.sigma_mean is not None else 0.0
            flip_v = summ.flip_rate if summ.flip_rate is not None else 0.0
            det_v = summ.both_detected_rate

            tau_mat[li, wi] = tau_v
            sigma_mat[li, wi] = sig_v
            flip_mat[li, wi] = flip_v
            det_mat[li, wi] = det_v

            cell_records.append(
                {
                    "layer_group": l_lbl,
                    "layers": l_idxs,
                    "head": "all",
                    "window": w_lbl,
                    "steps": s_idxs,
                    "tau": tau_v,
                    "tau_ci_low": summ.tau_ci_low,
                    "tau_ci_high": summ.tau_ci_high,
                    "sigma": sig_v,
                    "sigma_ci_low": summ.sigma_ci_low,
                    "sigma_ci_high": summ.sigma_ci_high,
                    "flip_rate": flip_v,
                    "detection_rate": det_v,
                    "color_preserve_rate": summ.color_preserve_rate,
                    "n_valid": summ.n_valid,
                    "n_total": summ.n_total,
                }
            )
            print(
                f"[E1 Full] {l_lbl:>7} x {w_lbl:<9} | tau={tau_v:.3f} "
                f"[{summ.tau_ci_low:.3f},{summ.tau_ci_high:.3f}] flip={flip_v:.2f} "
                f"sigma={sig_v:.3f} det={det_v:.2f}"
            )
            score = (tau_v * sig_v * det_v) if det_v >= 0.75 else -1.0
            if score > best_score:
                best_score = score
                best_li, best_wi = li, wi
                best_imgs = cell_imgs

    peak_l_lbl, peak_layers = layer_groups[best_li]
    peak_w_lbl, peak_steps = WINDOWS[best_wi]

    # Save sample patched images for first 6 instances
    for pair, sd in instances[:6]:
        inst_id = f"{pair.pair_id}_s{sd}"
        clean_imgs[inst_id].save(img_dir / f"full_{inst_id}_clean.png")
        corrupt_imgs[inst_id].save(img_dir / f"full_{inst_id}_corrupt.png")
        best_imgs[inst_id].save(img_dir / f"full_{inst_id}_peak_{peak_l_lbl}_{peak_w_lbl}.png")

    # 2. Per-(l, h, t) head breakdown at peak_layers AND at L2-3 (if peak is not L2-3)
    head_layers = sorted(set(peak_layers + [2, 3]))
    head_records: list[dict] = []
    head_tau_mat = np.zeros((len(head_layers), num_heads), dtype=np.float64)
    for li, l_idx in enumerate(head_layers):
        for h_idx in range(num_heads):
            outcomes = []
            for pair, sd in instances:
                inst_id = f"{pair.pair_id}_s{sd}"
                spec = InterventionSpec(
                    kind="patch",
                    hook_point="attn2_head_pre_proj",
                    layers=[l_idx],
                    steps=peak_steps,
                    heads=[h_idx],
                    source_cache=clean_caches[inst_id],
                    branch="cond",
                )
                ctrl = InterventionController(runner.transformer, [spec])
                with ctrl.active():
                    res_h = runner.generate(
                        prompt_entry=encoded[pair.corrupt_prompt],
                        negative_entry=neg_entry,
                        seed=sd,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        step_callbacks=[ctrl],
                    )
                ev_h = evaluator.evaluate_image(
                    res_h.image,
                    obj_a=pair.obj_a,
                    obj_b=pair.obj_b,
                    color_a=pair.color_a_corrupt,
                    color_b=pair.color_b_corrupt,
                    expected_relation=pair.relation_clean,
                )
                outcomes.append(
                    compute_pair_causal_outcome(
                        pair_id=inst_id,
                        task="spatial_lr",
                        clean=clean_evals[inst_id],
                        corrupt=corrupt_evals[inst_id],
                        patched=ev_h,
                        clean_relation=pair.relation_clean,
                        corrupt_relation=pair.relation_corrupt,
                    )
                )
            summ = summarize_causal_outcomes(outcomes, valid_only=True)
            tau_v = summ.tau_mean if summ.tau_mean is not None else 0.0
            head_tau_mat[li, h_idx] = tau_v
            head_records.append(
                {
                    "layer": l_idx,
                    "head": h_idx,
                    "window": peak_w_lbl,
                    "steps": peak_steps,
                    "tau": tau_v,
                    "tau_ci_low": summ.tau_ci_low,
                    "tau_ci_high": summ.tau_ci_high,
                    "sigma": summ.sigma_mean if summ.sigma_mean is not None else 0.0,
                    "flip_rate": summ.flip_rate if summ.flip_rate is not None else 0.0,
                    "detection_rate": summ.both_detected_rate,
                }
            )
            print(
                f"[E1 Full Head] L{l_idx:02d}H{h_idx:02d}@{peak_w_lbl} | tau={tau_v:.3f} "
                f"flip={head_records[-1]['flip_rate']:.2f} sigma={head_records[-1]['sigma']:.3f}"
            )

    # 3. Four controls at peak cell
    rng = random.Random(2026)
    rand_layer = rng.choice(list(range(num_layers)))
    rand_head = rng.randint(0, num_heads - 1)
    wrong_steps = WINDOWS[-1][1] if best_wi != len(WINDOWS) - 1 else WINDOWS[0][1]
    wrong_w_lbl = WINDOWS[-1][0] if best_wi != len(WINDOWS) - 1 else WINDOWS[0][0]

    control_defs = {
        "random_head": {
            "desc": f"Patch single random head L{rand_layer}H{rand_head} at {peak_w_lbl}",
            "hook_point": "attn2_head_pre_proj",
            "layers": [rand_layer],
            "steps": peak_steps,
            "heads": [rand_head],
            "cache_src": "clean",
        },
        "wrong_timestep": {
            "desc": f"Patch {peak_l_lbl} at late window {wrong_w_lbl}",
            "hook_point": hook_point,
            "layers": peak_layers,
            "steps": wrong_steps,
            "heads": None,
            "cache_src": "clean",
        },
        "same_prompt_noop": {
            "desc": f"Patch corrupt cache back into corrupt at {peak_l_lbl}@{peak_w_lbl}",
            "hook_point": hook_point,
            "layers": peak_layers,
            "steps": peak_steps,
            "heads": None,
            "cache_src": "corrupt",
        },
        "seed_variation": {
            "desc": f"Patch clean cache from shifted seed (s+999) into corrupt at {peak_l_lbl}@{peak_w_lbl}",
            "hook_point": hook_point,
            "layers": peak_layers,
            "steps": peak_steps,
            "heads": None,
            "cache_src": "shifted_clean",
        },
    }

    control_results: dict[str, dict] = {}
    control_imgs: dict[str, dict[str, Image.Image]] = {}
    for cname, cfg in control_defs.items():
        outcomes = []
        control_imgs[cname] = {}
        for pair, sd in instances:
            inst_id = f"{pair.pair_id}_s{sd}"
            if cfg["cache_src"] == "clean":
                src_cache = clean_caches[inst_id]
            elif cfg["cache_src"] == "corrupt":
                src_cache = corrupt_caches[inst_id]
            else:
                src_cache = shifted_clean_caches[inst_id]

            spec = InterventionSpec(
                kind="patch",
                hook_point=cfg["hook_point"],  # type: ignore[arg-type]
                layers=cfg["layers"],  # type: ignore[arg-type]
                steps=cfg["steps"],  # type: ignore[arg-type]
                heads=cfg["heads"],  # type: ignore[arg-type]
                source_cache=src_cache,
                branch="cond",
            )
            ctrl = InterventionController(runner.transformer, [spec])
            with ctrl.active():
                res_ctrl = runner.generate(
                    prompt_entry=encoded[pair.corrupt_prompt],
                    negative_entry=neg_entry,
                    seed=sd,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    step_callbacks=[ctrl],
                )
            control_imgs[cname][inst_id] = res_ctrl.image
            ev_ctrl = evaluator.evaluate_image(
                res_ctrl.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_corrupt,
                color_b=pair.color_b_corrupt,
                expected_relation=pair.relation_clean,
            )
            outcomes.append(
                compute_pair_causal_outcome(
                    pair_id=inst_id,
                    task="spatial_lr",
                    clean=clean_evals[inst_id],
                    corrupt=corrupt_evals[inst_id],
                    patched=ev_ctrl,
                    clean_relation=pair.relation_clean,
                    corrupt_relation=pair.relation_corrupt,
                )
            )
        summ = summarize_causal_outcomes(outcomes, valid_only=True)
        control_results[cname] = {"description": cfg["desc"], **summ.to_dict()}
        print(
            f"[E1 Full Control] {cname:<18} | tau={summ.tau_mean} flip={summ.flip_rate} "
            f"sigma={summ.sigma_mean} color_pres={summ.color_preserve_rate}"
        )

    # Save figures
    plot_layer_timestep_heatmap(
        tau_mat,
        layers=[g[1][0] for g in layer_groups],
        step_labels=[w[0] for w in WINDOWS],
        title=f"E1 Full Sweep Spatial Transfer (tau) [{hook_point}, 512-MS, test split]",
        cbar_label="Transfer Score (tau)",
        out_path=out_path / "e1_tau_map.png",
        cmap="magma",
    )
    plot_layer_timestep_heatmap(
        head_tau_mat,
        layers=head_layers,
        step_labels=[f"H{h}" for h in range(num_heads)],
        title=f"E1 Full Sweep Per-Head Tau @ {peak_w_lbl} [512-MS]",
        cbar_label="Transfer Score (tau)",
        out_path=img_dir / "full_head_tau.png",
        cmap="magma",
    )

    grid_rows = []
    for pair, sd in instances[:6]:
        inst_id = f"{pair.pair_id}_s{sd}"
        grid_rows.append(
            (
                inst_id,
                [
                    ("Clean (L)", clean_imgs[inst_id]),
                    ("Corrupt (R)", corrupt_imgs[inst_id]),
                    (f"Peak {peak_l_lbl}@{peak_w_lbl}", best_imgs[inst_id]),
                    ("RandHead", control_imgs["random_head"][inst_id]),
                    ("WrongStep", control_imgs["wrong_timestep"][inst_id]),
                    ("SeedVar", control_imgs["seed_variation"][inst_id]),
                ],
            )
        )
    save_counterfactual_grid(grid_rows, out_path / "e1_full_grid.png", tile_size=224)

    l23_rec = next((r for r in cell_records if r["layer_group"] == "L2-3" and r["window"] == peak_w_lbl), None)
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    results = {
        "checkpoint": model_id,
        "checkpoint_label": "512-MS" if "512-MS" in model_id else "1024-MS",
        "split": "test",
        "host": socket.gethostname(),
        "gpus": gpu_names,
        "hook_point": hook_point,
        "pairs": [
            {
                "pair_id": p.pair_id,
                "clean_prompt": p.clean_prompt,
                "corrupt_prompt": p.corrupt_prompt,
            }
            for p, _ in instances[:: len(SEEDS)]
        ],
        "seeds": SEEDS,
        "n_instances": len(instances),
        "n_valid_counterfactual": cell_records[0]["n_valid"] if cell_records else 0,
        "peak_cell": {
            "layer_group": peak_l_lbl,
            "layers": peak_layers,
            "window": peak_w_lbl,
            "steps": peak_steps,
            "tau": float(tau_mat[best_li, best_wi]),
            "sigma": float(sigma_mat[best_li, best_wi]),
            "flip_rate": float(flip_mat[best_li, best_wi]),
            "detection_rate": float(det_mat[best_li, best_wi]),
        },
        "l2_3_replication": l23_rec,
        "layer_timestep_cells": cell_records,
        "per_l_h_t": head_records,
        "controls": control_results,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with (out_path / "e1_full_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log_lines = [
        "# E1 Full Sweep Run Log",
        "",
        "## Execution Environment",
        "",
        f"- Host: `{socket.gethostname()}`",
        f"- GPUs visible: `{gpu_names}` (gen: `{gen_device}`, eval: `{eval_device}`)",
        f"- Python: `{platform.python_version()}`",
        f"- Torch: `{torch.__version__}`, Diffusers: `{diffusers.__version__}`, Transformers: `{transformers.__version__}`",
        "",
        "## Model and Generation Settings",
        "",
        f"- Checkpoint: `{model_id}` (`{results['checkpoint_label']}`)",
        "- Split: `test` (explicit template `on_the_side_explicit`, disjoint objects from `dev`)",
        "- Sampler: `DPMSolverMultistepScheduler`",
        f"- Steps: `{num_inference_steps}`",
        f"- Guidance scale: `{guidance_scale}`",
        "- Precision: `torch.bfloat16`",
        f"- Hook point: `{hook_point}` (conditional CFG branch only)",
        f"- Seeds: `{SEEDS}`",
        f"- Total (pair, seed) instances: `{len(instances)}` (valid counterfactuals: `{results['n_valid_counterfactual']}`)",
        "",
        "## Test Pairs",
        "",
    ]
    for p_info in results["pairs"]:
        log_lines.append(
            f"- `{p_info['pair_id']}`: Clean = `{p_info['clean_prompt']}` | Corrupt = `{p_info['corrupt_prompt']}`"
        )

    log_lines.extend(
        [
            "",
            "## Summary Findings",
            "",
            f"- Peak cell: `{peak_l_lbl} @ {peak_w_lbl}` | `tau = {results['peak_cell']['tau']:.4f}`, `sigma = {results['peak_cell']['sigma']:.4f}`, `flip_rate = {results['peak_cell']['flip_rate']:.4f}`, `detection_rate = {results['peak_cell']['detection_rate']:.4f}`",
            f"- L2-3 @ {peak_w_lbl} replication: `tau = {l23_rec['tau']:.4f}` (`flip_rate = {l23_rec['flip_rate']:.4f}`, `sigma = {l23_rec['sigma']:.4f}`)" if l23_rec else "- L2-3 replication: N/A",
            "",
            "## Controls",
            "",
            "| Control | Description | Tau | Flip Rate | Sigma | Color Preserve |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for cname, cdata in control_results.items():
        tau_s = f"{cdata['tau_mean']:.4f}" if cdata["tau_mean"] is not None else "N/A"
        fl_s = f"{cdata['flip_rate']:.4f}" if cdata["flip_rate"] is not None else "N/A"
        sg_s = f"{cdata['sigma_mean']:.4f}" if cdata["sigma_mean"] is not None else "N/A"
        cp_s = f"{cdata['color_preserve_rate']:.4f}" if cdata["color_preserve_rate"] is not None else "N/A"
        log_lines.append(f"| {cname} | {cdata['description']} | {tau_s} | {fl_s} | {sg_s} | {cp_s} |")

    (out_path / "e1_full_run_log.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[E1 Full] DONE in {results['elapsed_sec']}s.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 full sweep on test split")
    parser.add_argument("--num-pairs", type=int, default=6)
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--out-dir", default="experiments/e1")
    args = parser.parse_args()
    run_e1_full(
        num_pairs=args.num_pairs,
        gen_device=args.gen_device,
        eval_device=args.eval_device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
