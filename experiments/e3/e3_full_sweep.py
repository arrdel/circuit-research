"""Day 6 Task 1: Full E3 Color-Binding Sweep on fresh test-split pairs (512-MS)."""

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
import torch
import transformers
from PIL import Image

from benchmark.counterfactual_pairs import CounterfactualPair, build_color_binding_pairs
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


SEEDS = [42, 123, 456]
PEAK_STEPS = [4, 5, 6, 7]
WRONG_STEPS = [16, 17, 18, 19]


def _select_color_test_pairs(
    runner: PixArtSigmaRunner,
    encoded: dict,
    evaluator: SpatialColorEvaluator,
    candidates: list[CounterfactualPair],
    target_num_pairs: int = 6,
    steps: int = 20,
    guidance: float = 4.5,
) -> list[tuple[CounterfactualPair, int]]:
    neg_entry = encoded[""]
    chosen: list[CounterfactualPair] = []

    for pair in candidates:
        if len(chosen) >= target_num_pairs:
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
            if not (
                ev_c.both_detected
                and ev_c.color_binding_correct
                and ev_r.both_detected
                and ev_r.color_binding_correct
            ):
                continue
            ev_r_in_clean = evaluator.evaluate_image(
                res_r.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_clean,
                color_b=pair.color_b_clean,
                expected_relation=pair.relation_clean,
                compute_embedding=False,
            )
            mc = (ev_c.color_a_margin or 0.0) + (ev_c.color_b_margin or 0.0)
            mr = (ev_r_in_clean.color_a_margin or 0.0) + (ev_r_in_clean.color_b_margin or 0.0)
            if (mc - mr) > 0.02:
                valid_seeds += 1
        if valid_seeds >= 2:
            chosen.append(pair)
            print(f"[E3 Full] Selected fresh test pair {pair.pair_id} ({valid_seeds}/3 valid seeds)")

    return [(p, sd) for p in chosen for sd in SEEDS]


def run_e3_full(
    model_id: str = DEFAULT_MODEL_ID,
    hook_point: HookPoint = "attn2_out",
    num_pairs: int = 6,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
    out_dir: str = "experiments/e3",
) -> dict:
    t0 = time.time()
    out_path = Path(out_dir)
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    all_color = build_color_binding_pairs(
        num_dev=40,
        num_test=40,
        num_ood=12,
        templates=("fixed_lr_swap_colors",),
        base_seed=6000,
    )
    test_candidates = [p for p in all_color if p.split == "test"]
    prompts = [p.clean_prompt for p in test_candidates] + [p.corrupt_prompt for p in test_candidates]

    runner = PixArtSigmaRunner(model_id=model_id, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    instances = _select_color_test_pairs(
        runner,
        encoded,
        evaluator,
        test_candidates,
        target_num_pairs=num_pairs,
        steps=num_inference_steps,
        guidance=guidance_scale,
    )
    if not instances:
        raise RuntimeError("No valid fresh test color-binding pairs found.")

    num_layers = runner.num_layers
    num_heads = runner.num_heads
    target_layers = [2, 3, 10, 11]

    clean_caches: dict[str, ActivationCache] = {}
    corrupt_caches: dict[str, ActivationCache] = {}
    shifted_caches: dict[str, ActivationCache] = {}
    clean_evals: dict[str, ImageEvalResult] = {}
    corrupt_evals: dict[str, ImageEvalResult] = {}
    clean_imgs: dict[str, Image.Image] = {}
    corrupt_imgs: dict[str, Image.Image] = {}

    print(f"[E3 Full] Caching activations for {len(instances)} fresh test instances...")
    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        mgr_c = DiTHookManager(
            runner.transformer,
            layers=list(range(num_layers)),
            hook_points=[hook_point, "attn2_head_pre_proj"],
            steps=PEAK_STEPS + WRONG_STEPS,
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
            layers=target_layers,
            hook_points=[hook_point],
            steps=PEAK_STEPS,
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
            layers=target_layers,
            hook_points=[hook_point],
            steps=PEAK_STEPS,
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
        shifted_caches[inst_id] = s_cache
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
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )

    # 1. Test E1 peak cells on fresh test-split color-binding pairs
    candidate_cells = [
        ("L10-11", [10, 11], "W1[4-7]", PEAK_STEPS),
        ("L2-3", [2, 3], "W1[4-7]", PEAK_STEPS),
    ]
    cell_results: list[dict] = []
    cell_imgs: dict[str, dict[str, Image.Image]] = {}

    for l_lbl, l_idxs, w_lbl, s_idxs in candidate_cells:
        outcomes: list[PairCausalOutcome] = []
        cell_imgs[l_lbl] = {}
        spatial_preserved_list: list[float] = []

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
            cell_imgs[l_lbl][inst_id] = res_p.image
            ev_p = evaluator.evaluate_image(
                res_p.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_clean,
                color_b=pair.color_b_clean,
                expected_relation=pair.relation_clean,
            )
            outc = compute_pair_causal_outcome(
                pair_id=inst_id,
                task="color_binding",
                clean=clean_evals[inst_id],
                corrupt=corrupt_evals[inst_id],
                patched=ev_p,
                clean_relation=pair.relation_clean,
                corrupt_relation=pair.relation_corrupt,
            )
            outcomes.append(outc)
            if outc.valid_counterfactual:
                spatial_preserved_list.append(
                    1.0 if ev_p.relation_pred == corrupt_evals[inst_id].relation_pred else 0.0
                )

        summ = summarize_causal_outcomes(outcomes, valid_only=True)
        rec = {
            "layer_group": l_lbl,
            "layers": l_idxs,
            "window": w_lbl,
            "steps": s_idxs,
            "tau": summ.tau_mean if summ.tau_mean is not None else 0.0,
            "tau_ci_low": summ.tau_ci_low,
            "tau_ci_high": summ.tau_ci_high,
            "sigma": summ.sigma_mean if summ.sigma_mean is not None else 0.0,
            "sigma_ci_low": summ.sigma_ci_low,
            "sigma_ci_high": summ.sigma_ci_high,
            "spatial_preserve_rate": float(np.mean(spatial_preserved_list)) if spatial_preserved_list else None,
            "flip_rate": summ.flip_rate if summ.flip_rate is not None else 0.0,
            "detection_rate": summ.both_detected_rate,
            "n_valid": summ.n_valid,
            "n_total": summ.n_total,
        }
        cell_results.append(rec)
        print(
            f"[E3 Full] {l_lbl} @ {w_lbl} | tau_color={rec['tau']:.4f} "
            f"[{rec['tau_ci_low']:.3f},{rec['tau_ci_high']:.3f}] flip={rec['flip_rate']:.4f} "
            f"sigma={rec['sigma']:.4f} spatial_pres={rec['spatial_preserve_rate']:.4f}"
        )

    peak_cell = max(cell_results, key=lambda r: r["tau"])
    peak_l_lbl = peak_cell["layer_group"]
    peak_layers = peak_cell["layers"]

    for pair, sd in instances[:6]:
        inst_id = f"{pair.pair_id}_s{sd}"
        clean_imgs[inst_id].save(img_dir / f"full_{inst_id}_clean.png")
        corrupt_imgs[inst_id].save(img_dir / f"full_{inst_id}_corrupt.png")
        cell_imgs[peak_l_lbl][inst_id].save(img_dir / f"full_{inst_id}_{peak_l_lbl}_W1[4-7].png")

    # 2. Per-(l, h, t) head breakdown across L2, L3, L10, L11 at W1[4-7]
    head_records: list[dict] = []
    head_tau_mat = np.zeros((len(target_layers), num_heads), dtype=np.float64)
    for li, l_idx in enumerate(target_layers):
        for h_idx in range(num_heads):
            outcomes = []
            for pair, sd in instances:
                inst_id = f"{pair.pair_id}_s{sd}"
                spec = InterventionSpec(
                    kind="patch",
                    hook_point="attn2_head_pre_proj",
                    layers=[l_idx],
                    steps=PEAK_STEPS,
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
                    color_a=pair.color_a_clean,
                    color_b=pair.color_b_clean,
                    expected_relation=pair.relation_clean,
                )
                outcomes.append(
                    compute_pair_causal_outcome(
                        pair_id=inst_id,
                        task="color_binding",
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
                    "window": "W1[4-7]",
                    "steps": PEAK_STEPS,
                    "tau": tau_v,
                    "tau_ci_low": summ.tau_ci_low,
                    "tau_ci_high": summ.tau_ci_high,
                    "sigma": summ.sigma_mean if summ.sigma_mean is not None else 0.0,
                    "flip_rate": summ.flip_rate if summ.flip_rate is not None else 0.0,
                    "detection_rate": summ.both_detected_rate,
                }
            )
            print(
                f"[E3 Full Head] L{l_idx:02d}H{h_idx:02d}@W1[4-7] | tau={tau_v:.3f} "
                f"flip={head_records[-1]['flip_rate']:.2f} sigma={head_records[-1]['sigma']:.3f}"
            )

    # 3. Four controls around the peak cell
    rng = random.Random(2026)
    rand_layer = rng.choice(list(range(num_layers)))
    rand_head = rng.randint(0, num_heads - 1)

    control_defs = {
        "random_head": {
            "desc": f"Patch single random head L{rand_layer}H{rand_head} at W1[4-7]",
            "hook_point": "attn2_head_pre_proj",
            "layers": [rand_layer],
            "steps": PEAK_STEPS,
            "heads": [rand_head],
            "cache_src": "clean",
        },
        "wrong_timestep": {
            "desc": f"Patch {peak_l_lbl} at late window W4[16-19]",
            "hook_point": hook_point,
            "layers": peak_layers,
            "steps": WRONG_STEPS,
            "heads": None,
            "cache_src": "clean",
        },
        "same_prompt_noop": {
            "desc": f"Patch corrupt cache back into corrupt at {peak_l_lbl}@W1[4-7]",
            "hook_point": hook_point,
            "layers": peak_layers,
            "steps": PEAK_STEPS,
            "heads": None,
            "cache_src": "corrupt",
        },
        "seed_variation": {
            "desc": f"Patch clean cache from shifted seed (s+999) into corrupt at {peak_l_lbl}@W1[4-7]",
            "hook_point": hook_point,
            "layers": peak_layers,
            "steps": PEAK_STEPS,
            "heads": None,
            "cache_src": "shifted",
        },
    }

    control_results: dict[str, dict] = {}
    control_imgs: dict[str, dict[str, Image.Image]] = {}
    for cname, cfg in control_defs.items():
        outcomes = []
        control_imgs[cname] = {}
        spatial_pres_ctrl = []
        for pair, sd in instances:
            inst_id = f"{pair.pair_id}_s{sd}"
            if cfg["cache_src"] == "clean":
                src = clean_caches[inst_id]
            elif cfg["cache_src"] == "corrupt":
                src = corrupt_caches[inst_id]
            else:
                src = shifted_caches[inst_id]
            spec = InterventionSpec(
                kind="patch",
                hook_point=cfg["hook_point"],  # type: ignore[arg-type]
                layers=cfg["layers"],  # type: ignore[arg-type]
                steps=cfg["steps"],  # type: ignore[arg-type]
                heads=cfg["heads"],  # type: ignore[arg-type]
                source_cache=src,
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
                color_a=pair.color_a_clean,
                color_b=pair.color_b_clean,
                expected_relation=pair.relation_clean,
            )
            outc = compute_pair_causal_outcome(
                pair_id=inst_id,
                task="color_binding",
                clean=clean_evals[inst_id],
                corrupt=corrupt_evals[inst_id],
                patched=ev_ctrl,
                clean_relation=pair.relation_clean,
                corrupt_relation=pair.relation_corrupt,
            )
            outcomes.append(outc)
            if outc.valid_counterfactual:
                spatial_pres_ctrl.append(
                    1.0 if ev_ctrl.relation_pred == corrupt_evals[inst_id].relation_pred else 0.0
                )
        summ = summarize_causal_outcomes(outcomes, valid_only=True)
        control_results[cname] = {
            "description": cfg["desc"],
            "spatial_preserve_rate": float(np.mean(spatial_pres_ctrl)) if spatial_pres_ctrl else None,
            **summ.to_dict(),
        }
        print(
            f"[E3 Full Control] {cname:<18} | tau={summ.tau_mean} flip={summ.flip_rate} "
            f"sigma={summ.sigma_mean} spatial_pres={control_results[cname]['spatial_preserve_rate']}"
        )

    best_head = max(head_records, key=lambda r: r["tau"])
    plot_layer_timestep_heatmap(
        head_tau_mat,
        layers=target_layers,
        step_labels=[f"H{h}" for h in range(num_heads)],
        title="E3 Full Sweep Color-Binding Per-Head Tau @ W1[4-7] [512-MS, test split]",
        cbar_label="Color Transfer (tau)",
        out_path=img_dir / "e3_full_head_tau.png",
        cmap="magma",
    )

    grid_rows = []
    for pair, sd in instances[:6]:
        inst_id = f"{pair.pair_id}_s{sd}"
        other_lbl = "L2-3" if peak_l_lbl != "L2-3" else "L10-11"
        grid_rows.append(
            (
                inst_id,
                [
                    ("Clean", clean_imgs[inst_id]),
                    ("Corrupt", corrupt_imgs[inst_id]),
                    (f"{peak_l_lbl}@W1", cell_imgs[peak_l_lbl][inst_id]),
                    (f"{other_lbl}@W1", cell_imgs[other_lbl][inst_id]),
                    ("RandHead", control_imgs["random_head"][inst_id]),
                    ("WrongStep", control_imgs["wrong_timestep"][inst_id]),
                    ("SeedVar", control_imgs["seed_variation"][inst_id]),
                ],
            )
        )
    save_counterfactual_grid(grid_rows, out_path / "e3_full_grid.png", tile_size=224)

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
        "n_valid_counterfactual": peak_cell["n_valid"],
        "peak_cells": cell_results,
        "peak_cell": peak_cell,
        "peak_single_head": best_head,
        "per_l_h_t": head_records,
        "controls": control_results,
        "confirms_pilot": (peak_l_lbl == "L10-11") and (peak_cell["tau"] > (control_results["random_head"]["tau_mean"] or 0.0)),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with (out_path / "e3_full_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log_lines = [
        "# E3 Color-Binding Full Sweep Run Log",
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
        "- Split: `test` (fresh object-disjoint color-binding pairs, template `fixed_lr_swap_colors`)",
        "- Sampler: `DPMSolverMultistepScheduler`",
        f"- Steps: `{num_inference_steps}`",
        f"- Guidance scale: `{guidance_scale}`",
        "- Precision: `torch.bfloat16`",
        f"- Hook point: `{hook_point}` (conditional CFG branch only)",
        f"- Seeds: `{SEEDS}`",
        f"- Total (pair, seed) instances: `{len(instances)}` (valid counterfactuals: `{results['n_valid_counterfactual']}`)",
        "",
        "## Fresh Test Pairs",
        "",
    ]
    for p_info in results["pairs"]:
        log_lines.append(
            f"- `{p_info['pair_id']}`: Clean = `{p_info['clean_prompt']}` | Corrupt = `{p_info['corrupt_prompt']}`"
        )

    log_lines.extend(
        [
            "",
            "## Peak Cell Results",
            "",
            "| Cell | Tau (Color) | 95% CI | Flip Rate | Sigma | Spatial Pres. | Detection Rate |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for c in cell_results:
        log_lines.append(
            f"| {c['layer_group']} @ {c['window']} | {c['tau']:.4f} | "
            f"[{c['tau_ci_low']:.3f}, {c['tau_ci_high']:.3f}] | {c['flip_rate']:.4f} | "
            f"{c['sigma']:.4f} | {c['spatial_preserve_rate']:.4f} | {c['detection_rate']:.4f} |"
        )

    log_lines.extend(
        [
            "",
            f"- Peak single head `(l, h, t)`: `L{best_head['layer']}H{best_head['head']} @ W1[4-7]` (`tau = {best_head['tau']:.4f}`, `flip_rate = {best_head['flip_rate']:.4f}`, `sigma = {best_head['sigma']:.4f}`)",
            f"- Confirms E3 Pilot (`L10-11` > `L2-3` and beats `random_head`): `{results['confirms_pilot']}`",
            "",
            "## Controls",
            "",
            "| Control | Description | Tau (Color) | Flip Rate | Sigma | Spatial Pres. |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for cname, cdata in control_results.items():
        sp = f"{cdata['spatial_preserve_rate']:.4f}" if cdata["spatial_preserve_rate"] is not None else "N/A"
        log_lines.append(
            f"| {cname} | {cdata['description']} | {(cdata['tau_mean'] or 0.0):.4f} | "
            f"{(cdata['flip_rate'] or 0.0):.4f} | {(cdata['sigma_mean'] or 0.0):.4f} | {sp} |"
        )
    (out_path / "e3_full_run_log.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[E3 Full] DONE in {results['elapsed_sec']}s.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 Full Sweep on fresh test-split color pairs")
    parser.add_argument("--num-pairs", type=int, default=6)
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--out-dir", default="experiments/e3")
    args = parser.parse_args()
    run_e3_full(
        num_pairs=args.num_pairs,
        gen_device=args.gen_device,
        eval_device=args.eval_device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
