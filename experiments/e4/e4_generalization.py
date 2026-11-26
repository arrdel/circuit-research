"""Day 7 Task 1: E4 Generalization (a) Terse-template rescue and (b) Held-out OOD combinations."""

from __future__ import annotations

import argparse
import gc
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

from benchmark.counterfactual_pairs import (
    CounterfactualPair,
    build_color_binding_pairs,
    build_spatial_lr_pairs,
)
from evaluation.causal_metrics import (
    compute_pair_causal_outcome,
    summarize_causal_outcomes,
)
from evaluation.spatial_eval import ImageEvalResult, SpatialColorEvaluator
from instrumentation.hooks import ActivationCache, DiTHookManager
from interventions.patching import InterventionController, InterventionSpec
from models.pixart_sigma import DEFAULT_MODEL_ID, PixArtSigmaRunner
from visualization.heatmaps import save_counterfactual_grid


MODEL_1024 = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
SEEDS = [42, 123, 456]
PEAK_STEPS = [4, 5, 6, 7]
CONSOL_STEPS = [8, 9, 10, 11]
WRONG_STEPS = [16, 17, 18, 19]

E1_COALITION = {10: [6, 8, 15], 11: [1, 11, 13]}


def _build_circuit_specs(
    mode: str,
    src_cache: ActivationCache,
) -> list[InterventionSpec]:
    if mode == "L3H02_W1":
        return [
            InterventionSpec(
                kind="patch",
                hook_point="attn2_head_pre_proj",
                layers=[3],
                steps=PEAK_STEPS,
                heads=[2],
                source_cache=src_cache,
                branch="cond",
            )
        ]
    if mode == "L10_11_coalition_W1":
        return [
            InterventionSpec(
                kind="patch",
                hook_point="attn2_head_pre_proj",
                layers=[l_idx],
                steps=PEAK_STEPS,
                heads=h_list,
                source_cache=src_cache,
                branch="cond",
            )
            for l_idx, h_list in E1_COALITION.items()
        ]
    if mode == "L10_11_full_W1":
        return [
            InterventionSpec(
                kind="patch",
                hook_point="attn2_out",
                layers=[10, 11],
                steps=PEAK_STEPS,
                source_cache=src_cache,
                branch="cond",
            )
        ]
    if mode == "L6_7_attn1_W2":
        return [
            InterventionSpec(
                kind="patch",
                hook_point="attn1_out",
                layers=[6, 7],
                steps=CONSOL_STEPS,
                source_cache=src_cache,
                branch="cond",
            )
        ]
    if mode == "combined_circuit":
        return (
            _build_circuit_specs("L3H02_W1", src_cache)
            + _build_circuit_specs("L10_11_coalition_W1", src_cache)
            + _build_circuit_specs("L6_7_attn1_W2", src_cache)
        )
    if mode == "random_head":
        return [
            InterventionSpec(
                kind="patch",
                hook_point="attn2_head_pre_proj",
                layers=[3],
                steps=PEAK_STEPS,
                heads=[10],
                source_cache=src_cache,
                branch="cond",
            )
        ]
    raise ValueError(f"Unknown mode: {mode}")


def run_terse_rescue_512(
    runner: PixArtSigmaRunner,
    evaluator: SpatialColorEvaluator,
    img_dir: Path,
    target_instances: int = 8,
) -> dict:
    """E4(a.1): Rescue failing 'to_the_side_of' terse prompts using explicit donors (512-MS)."""
    terse_pairs = build_spatial_lr_pairs(
        num_dev=24, num_test=24, num_ood=0, templates=("to_the_side_of",), base_seed=7000
    )
    explicit_pairs = build_spatial_lr_pairs(
        num_dev=24, num_test=24, num_ood=0, templates=("on_the_side_explicit",), base_seed=7000
    )
    # Note: because templates have same length pools & rng seed, pair i shares exact (obj_a, obj_b, color_a, color_b)!
    prompts = []
    for tp, ep in zip(terse_pairs, explicit_pairs):
        prompts.extend([tp.clean_prompt, tp.corrupt_prompt, ep.clean_prompt, ep.corrupt_prompt])
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]

    rescue_items = []
    for tp, ep in zip(terse_pairs, explicit_pairs):
        if len(rescue_items) >= target_instances:
            break
        assert (tp.obj_a, tp.obj_b) == (ep.obj_a, ep.obj_b)
        for sd in SEEDS:
            if len(rescue_items) >= target_instances:
                break
            # Check clean ("left") failing on terse AND succeeding on explicit donor
            res_terse = runner.generate(prompt_entry=encoded[tp.clean_prompt], negative_entry=neg_entry, seed=sd)
            ev_terse = evaluator.evaluate_image(
                res_terse.image,
                obj_a=tp.obj_a,
                obj_b=tp.obj_b,
                color_a=tp.color_a_clean,
                color_b=tp.color_b_clean,
                expected_relation="left",
            )
            if not (ev_terse.dx is not None and ev_terse.relation_pred == "right"):
                continue

            mgr = DiTHookManager(
                runner.transformer,
                layers=[3, 6, 7, 10, 11],
                hook_points=["attn2_head_pre_proj", "attn2_out", "attn1_out"],
                steps=PEAK_STEPS + CONSOL_STEPS,
                branch="cond",
                store_cpu=True,
            )
            with mgr.active() as d_cache:
                res_donor = runner.generate(
                    prompt_entry=encoded[ep.clean_prompt],
                    negative_entry=neg_entry,
                    seed=sd,
                    step_callbacks=[mgr],
                )
                mgr.cache = ActivationCache()
            ev_donor = evaluator.evaluate_image(
                res_donor.image,
                obj_a=ep.obj_a,
                obj_b=ep.obj_b,
                color_a=ep.color_a_clean,
                color_b=ep.color_b_clean,
                expected_relation="left",
            )
            if not (ev_donor.dx is not None and ev_donor.relation_pred == "left" and (ev_donor.dx - ev_terse.dx) > 0.04):
                continue

            rescue_items.append(
                {
                    "id": f"terse_{tp.pair_id}_s{sd}",
                    "terse_prompt": tp.clean_prompt,
                    "donor_prompt": ep.clean_prompt,
                    "pair": tp,
                    "seed": sd,
                    "donor_cache": d_cache,
                    "donor_eval": ev_donor,
                    "terse_eval": ev_terse,
                    "donor_img": res_donor.image,
                    "terse_img": res_terse.image,
                }
            )
            print(f"[E4(a) 512-MS] Found failing terse instance: {tp.pair_id}_s{sd} (terse dx={ev_terse.dx:.3f}, donor dx={ev_donor.dx:.3f})")

    modes = [
        "L3H02_W1",
        "L10_11_coalition_W1",
        "L10_11_full_W1",
        "L6_7_attn1_W2",
        "combined_circuit",
        "random_head",
    ]
    mode_results = {}
    grid_rows = []
    per_mode_imgs = {m: {} for m in modes}

    for mode in modes:
        outcomes = []
        for item in rescue_items:
            specs = _build_circuit_specs(mode, item["donor_cache"])
            ctrl = InterventionController(runner.transformer, specs)
            with ctrl.active():
                res_p = runner.generate(
                    prompt_entry=encoded[item["terse_prompt"]],
                    negative_entry=neg_entry,
                    seed=item["seed"],
                    step_callbacks=[ctrl],
                )
            per_mode_imgs[mode][item["id"]] = res_p.image
            p = item["pair"]
            ev_p = evaluator.evaluate_image(
                res_p.image,
                obj_a=p.obj_a,
                obj_b=p.obj_b,
                color_a=p.color_a_clean,
                color_b=p.color_b_clean,
                expected_relation="left",
            )
            outcomes.append(
                compute_pair_causal_outcome(
                    pair_id=item["id"],
                    task="spatial_lr",
                    clean=item["donor_eval"],
                    corrupt=item["terse_eval"],
                    patched=ev_p,
                    clean_relation="left",
                    corrupt_relation="right",
                )
            )
        summ = summarize_causal_outcomes(outcomes, valid_only=True).to_dict()
        mode_results[mode] = summ
        print(
            f"[E4(a) 512-MS] {mode:<20} | tau_rescue={summ['tau_mean']:.4f} "
            f"flip={summ['flip_rate']:.4f} sigma={summ['sigma_mean']:.4f}"
        )

    for item in rescue_items[:6]:
        iid = item["id"]
        item["donor_img"].save(img_dir / f"e4a_512_{iid}_donor.png")
        item["terse_img"].save(img_dir / f"e4a_512_{iid}_terse_fail.png")
        per_mode_imgs["combined_circuit"][iid].save(img_dir / f"e4a_512_{iid}_combined.png")
        grid_rows.append(
            (
                iid,
                [
                    ("Donor (Explicit L)", item["donor_img"]),
                    ("Terse Fail (R)", item["terse_img"]),
                    ("L3H02", per_mode_imgs["L3H02_W1"][iid]),
                    ("L10-11 Coal", per_mode_imgs["L10_11_coalition_W1"][iid]),
                    ("L6-7 attn1", per_mode_imgs["L6_7_attn1_W2"][iid]),
                    ("Combined", per_mode_imgs["combined_circuit"][iid]),
                    ("RandHead", per_mode_imgs["random_head"][iid]),
                ],
            )
        )
    if grid_rows:
        save_counterfactual_grid(grid_rows, Path("experiments/e4/e4a_terse_rescue_512_grid.png"), tile_size=224)

    return {
        "checkpoint": DEFAULT_MODEL_ID,
        "checkpoint_label": "512-MS",
        "n_failing_terse_instances": len(rescue_items),
        "instances": [
            {
                "id": it["id"],
                "terse_prompt": it["terse_prompt"],
                "donor_prompt": it["donor_prompt"],
                "terse_dx": it["terse_eval"].dx,
                "donor_dx": it["donor_eval"].dx,
            }
            for it in rescue_items
        ],
        "by_intervention": mode_results,
    }


def run_terse_rescue_1024_e0(
    gen_device: str,
    eval_device: str,
    img_dir: Path,
) -> dict:
    """E4(a.2): Test circuit rescue on exact E0 1024-MS failures (p1, p4, p5)."""
    e0_cases = [
        {
            "case_id": "e0_p1_s42",
            "terse_prompt": "Red cube left of blue sphere.",
            "donor_prompt": "A red cube on the left and a blue sphere on the right, clean studio photo",
            "seed": 42,
            "obj_a": "cube",
            "obj_b": "sphere",
            "color_a": "red",
            "color_b": "blue",
        },
        {
            "case_id": "e0_p1_s123",
            "terse_prompt": "Red cube left of blue sphere.",
            "donor_prompt": "A red cube on the left and a blue sphere on the right, clean studio photo",
            "seed": 123,
            "obj_a": "cube",
            "obj_b": "sphere",
            "color_a": "red",
            "color_b": "blue",
        },
        {
            "case_id": "e0_p4_s42",
            "terse_prompt": "Red cube left of blue sphere, green cylinder in front.",
            "donor_prompt": "A red cube on the left and a blue sphere on the right, green cylinder in front, clean studio photo",
            "seed": 42,
            "obj_a": "cube",
            "obj_b": "sphere",
            "color_a": "red",
            "color_b": "blue",
        },
        {
            "case_id": "e0_p5_s42",
            "terse_prompt": "Two red cubes and one blue sphere.",
            "donor_prompt": "Two red cubes on the left and one blue sphere on the right, clean studio photo",
            "seed": 42,
            "obj_a": "cube",
            "obj_b": "sphere",
            "color_a": "red",
            "color_b": "blue",
        },
    ]

    prompts = []
    for c in e0_cases:
        prompts.extend([c["terse_prompt"], c["donor_prompt"]])

    runner = PixArtSigmaRunner(model_id=MODEL_1024, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device, box_threshold=0.18, text_threshold=0.18)
    evaluator.load()

    modes = ["L3H02_W1", "L10_11_full_W1", "L6_7_attn1_W2", "combined_circuit", "random_head"]
    case_reports = []
    grid_rows = []

    for c in e0_cases:
        cid = c["case_id"]
        sd = c["seed"]
        res_terse = runner.generate(
            prompt_entry=encoded[c["terse_prompt"]], negative_entry=neg_entry, seed=sd, height=1024, width=1024
        )
        ev_terse = evaluator.evaluate_image(
            res_terse.image,
            obj_a=c["obj_a"],
            obj_b=c["obj_b"],
            color_a=c["color_a"],
            color_b=c["color_b"],
            expected_relation="left",
        )

        mgr = DiTHookManager(
            runner.transformer,
            layers=[3, 6, 7, 10, 11],
            hook_points=["attn2_head_pre_proj", "attn2_out", "attn1_out"],
            steps=PEAK_STEPS + CONSOL_STEPS,
            branch="cond",
            store_cpu=True,
        )
        with mgr.active() as d_cache:
            res_donor = runner.generate(
                prompt_entry=encoded[c["donor_prompt"]],
                negative_entry=neg_entry,
                seed=sd,
                height=1024,
                width=1024,
                step_callbacks=[mgr],
            )
            mgr.cache = ActivationCache()
        ev_donor = evaluator.evaluate_image(
            res_donor.image,
            obj_a=c["obj_a"],
            obj_b=c["obj_b"],
            color_a=c["color_a"],
            color_b=c["color_b"],
            expected_relation="left",
        )

        res_terse.image.save(img_dir / f"e4a_1024_{cid}_terse.png")
        res_donor.image.save(img_dir / f"e4a_1024_{cid}_donor.png")

        row_tiles = [("Donor", res_donor.image), ("Terse Base", res_terse.image)]
        per_mode = {}
        for mode in modes:
            specs = _build_circuit_specs(mode, d_cache)
            ctrl = InterventionController(runner.transformer, specs)
            with ctrl.active():
                res_p = runner.generate(
                    prompt_entry=encoded[c["terse_prompt"]],
                    negative_entry=neg_entry,
                    seed=sd,
                    height=1024,
                    width=1024,
                    step_callbacks=[ctrl],
                )
            res_p.image.save(img_dir / f"e4a_1024_{cid}_{mode}.png")
            ev_p = evaluator.evaluate_image(
                res_p.image,
                obj_a=c["obj_a"],
                obj_b=c["obj_b"],
                color_a=c["color_a"],
                color_b=c["color_b"],
                expected_relation="left",
            )
            outc = compute_pair_causal_outcome(
                pair_id=cid,
                task="spatial_lr",
                clean=ev_donor,
                corrupt=ev_terse,
                patched=ev_p,
                clean_relation="left",
                corrupt_relation="right",
            )
            per_mode[mode] = {
                "dx": ev_p.dx,
                "relation_pred": ev_p.relation_pred,
                "color_binding_correct": ev_p.color_binding_correct,
                "tau_toward_donor": outc.tau,
            }
            row_tiles.append((mode.replace("_W1", "").replace("_W2", ""), res_p.image))
            print(f"[E4(a) 1024-MS {cid}] {mode:<18} | dx={ev_p.dx} rel={ev_p.relation_pred} tau={outc.tau}")

        grid_rows.append((cid, row_tiles))
        case_reports.append(
            {
                "case_id": cid,
                "terse_prompt": c["terse_prompt"],
                "donor_prompt": c["donor_prompt"],
                "seed": sd,
                "terse_base": {"dx": ev_terse.dx, "rel": ev_terse.relation_pred, "color_ok": ev_terse.color_binding_correct},
                "donor_base": {"dx": ev_donor.dx, "rel": ev_donor.relation_pred, "color_ok": ev_donor.color_binding_correct},
                "interventions": per_mode,
            }
        )

    save_counterfactual_grid(grid_rows, Path("experiments/e4/e4a_terse_rescue_1024_grid.png"), tile_size=224)

    del runner
    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": MODEL_1024,
        "checkpoint_label": "1024-MS",
        "cases": case_reports,
    }


def run_ood_generalization(
    runner: PixArtSigmaRunner,
    evaluator: SpatialColorEvaluator,
    img_dir: Path,
    task: str,
    num_pairs: int = 3,
) -> dict:
    """E4(b): Held-out OOD object combinations for E1 spatial or E3 color binding (512-MS)."""
    if task == "spatial_lr":
        all_p = build_spatial_lr_pairs(
            num_dev=6, num_test=6, num_ood=20, templates=("on_the_side_explicit",), base_seed=3000
        )
    else:
        all_p = build_color_binding_pairs(
            num_dev=6, num_test=6, num_ood=30, templates=("fixed_lr_swap_colors",), base_seed=6000
        )
    ood_candidates = [p for p in all_p if p.split == "ood"]
    prompts = [p.clean_prompt for p in ood_candidates] + [p.corrupt_prompt for p in ood_candidates]
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]

    chosen: list[CounterfactualPair] = []
    for pair in ood_candidates:
        if len(chosen) >= num_pairs:
            break
        valid_seeds = 0
        for sd in SEEDS:
            res_c = runner.generate(prompt_entry=encoded[pair.clean_prompt], negative_entry=neg_entry, seed=sd)
            res_r = runner.generate(prompt_entry=encoded[pair.corrupt_prompt], negative_entry=neg_entry, seed=sd)
            ev_c = evaluator.evaluate_image(
                res_c.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_clean,
                color_b=pair.color_b_clean,
                expected_relation=pair.relation_clean,
                compute_embedding=False,
            )
            if task == "spatial_lr":
                ev_r = evaluator.evaluate_image(
                    res_r.image,
                    obj_a=pair.obj_a,
                    obj_b=pair.obj_b,
                    color_a=pair.color_a_corrupt,
                    color_b=pair.color_b_corrupt,
                    expected_relation=pair.relation_corrupt,
                    compute_embedding=False,
                )
                ok = (
                    ev_c.dx is not None
                    and ev_r.dx is not None
                    and ev_c.relation_pred == pair.relation_clean
                    and ev_r.relation_pred == pair.relation_corrupt
                    and (ev_c.dx - ev_r.dx) > 0.04
                )
            else:
                ev_r_nat = evaluator.evaluate_image(
                    res_r.image,
                    obj_a=pair.obj_a,
                    obj_b=pair.obj_b,
                    color_a=pair.color_a_corrupt,
                    color_b=pair.color_b_corrupt,
                    expected_relation=pair.relation_corrupt,
                    compute_embedding=False,
                )
                ev_r_cln = evaluator.evaluate_image(
                    res_r.image,
                    obj_a=pair.obj_a,
                    obj_b=pair.obj_b,
                    color_a=pair.color_a_clean,
                    color_b=pair.color_b_clean,
                    expected_relation=pair.relation_clean,
                    compute_embedding=False,
                )
                ok = bool(
                    ev_c.both_detected
                    and ev_c.color_binding_correct
                    and ev_r_nat.both_detected
                    and ev_r_nat.color_binding_correct
                    and ((ev_c.color_a_margin or 0.0) + (ev_c.color_b_margin or 0.0))
                    - ((ev_r_cln.color_a_margin or 0.0) + (ev_r_cln.color_b_margin or 0.0))
                    > 0.02
                )
            if ok:
                valid_seeds += 1
        if valid_seeds >= 2:
            chosen.append(pair)
            print(f"[E4(b) OOD {task}] Selected {pair.pair_id} ({valid_seeds}/3 valid seeds)")

    instances = [(p, sd) for p in chosen for sd in SEEDS]
    clean_caches = {}
    corrupt_caches = {}
    shifted_caches = {}
    clean_evals = {}
    corrupt_evals = {}
    clean_imgs = {}
    corrupt_imgs = {}

    for pair, sd in instances:
        iid = f"{pair.pair_id}_s{sd}"
        mgr_c = DiTHookManager(
            runner.transformer,
            layers=[2, 3, 6, 7, 10, 11],
            hook_points=["attn2_out", "attn1_out", "attn2_head_pre_proj"],
            steps=PEAK_STEPS + CONSOL_STEPS + WRONG_STEPS,
            branch="cond",
            store_cpu=True,
        )
        with mgr_c.active() as cc:
            res_c = runner.generate(
                prompt_entry=encoded[pair.clean_prompt], negative_entry=neg_entry, seed=sd, step_callbacks=[mgr_c]
            )
            mgr_c.cache = ActivationCache()

        mgr_r = DiTHookManager(
            runner.transformer,
            layers=[10, 11],
            hook_points=["attn2_out"],
            steps=PEAK_STEPS,
            branch="cond",
            store_cpu=True,
        )
        with mgr_r.active() as rc:
            res_r = runner.generate(
                prompt_entry=encoded[pair.corrupt_prompt], negative_entry=neg_entry, seed=sd, step_callbacks=[mgr_r]
            )
            mgr_r.cache = ActivationCache()

        mgr_s = DiTHookManager(
            runner.transformer,
            layers=[10, 11],
            hook_points=["attn2_out"],
            steps=PEAK_STEPS,
            branch="cond",
            store_cpu=True,
        )
        with mgr_s.active() as sc:
            _ = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=sd + 999,
                step_callbacks=[mgr_s],
                decode_image=False,
            )
            mgr_s.cache = ActivationCache()

        clean_caches[iid] = cc
        corrupt_caches[iid] = rc
        shifted_caches[iid] = sc
        clean_imgs[iid] = res_c.image
        corrupt_imgs[iid] = res_r.image
        clean_evals[iid] = evaluator.evaluate_image(
            res_c.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )
        corrupt_evals[iid] = evaluator.evaluate_image(
            res_r.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_corrupt if task == "spatial_lr" else pair.color_a_clean,
            color_b=pair.color_b_corrupt if task == "spatial_lr" else pair.color_b_clean,
            expected_relation=pair.relation_corrupt if task == "spatial_lr" else pair.relation_clean,
        )

    if task == "spatial_lr":
        conds = {
            "L10-11_W1": ("attn2_out", [10, 11], PEAK_STEPS, None, "clean"),
            "L2-3_W1": ("attn2_out", [2, 3], PEAK_STEPS, None, "clean"),
            "L3H02_W1": ("attn2_head_pre_proj", [3], PEAK_STEPS, [2], "clean"),
            "L6-7_attn1_W2": ("attn1_out", [6, 7], CONSOL_STEPS, None, "clean"),
            "ctrl_random_head": ("attn2_head_pre_proj", [3], PEAK_STEPS, [10], "clean"),
            "ctrl_wrong_timestep": ("attn2_out", [10, 11], WRONG_STEPS, None, "clean"),
            "ctrl_same_prompt_noop": ("attn2_out", [10, 11], PEAK_STEPS, None, "corrupt"),
            "ctrl_seed_variation": ("attn2_out", [10, 11], PEAK_STEPS, None, "shifted"),
        }
    else:
        conds = {
            "L10-11_W1": ("attn2_out", [10, 11], PEAK_STEPS, None, "clean"),
            "L2-3_W1": ("attn2_out", [2, 3], PEAK_STEPS, None, "clean"),
            "L11H01_W1": ("attn2_head_pre_proj", [11], PEAK_STEPS, [1], "clean"),
            "ctrl_random_head": ("attn2_head_pre_proj", [3], PEAK_STEPS, [10], "clean"),
            "ctrl_wrong_timestep": ("attn2_out", [10, 11], WRONG_STEPS, None, "clean"),
            "ctrl_same_prompt_noop": ("attn2_out", [10, 11], PEAK_STEPS, None, "corrupt"),
            "ctrl_seed_variation": ("attn2_out", [10, 11], PEAK_STEPS, None, "shifted"),
        }

    cond_results = {}
    saved_imgs = {c: {} for c in conds}
    for cname, (hp, layers, steps, heads, src_kind) in conds.items():
        outcomes = []
        for pair, sd in instances:
            iid = f"{pair.pair_id}_s{sd}"
            src = clean_caches[iid] if src_kind == "clean" else (corrupt_caches[iid] if src_kind == "corrupt" else shifted_caches[iid])
            spec = InterventionSpec(
                kind="patch",
                hook_point=hp,  # type: ignore[arg-type]
                layers=layers,
                steps=steps,
                heads=heads,
                source_cache=src,
                branch="cond",
            )
            ctrl = InterventionController(runner.transformer, [spec])
            with ctrl.active():
                res_p = runner.generate(
                    prompt_entry=encoded[pair.corrupt_prompt],
                    negative_entry=neg_entry,
                    seed=sd,
                    step_callbacks=[ctrl],
                )
            saved_imgs[cname][iid] = res_p.image
            ev_p = evaluator.evaluate_image(
                res_p.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_corrupt if task == "spatial_lr" else pair.color_a_clean,
                color_b=pair.color_b_corrupt if task == "spatial_lr" else pair.color_b_clean,
                expected_relation=pair.relation_clean,
            )
            outcomes.append(
                compute_pair_causal_outcome(
                    pair_id=iid,
                    task=task,
                    clean=clean_evals[iid],
                    corrupt=corrupt_evals[iid],
                    patched=ev_p,
                    clean_relation=pair.relation_clean,
                    corrupt_relation=pair.relation_corrupt,
                )
            )
        summ = summarize_causal_outcomes(outcomes, valid_only=True).to_dict()
        cond_results[cname] = summ
        print(f"[E4(b) OOD {task}] {cname:<22} | tau={summ['tau_mean']:.4f} flip={summ['flip_rate']:.4f} sigma={summ['sigma_mean']:.4f}")

    grid_rows = []
    for pair, sd in instances[:4]:
        iid = f"{pair.pair_id}_s{sd}"
        clean_imgs[iid].save(img_dir / f"e4b_{task}_{iid}_clean.png")
        corrupt_imgs[iid].save(img_dir / f"e4b_{task}_{iid}_corrupt.png")
        saved_imgs["L10-11_W1"][iid].save(img_dir / f"e4b_{task}_{iid}_L10-11_W1.png")
        tiles = [("Clean", clean_imgs[iid]), ("Corrupt", corrupt_imgs[iid]), ("L10-11", saved_imgs["L10-11_W1"][iid]), ("L2-3", saved_imgs["L2-3_W1"][iid])]
        if task == "spatial_lr":
            tiles.extend([("L3H02", saved_imgs["L3H02_W1"][iid]), ("L6-7 a1", saved_imgs["L6-7_attn1_W2"][iid])])
        else:
            tiles.append(("L11H01", saved_imgs["L11H01_W1"][iid]))
        tiles.append(("RandHead", saved_imgs["ctrl_random_head"][iid]))
        grid_rows.append((iid, tiles))
    if grid_rows:
        save_counterfactual_grid(grid_rows, Path(f"experiments/e4/e4b_ood_{task}_grid.png"), tile_size=224)

    return {
        "checkpoint": DEFAULT_MODEL_ID,
        "checkpoint_label": "512-MS",
        "task": task,
        "pairs": [{"pair_id": p.pair_id, "clean_prompt": p.clean_prompt, "corrupt_prompt": p.corrupt_prompt} for p in chosen],
        "seeds": SEEDS,
        "n_instances": len(instances),
        "conditions": cond_results,
    }


def run_e4(gen_device: str = "cuda:0", eval_device: str = "cuda:0") -> dict:
    t0 = time.time()
    out_dir = Path("experiments/e4")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    runner = PixArtSigmaRunner(model_id=DEFAULT_MODEL_ID, device=gen_device, dtype=torch.bfloat16)
    runner.load_denoiser()
    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    res_a_512 = run_terse_rescue_512(runner, evaluator, img_dir, target_instances=8)
    res_b_spatial = run_ood_generalization(runner, evaluator, img_dir, task="spatial_lr", num_pairs=3)
    res_b_color = run_ood_generalization(runner, evaluator, img_dir, task="color_binding", num_pairs=3)

    del runner
    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    res_a_1024 = run_terse_rescue_1024_e0(gen_device, eval_device, img_dir)

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    payload = {
        "host": socket.gethostname(),
        "gpus": gpu_names,
        "e4a_terse_rescue_512_ms": res_a_512,
        "e4a_terse_rescue_1024_ms_e0": res_a_1024,
        "e4b_ood_spatial_512_ms": res_b_spatial,
        "e4b_ood_color_512_ms": res_b_color,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out_dir / "e4_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# E4 Generalization Run Log",
        "",
        "## Execution Environment",
        "",
        f"- Host: `{socket.gethostname()}`",
        f"- GPUs visible: `{gpu_names}` (gen: `{gen_device}`, eval: `{eval_device}`)",
        f"- Python: `{platform.python_version()}`",
        f"- Torch: `{torch.__version__}`, Diffusers: `{diffusers.__version__}`, Transformers: `{transformers.__version__}`",
        "",
        "## Part (a.1): Terse-Template Rescue (`512-MS`, `to_the_side_of` failing instances)",
        "",
        f"- Checkpoint: `{DEFAULT_MODEL_ID}` (`512-MS`), `N = {res_a_512['n_failing_terse_instances']}` failing terse instances.",
        "",
        "| Intervention | Tau (Rescue) | Flip Rate | Sigma | Detection Rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode, s in res_a_512["by_intervention"].items():
        lines.append(
            f"| {mode} | {(s['tau_mean'] or 0.0):.4f} | {(s['flip_rate'] or 0.0):.4f} | "
            f"{(s['sigma_mean'] or 0.0):.4f} | {s['both_detected_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Part (a.2): E0 Exact Failure Rescue (`1024-MS`)",
            "",
            "| Case ID | Terse Base dx (rel) | Donor dx (rel) | L3H02 dx (rel) | L10-11 dx (rel) | L6-7 attn1 dx (rel) | Combined dx (rel) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for c in res_a_1024["cases"]:
        iv = c["interventions"]
        lines.append(
            f"| {c['case_id']} | {c['terse_base']['dx']} ({c['terse_base']['rel']}) | "
            f"{c['donor_base']['dx']} ({c['donor_base']['rel']}) | "
            f"{iv['L3H02_W1']['dx']} ({iv['L3H02_W1']['relation_pred']}) | "
            f"{iv['L10_11_full_W1']['dx']} ({iv['L10_11_full_W1']['relation_pred']}) | "
            f"{iv['L6_7_attn1_W2']['dx']} ({iv['L6_7_attn1_W2']['relation_pred']}) | "
            f"{iv['combined_circuit']['dx']} ({iv['combined_circuit']['relation_pred']}) |"
        )

    for label, sub in [("Part (b.1): Held-Out OOD Spatial (`512-MS`)", res_b_spatial), ("Part (b.2): Held-Out OOD Color Binding (`512-MS`)", res_b_color)]:
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"- Pairs: `{[p['pair_id'] for p in sub['pairs']]}`",
                "",
                "| Condition | Tau | Flip Rate | Sigma | Detection Rate |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for cname, s in sub["conditions"].items():
            lines.append(
                f"| {cname} | {(s['tau_mean'] or 0.0):.4f} | {(s['flip_rate'] or 0.0):.4f} | "
                f"{(s['sigma_mean'] or 0.0):.4f} | {s['both_detected_rate']:.4f} |"
            )

    (out_dir / "e4_run_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[E4] DONE in {payload['elapsed_sec']}s.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="E4 Generalization")
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    args = parser.parse_args()
    run_e4(gen_device=args.gen_device, eval_device=args.eval_device)


if __name__ == "__main__":
    main()
