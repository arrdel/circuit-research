"""Day 3 Task 2: 1024-MS confirmation and sublayer comparison (attn2_out, attn1_out, ff_out, attn2_in_txt)."""

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
import torch
import transformers

from benchmark.counterfactual_pairs import CounterfactualPair, build_spatial_lr_pairs
from evaluation.causal_metrics import (
    compute_pair_causal_outcome,
    summarize_causal_outcomes,
)
from evaluation.spatial_eval import ImageEvalResult, SpatialColorEvaluator
from instrumentation.hooks import ActivationCache, DiTHookManager, HookPoint
from interventions.patching import InterventionController, InterventionSpec
from models.pixart_sigma import PixArtSigmaRunner


MODEL_1024 = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
MODEL_512 = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
SEEDS = [42, 123, 456]


def _select_pairs(
    runner: PixArtSigmaRunner,
    encoded: dict,
    evaluator: SpatialColorEvaluator,
    candidates: list[CounterfactualPair],
    target_pairs: int = 3,
    steps: int = 20,
    guidance: float = 4.5,
    height: int | None = None,
    width: int | None = None,
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
                height=height,
                width=width,
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
                height=height,
                width=width,
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
            print(f"[E1-T2] Selected {pair.pair_id} ({valid_seeds}/3 valid seeds)")
    return [(p, sd) for p in chosen for sd in SEEDS]


def run_1024_confirmation(
    num_pairs: int = 3,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
    img_out_dir: str = "experiments/e1/images",
) -> dict:
    out_img_dir = Path(img_out_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    all_explicit = build_spatial_lr_pairs(
        num_dev=24,
        num_test=24,
        num_ood=6,
        templates=("on_the_side_explicit",),
        base_seed=3000,
    )
    test_candidates = [p for p in all_explicit if p.split == "test"]
    prompts = [p.clean_prompt for p in test_candidates] + [p.corrupt_prompt for p in test_candidates]

    runner = PixArtSigmaRunner(model_id=MODEL_1024, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    instances = _select_pairs(
        runner, encoded, evaluator, test_candidates, target_pairs=num_pairs, height=1024, width=1024
    )
    if not instances:
        raise RuntimeError("No valid 1024-MS pairs selected.")

    peak_steps = [4, 5, 6, 7]
    wrong_steps = [16, 17, 18, 19]
    target_layers = [2, 3, 10, 11]

    clean_caches: dict[str, ActivationCache] = {}
    corrupt_caches: dict[str, ActivationCache] = {}
    shifted_caches: dict[str, ActivationCache] = {}
    clean_evals: dict[str, ImageEvalResult] = {}
    corrupt_evals: dict[str, ImageEvalResult] = {}

    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        mgr_c = DiTHookManager(
            runner.transformer,
            layers=target_layers,
            hook_points=["attn2_out"],
            steps=peak_steps + wrong_steps,
            branch="cond",
            store_cpu=True,
        )
        with mgr_c.active() as c_cache:
            res_c = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=sd,
                height=1024,
                width=1024,
                step_callbacks=[mgr_c],
            )
            mgr_c.cache = ActivationCache()

        mgr_r = DiTHookManager(
            runner.transformer,
            layers=[10, 11],
            hook_points=["attn2_out"],
            steps=peak_steps,
            branch="cond",
            store_cpu=True,
        )
        with mgr_r.active() as r_cache:
            res_r = runner.generate(
                prompt_entry=encoded[pair.corrupt_prompt],
                negative_entry=neg_entry,
                seed=sd,
                height=1024,
                width=1024,
                step_callbacks=[mgr_r],
            )
            mgr_r.cache = ActivationCache()

        mgr_s = DiTHookManager(
            runner.transformer,
            layers=[10, 11],
            hook_points=["attn2_out"],
            steps=peak_steps,
            branch="cond",
            store_cpu=True,
        )
        with mgr_s.active() as s_cache:
            _ = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=sd + 999,
                height=1024,
                width=1024,
                step_callbacks=[mgr_s],
                decode_image=False,
            )
            mgr_s.cache = ActivationCache()

        clean_caches[inst_id] = c_cache
        corrupt_caches[inst_id] = r_cache
        shifted_caches[inst_id] = s_cache
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
        if sd == SEEDS[0]:
            res_c.image.save(out_img_dir / f"1024_{inst_id}_clean.png")
            res_r.image.save(out_img_dir / f"1024_{inst_id}_corrupt.png")

    rng = random.Random(2026)
    rand_layer = rng.choice([2, 3, 10, 11])
    rand_head = rng.randint(0, runner.num_heads - 1)

    # For random_head, capture attn2_head_pre_proj on rand_layer @ peak_steps
    head_clean_caches: dict[str, ActivationCache] = {}
    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        mgr_h = DiTHookManager(
            runner.transformer,
            layers=[rand_layer],
            hook_points=["attn2_head_pre_proj"],
            steps=peak_steps,
            branch="cond",
            store_cpu=True,
        )
        with mgr_h.active() as hc:
            _ = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=sd,
                height=1024,
                width=1024,
                step_callbacks=[mgr_h],
                decode_image=False,
            )
            mgr_h.cache = ActivationCache()
        head_clean_caches[inst_id] = hc

    conditions = {
        "L10-11_W1[4-7]": ("attn2_out", [10, 11], peak_steps, None, "clean"),
        "L2-3_W1[4-7]": ("attn2_out", [2, 3], peak_steps, None, "clean"),
        "ctrl_random_head": ("attn2_head_pre_proj", [rand_layer], peak_steps, [rand_head], "head_clean"),
        "ctrl_wrong_timestep": ("attn2_out", [10, 11], wrong_steps, None, "clean"),
        "ctrl_same_prompt_noop": ("attn2_out", [10, 11], peak_steps, None, "corrupt"),
        "ctrl_seed_variation": ("attn2_out", [10, 11], peak_steps, None, "shifted"),
    }

    cond_results: dict[str, dict] = {}
    for cname, (hp, l_idxs, s_idxs, h_idxs, src_kind) in conditions.items():
        outcomes = []
        for pair, sd in instances:
            inst_id = f"{pair.pair_id}_s{sd}"
            if src_kind == "clean":
                src = clean_caches[inst_id]
            elif src_kind == "head_clean":
                src = head_clean_caches[inst_id]
            elif src_kind == "corrupt":
                src = corrupt_caches[inst_id]
            else:
                src = shifted_caches[inst_id]

            spec = InterventionSpec(
                kind="patch",
                hook_point=hp,  # type: ignore[arg-type]
                layers=l_idxs,
                steps=s_idxs,
                heads=h_idxs,
                source_cache=src,
                branch="cond",
            )
            ctrl = InterventionController(runner.transformer, [spec])
            with ctrl.active():
                res_p = runner.generate(
                    prompt_entry=encoded[pair.corrupt_prompt],
                    negative_entry=neg_entry,
                    seed=sd,
                    height=1024,
                    width=1024,
                    step_callbacks=[ctrl],
                )
            if sd == SEEDS[0] and cname in ("L10-11_W1[4-7]", "L2-3_W1[4-7]"):
                res_p.image.save(out_img_dir / f"1024_{inst_id}_{cname}.png")
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
        summ = summarize_causal_outcomes(outcomes, valid_only=True).to_dict()
        cond_results[cname] = summ
        print(
            f"[1024-MS] {cname:<22} | tau={summ['tau_mean']:.4f} flip={summ['flip_rate']:.2f} "
            f"sigma={summ['sigma_mean']:.4f} det={summ['both_detected_rate']:.2f}"
        )

    del runner
    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": MODEL_1024,
        "checkpoint_label": "1024-MS",
        "pairs": [p.pair_id for p, _ in instances[:: len(SEEDS)]],
        "seeds": SEEDS,
        "n_instances": len(instances),
        "random_head_desc": f"L{rand_layer}H{rand_head}@W1[4-7]",
        "conditions": cond_results,
    }


def run_sublayer_comparison(
    num_pairs: int = 4,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
) -> dict:
    all_explicit = build_spatial_lr_pairs(
        num_dev=24,
        num_test=24,
        num_ood=6,
        templates=("on_the_side_explicit",),
        base_seed=3000,
    )
    test_candidates = [p for p in all_explicit if p.split == "test"]
    prompts = [p.clean_prompt for p in test_candidates] + [p.corrupt_prompt for p in test_candidates]

    runner = PixArtSigmaRunner(model_id=MODEL_512, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    instances = _select_pairs(runner, encoded, evaluator, test_candidates, target_pairs=num_pairs)
    hook_points: list[HookPoint] = ["attn2_out", "attn1_out", "ff_out", "attn2_in_txt"]
    cells = [
        ("L10-11_W1[4-7]", [10, 11], [4, 5, 6, 7]),
        ("L2-3_W1[4-7]", [2, 3], [4, 5, 6, 7]),
        ("L6-7_W2[8-11]", [6, 7], [8, 9, 10, 11]),
    ]
    all_layers = [2, 3, 6, 7, 10, 11]
    all_steps = [4, 5, 6, 7, 8, 9, 10, 11]

    clean_caches: dict[str, ActivationCache] = {}
    clean_evals: dict[str, ImageEvalResult] = {}
    corrupt_evals: dict[str, ImageEvalResult] = {}

    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        mgr = DiTHookManager(
            runner.transformer,
            layers=all_layers,
            hook_points=hook_points,
            steps=all_steps,
            branch="cond",
            store_cpu=True,
        )
        with mgr.active() as c_cache:
            res_c = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=sd,
                step_callbacks=[mgr],
            )
            mgr.cache = ActivationCache()
        res_r = runner.generate(
            prompt_entry=encoded[pair.corrupt_prompt],
            negative_entry=neg_entry,
            seed=sd,
        )
        clean_caches[inst_id] = c_cache
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

    sublayer_results: dict[str, dict[str, dict]] = {}
    for cell_lbl, l_idxs, s_idxs in cells:
        sublayer_results[cell_lbl] = {}
        for hp in hook_points:
            outcomes = []
            for pair, sd in instances:
                inst_id = f"{pair.pair_id}_s{sd}"
                spec = InterventionSpec(
                    kind="patch",
                    hook_point=hp,
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
                        step_callbacks=[ctrl],
                    )
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
            summ = summarize_causal_outcomes(outcomes, valid_only=True).to_dict()
            sublayer_results[cell_lbl][hp] = summ
            print(
                f"[512-MS Sublayer] {cell_lbl:<15} x {hp:<13} | tau={summ['tau_mean']:.4f} "
                f"flip={summ['flip_rate']:.2f} sigma={summ['sigma_mean']:.4f}"
            )

    return {
        "checkpoint": MODEL_512,
        "checkpoint_label": "512-MS",
        "pairs": [p.pair_id for p, _ in instances[:: len(SEEDS)]],
        "seeds": SEEDS,
        "n_instances": len(instances),
        "by_cell_and_hook_point": sublayer_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 1024-MS confirmation and sublayer comparison")
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--out-json", default="experiments/e1/e1_1024_check.json")
    parser.add_argument("--out-log", default="experiments/e1/e1_1024_run_log.md")
    args = parser.parse_args()

    t0 = time.time()
    res_1024 = run_1024_confirmation(num_pairs=3, gen_device=args.gen_device, eval_device=args.eval_device)
    res_sub = run_sublayer_comparison(num_pairs=4, gen_device=args.gen_device, eval_device=args.eval_device)

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    payload = {
        "host": socket.gethostname(),
        "gpus": gpu_names,
        "confirmation_1024_ms": res_1024,
        "sublayer_comparison_512_ms": res_sub,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# E1 1024-MS Confirmation and Sublayer Comparison Run Log",
        "",
        "## Execution Environment",
        "",
        f"- Host: `{socket.gethostname()}`",
        f"- GPUs visible: `{gpu_names}` (gen: `{args.gen_device}`, eval: `{args.eval_device}`)",
        f"- Python: `{platform.python_version()}`",
        f"- Torch: `{torch.__version__}`, Diffusers: `{diffusers.__version__}`, Transformers: `{transformers.__version__}`",
        "",
        "## Part A: 1024-MS Confirmation (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`)",
        "",
        f"- Pairs: `{res_1024['pairs']}`, Seeds: `{res_1024['seeds']}`, Random-head control: `{res_1024['random_head_desc']}`",
        "",
        "| Condition | Tau | Flip Rate | Sigma | Detection Rate | Color Preserve |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cname, s in res_1024["conditions"].items():
        cp = f"{s['color_preserve_rate']:.4f}" if s["color_preserve_rate"] is not None else "N/A"
        lines.append(
            f"| {cname} | {s['tau_mean']:.4f} | {s['flip_rate']:.4f} | {s['sigma_mean']:.4f} | "
            f"{s['both_detected_rate']:.4f} | {cp} |"
        )

    lines.extend(
        [
            "",
            "## Part B: Sublayer Comparison (`PixArt-alpha/PixArt-Sigma-XL-2-512-MS`)",
            "",
            f"- Pairs: `{res_sub['pairs']}`, Seeds: `{res_sub['seeds']}`",
            "",
            "| Cell | Hook Point | Tau | Flip Rate | Sigma | Detection Rate |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for cell_lbl, hp_map in res_sub["by_cell_and_hook_point"].items():
        for hp, s in hp_map.items():
            lines.append(
                f"| {cell_lbl} | {hp} | {s['tau_mean']:.4f} | {s['flip_rate']:.4f} | "
                f"{s['sigma_mean']:.4f} | {s['both_detected_rate']:.4f} |"
            )

    Path(args.out_log).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[E1-T2] Saved {args.out_json} and {args.out_log}")


if __name__ == "__main__":
    main()
