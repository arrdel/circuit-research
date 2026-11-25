"""E1: Layer x Timestep activation patching for selective spatial relation transfer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark.counterfactual_pairs import CounterfactualPair, load_pairs_jsonl
from evaluation.causal_metrics import (
    PairCausalOutcome,
    compute_pair_causal_outcome,
    summarize_causal_outcomes,
)
from evaluation.spatial_eval import ImageEvalResult, SpatialColorEvaluator
from instrumentation.hooks import DiTHookManager, HookPoint
from interventions.patching import InterventionController, InterventionSpec
from models.pixart_sigma import DEFAULT_MODEL_ID, PixArtSigmaRunner
from visualization.heatmaps import plot_layer_timestep_heatmap, save_counterfactual_grid


DEFAULT_WINDOWS: list[tuple[str, list[int]]] = [
    ("W0[0-3]", [0, 1, 2, 3]),
    ("W1[4-7]", [4, 5, 6, 7]),
    ("W2[8-11]", [8, 9, 10, 11]),
    ("W3[12-15]", [12, 13, 14, 15]),
    ("W4[16-19]", [16, 17, 18, 19]),
]


def _make_layer_groups(num_layers: int, group_size: int) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = []
    for start in range(0, num_layers, group_size):
        idxs = list(range(start, min(num_layers, start + group_size)))
        label = f"L{idxs[0]}" if len(idxs) == 1 else f"L{idxs[0]}-{idxs[-1]}"
        groups.append((label, idxs))
    return groups


def run_e1(
    model_id: str = DEFAULT_MODEL_ID,
    pairs_path: str = "artifacts/data/spatial_lr.jsonl",
    e0_summary_path: str = "artifacts/e0_baseline/summary.json",
    hook_point: HookPoint = "block_out",
    layer_group_size: int = 2,
    max_pairs: int = 12,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:1",
    out_dir: str = "artifacts/e1_spatial_patching",
) -> dict:
    t_start = time.time()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = load_pairs_jsonl(pairs_path)
    dev_pairs = [p for p in all_pairs if p.split == "dev"]

    valid_ids: set[str] = set()
    e0_file = Path(e0_summary_path)
    if e0_file.exists():
        with e0_file.open("r", encoding="utf-8") as f:
            e0_data = json.load(f)
        valid_ids = set(e0_data.get("spatial_lr", {}).get("valid_pair_ids", []))

    selected_pairs: list[CounterfactualPair] = [p for p in dev_pairs if p.pair_id in valid_ids]
    if len(selected_pairs) < max_pairs:
        remaining = [p for p in dev_pairs if p.pair_id not in valid_ids]
        selected_pairs.extend(remaining[: max_pairs - len(selected_pairs)])
    selected_pairs = selected_pairs[:max_pairs]

    if not selected_pairs:
        raise RuntimeError("No counterfactual pairs available for E1.")

    # Cache prompt embeddings
    prompts = []
    for p in selected_pairs:
        prompts.extend([p.clean_prompt, p.corrupt_prompt])

    runner = PixArtSigmaRunner(model_id=model_id, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    if torch.cuda.device_count() < 2:
        eval_device = gen_device
    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    num_layers = runner.num_layers
    layer_groups = _make_layer_groups(num_layers, layer_group_size)
    windows = DEFAULT_WINDOWS

    # Step 1: Capture clean activation caches and evaluate baseline clean/corrupt outputs
    clean_caches = {}
    clean_evals: dict[str, ImageEvalResult] = {}
    corrupt_evals: dict[str, ImageEvalResult] = {}
    clean_images = {}
    corrupt_images = {}

    print(f"[E1] Caching clean activations & baseline evaluations for {len(selected_pairs)} pairs...")
    for pair in selected_pairs:
        hook_mgr = DiTHookManager(
            runner.transformer,
            layers=list(range(num_layers)),
            hook_points=[hook_point],
            steps=list(range(num_inference_steps)),
            branch="cond",
            store_cpu=True,
        )
        with hook_mgr.active() as cache:
            res_clean = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=pair.seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                step_callbacks=[hook_mgr],
            )
            # Clone cache tensors out of hook_mgr
            clean_caches[pair.pair_id] = cache
            hook_mgr.cache = type(cache)()

        res_corrupt = runner.generate(
            prompt_entry=encoded[pair.corrupt_prompt],
            negative_entry=neg_entry,
            seed=pair.seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        clean_images[pair.pair_id] = res_clean.image
        corrupt_images[pair.pair_id] = res_corrupt.image
        clean_evals[pair.pair_id] = evaluator.evaluate_image(
            res_clean.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )
        corrupt_evals[pair.pair_id] = evaluator.evaluate_image(
            res_corrupt.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_corrupt,
            color_b=pair.color_b_corrupt,
            expected_relation=pair.relation_corrupt,
        )

    n_valid_e1 = sum(
        1
        for p in selected_pairs
        if clean_evals[p.pair_id].dx is not None
        and corrupt_evals[p.pair_id].dx is not None
        and clean_evals[p.pair_id].relation_pred == p.relation_clean
        and corrupt_evals[p.pair_id].relation_pred == p.relation_corrupt
    )
    print(f"[E1] Valid counterfactual pairs in slice: {n_valid_e1}/{len(selected_pairs)}")

    # Step 2: Sweep Layer Group x Timestep Window
    tau_mat = np.full((len(layer_groups), len(windows)), np.nan, dtype=np.float64)
    sigma_mat = np.full((len(layer_groups), len(windows)), np.nan, dtype=np.float64)
    flip_mat = np.full((len(layer_groups), len(windows)), np.nan, dtype=np.float64)

    grid_rows: list[dict] = []
    best_score = -1.0
    best_cell: tuple[int, int] = (0, 0)
    best_patched_images: dict[str, object] = {}

    for li, (l_label, layer_idxs) in enumerate(layer_groups):
        for wi, (w_label, step_idxs) in enumerate(windows):
            outcomes: list[PairCausalOutcome] = []
            cell_images: dict[str, object] = {}

            for pair in selected_pairs:
                spec = InterventionSpec(
                    kind="patch",
                    hook_point=hook_point,
                    layers=layer_idxs,
                    steps=step_idxs,
                    source_cache=clean_caches[pair.pair_id],
                    branch="cond",
                )
                ctrl = InterventionController(runner.transformer, [spec])
                with ctrl.active():
                    res_patched = runner.generate(
                        prompt_entry=encoded[pair.corrupt_prompt],
                        negative_entry=neg_entry,
                        seed=pair.seed,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        step_callbacks=[ctrl],
                    )
                cell_images[pair.pair_id] = res_patched.image
                ev_patched = evaluator.evaluate_image(
                    res_patched.image,
                    obj_a=pair.obj_a,
                    obj_b=pair.obj_b,
                    color_a=pair.color_a_corrupt,
                    color_b=pair.color_b_corrupt,
                    expected_relation=pair.relation_clean,
                )
                outcomes.append(
                    compute_pair_causal_outcome(
                        pair_id=pair.pair_id,
                        task="spatial_lr",
                        clean=clean_evals[pair.pair_id],
                        corrupt=corrupt_evals[pair.pair_id],
                        patched=ev_patched,
                        clean_relation=pair.relation_clean,
                        corrupt_relation=pair.relation_corrupt,
                    )
                )

            use_valid_only = n_valid_e1 >= 3
            summ = summarize_causal_outcomes(outcomes, valid_only=use_valid_only)
            tau_val = summ.tau_mean if summ.tau_mean is not None else 0.0
            sig_val = summ.sigma_mean if summ.sigma_mean is not None else 0.0
            flip_val = summ.flip_rate if summ.flip_rate is not None else 0.0

            tau_mat[li, wi] = tau_val
            sigma_mat[li, wi] = sig_val
            flip_mat[li, wi] = flip_val

            grid_rows.append(
                {
                    "layer_group": l_label,
                    "layers": layer_idxs,
                    "window": w_label,
                    "steps": step_idxs,
                    **summ.to_dict(),
                }
            )
            print(
                f"[E1] {l_label:>7} x {w_label:<9} | tau={tau_val:.3f} "
                f"flip={flip_val:.2f} sigma={sig_val:.3f} det={summ.both_detected_rate:.2f}"
            )

            # Combined selection objective favoring selective, non-destructive transfer
            det_ok = summ.both_detected_rate >= 0.80
            sel_score = (tau_val * sig_val * summ.both_detected_rate) if det_ok else -1.0
            if sel_score > best_score:
                best_score = sel_score
                best_cell = (li, wi)
                best_patched_images = cell_images

    # Step 3: Four Causal Controls around the peak cell
    best_li, best_wi = best_cell
    peak_label, peak_layers = layer_groups[best_li]
    peak_w_label, peak_steps = windows[best_wi]

    # Pick wrong layer far from peak
    wrong_li = 0 if best_li > len(layer_groups) // 2 else len(layer_groups) - 1
    wrong_label, wrong_layers = layer_groups[wrong_li]
    late_label, late_steps = windows[-1]

    controls_spec = {
        "peak_patch": ("patch", peak_layers, peak_steps, False),
        f"ctrl_wrong_layer_{wrong_label}": ("patch", wrong_layers, peak_steps, False),
        f"ctrl_late_window_{late_label}": ("patch", peak_layers, late_steps, False),
        "ctrl_shuffled_pair": ("patch", peak_layers, peak_steps, True),
        "ctrl_random_norm_matched": ("random_norm_matched", peak_layers, peak_steps, False),
    }

    control_results: dict[str, dict] = {}
    control_images: dict[str, dict[str, object]] = {}

    for ctrl_name, (kind, c_layers, c_steps, use_shuffled) in controls_spec.items():
        outcomes = []
        control_images[ctrl_name] = {}
        for idx, pair in enumerate(selected_pairs):
            src_pair = selected_pairs[(idx + 1) % len(selected_pairs)] if use_shuffled else pair
            spec = InterventionSpec(
                kind=kind,  # type: ignore[arg-type]
                hook_point=hook_point,
                layers=c_layers,
                steps=c_steps,
                source_cache=None if kind == "random_norm_matched" else clean_caches[src_pair.pair_id],
                branch="cond",
                seed=pair.seed + 77,
            )
            ctrl = InterventionController(runner.transformer, [spec])
            with ctrl.active():
                res_ctrl = runner.generate(
                    prompt_entry=encoded[pair.corrupt_prompt],
                    negative_entry=neg_entry,
                    seed=pair.seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    step_callbacks=[ctrl],
                )
            control_images[ctrl_name][pair.pair_id] = res_ctrl.image
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
                    pair_id=pair.pair_id,
                    task="spatial_lr",
                    clean=clean_evals[pair.pair_id],
                    corrupt=corrupt_evals[pair.pair_id],
                    patched=ev_ctrl,
                    clean_relation=pair.relation_clean,
                    corrupt_relation=pair.relation_corrupt,
                )
            )
        summ = summarize_causal_outcomes(outcomes, valid_only=(n_valid_e1 >= 3))
        control_results[ctrl_name] = summ.to_dict()
        print(
            f"[E1 Control] {ctrl_name:<28} | tau={summ.tau_mean} flip={summ.flip_rate} sigma={summ.sigma_mean}"
        )

    # Save heatmaps and sample counterfactual visual grids
    layer_labels = [lbl for lbl, _ in layer_groups]
    step_labels = [lbl for lbl, _ in windows]
    plot_layer_timestep_heatmap(
        tau_mat,
        layers=[g[1][0] for g in layer_groups],
        step_labels=step_labels,
        title=f"E1 Spatial Transfer Score (tau) [{hook_point}]",
        cbar_label="Transfer Score (tau)",
        out_path=fig_dir / "heatmap_tau.png",
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
    )
    plot_layer_timestep_heatmap(
        sigma_mat,
        layers=[g[1][0] for g in layer_groups],
        step_labels=step_labels,
        title=f"E1 Non-Target Specificity (sigma) [{hook_point}]",
        cbar_label="Specificity (sigma)",
        out_path=fig_dir / "heatmap_sigma.png",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    plot_layer_timestep_heatmap(
        flip_mat,
        layers=[g[1][0] for g in layer_groups],
        step_labels=step_labels,
        title=f"E1 Spatial Relation Flip Rate [{hook_point}]",
        cbar_label="Flip Rate",
        out_path=fig_dir / "heatmap_flip_rate.png",
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
    )

    vis_rows = []
    for pair in selected_pairs[:6]:
        pid = pair.pair_id
        vis_rows.append(
            (
                pid,
                [
                    ("Clean (L)", clean_images[pid]),
                    ("Corrupt (R)", corrupt_images[pid]),
                    (f"Peak {peak_label}@{peak_w_label}", best_patched_images[pid]),
                    ("WrongLayer", control_images[f"ctrl_wrong_layer_{wrong_label}"][pid]),
                    ("RandNorm", control_images["ctrl_random_norm_matched"][pid]),
                ],
            )
        )
    if vis_rows:
        save_counterfactual_grid(vis_rows, fig_dir / "counterfactual_examples.png")

    pd.DataFrame(grid_rows).to_csv(out_path / "sweep_results.csv", index=False)

    summary = {
        "model_id": model_id,
        "hook_point": hook_point,
        "n_pairs": len(selected_pairs),
        "n_valid_counterfactual": n_valid_e1,
        "peak_cell": {
            "layer_group": peak_label,
            "layers": peak_layers,
            "window": peak_w_label,
            "steps": peak_steps,
            "tau": float(tau_mat[best_li, best_wi]),
            "sigma": float(sigma_mat[best_li, best_wi]),
            "flip_rate": float(flip_mat[best_li, best_wi]),
        },
        "controls": control_results,
        "elapsed_sec": round(time.time() - t_start, 2),
    }
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 spatial relation activation patching sweep")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--pairs-path", default="artifacts/data/spatial_lr.jsonl")
    parser.add_argument("--e0-summary", default="artifacts/e0_baseline/summary.json")
    parser.add_argument("--hook-point", default="block_out")
    parser.add_argument("--layer-group-size", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:1")
    parser.add_argument("--out-dir", default="artifacts/e1_spatial_patching")
    args = parser.parse_args()

    run_e1(
        model_id=args.model_id,
        pairs_path=args.pairs_path,
        e0_summary_path=args.e0_summary,
        hook_point=args.hook_point,  # type: ignore[arg-type]
        layer_group_size=args.layer_group_size,
        max_pairs=args.max_pairs,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        gen_device=args.gen_device,
        eval_device=args.eval_device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
