"""E3: Color-attribute binding activation patching & spatial double-dissociation test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark.counterfactual_pairs import (
    CounterfactualPair,
    build_color_binding_pairs,
    load_pairs_jsonl,
    save_pairs_jsonl,
)
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


WINDOWS = [
    ("W0[0-3]", [0, 1, 2, 3]),
    ("W1[4-7]", [4, 5, 6, 7]),
    ("W2[8-11]", [8, 9, 10, 11]),
    ("W3[12-15]", [12, 13, 14, 15]),
    ("W4[16-19]", [16, 17, 18, 19]),
]


def run_e3(
    model_id: str = DEFAULT_MODEL_ID,
    hook_point: HookPoint = "attn2_out",
    layer_group_size: int = 2,
    num_pairs: int = 12,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
    out_dir: str = "artifacts/e3_color_binding",
) -> dict:
    t0 = time.time()
    out_path = Path(out_dir) / hook_point
    out_path.mkdir(parents=True, exist_ok=True)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rich_path = Path("artifacts/data/color_binding_rich.jsonl")
    if not rich_path.exists():
        pairs_all = build_color_binding_pairs(
            num_dev=60,
            num_test=80,
            num_ood=40,
            templates=("fixed_lr_swap_colors",),
            base_seed=6000,
        )
        save_pairs_jsonl(pairs_all, rich_path)
    else:
        pairs_all = load_pairs_jsonl(rich_path)

    pool = [p for p in pairs_all if p.split in ("dev", "test")]
    prompts = []
    for p in pool:
        prompts.extend([p.clean_prompt, p.corrupt_prompt])

    runner = PixArtSigmaRunner(model_id=model_id, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    num_layers = runner.num_layers
    layer_groups = [
        (
            f"L{s}" if layer_group_size == 1 else f"L{s}-{min(num_layers - 1, s + layer_group_size - 1)}",
            list(range(s, min(num_layers, s + layer_group_size))),
        )
        for s in range(0, num_layers, layer_group_size)
    ]

    valid_pairs: list[CounterfactualPair] = []
    clean_caches: dict[str, ActivationCache] = {}
    clean_evals: dict[str, ImageEvalResult] = {}
    corrupt_evals: dict[str, ImageEvalResult] = {}
    clean_images = {}
    corrupt_images = {}

    print(f"[E3:{hook_point}] Screening color-binding counterfactual pairs (target={num_pairs})...")
    for pair in pool:
        if len(valid_pairs) >= num_pairs:
            break
        res_r = runner.generate(
            prompt_entry=encoded[pair.corrupt_prompt],
            negative_entry=neg_entry,
            seed=pair.seed,
        )
        ev_r = evaluator.evaluate_image(
            res_r.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_corrupt,
            color_b=pair.color_b_corrupt,
            expected_relation=pair.relation_corrupt,
        )
        if not (ev_r.both_detected and ev_r.color_binding_correct):
            continue

        mgr = DiTHookManager(
            runner.transformer,
            layers=list(range(num_layers)),
            hook_points=[hook_point],
            steps=list(range(20)),
            branch="cond",
            store_cpu=True,
        )
        with mgr.active() as c_cache:
            res_c = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=pair.seed,
                step_callbacks=[mgr],
            )
            mgr.cache = ActivationCache()

        ev_c = evaluator.evaluate_image(
            res_c.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )
        if not (ev_c.both_detected and ev_c.color_binding_correct):
            continue

        # Re-evaluate corrupt under clean color frame so tau_color measures shift toward clean binding
        ev_r_in_clean_frame = evaluator.evaluate_image(
            res_r.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )
        if (ev_c.color_a_margin + ev_c.color_b_margin) - (
            ev_r_in_clean_frame.color_a_margin + ev_r_in_clean_frame.color_b_margin
        ) <= 0.02:
            continue

        valid_pairs.append(pair)
        clean_caches[pair.pair_id] = c_cache
        clean_evals[pair.pair_id] = ev_c
        corrupt_evals[pair.pair_id] = ev_r_in_clean_frame
        clean_images[pair.pair_id] = res_c.image
        corrupt_images[pair.pair_id] = res_r.image

    print(f"[E3:{hook_point}] Collected {len(valid_pairs)} valid color-binding pairs.")
    tau_mat = np.full((len(layer_groups), len(WINDOWS)), np.nan, dtype=np.float64)
    sigma_mat = np.full((len(layer_groups), len(WINDOWS)), np.nan, dtype=np.float64)
    flip_mat = np.full((len(layer_groups), len(WINDOWS)), np.nan, dtype=np.float64)

    rows: list[dict] = []
    best_score = -1.0
    best_label = ""
    best_images = {}

    for li, (l_lbl, l_idxs) in enumerate(layer_groups):
        for wi, (w_lbl, s_idxs) in enumerate(WINDOWS):
            outcomes: list[PairCausalOutcome] = []
            cell_imgs = {}
            for pair in valid_pairs:
                spec = InterventionSpec(
                    kind="patch",
                    hook_point=hook_point,
                    layers=l_idxs,
                    steps=s_idxs,
                    source_cache=clean_caches[pair.pair_id],
                    branch="cond",
                )
                ctrl = InterventionController(runner.transformer, [spec])
                with ctrl.active():
                    res_p = runner.generate(
                        prompt_entry=encoded[pair.corrupt_prompt],
                        negative_entry=neg_entry,
                        seed=pair.seed,
                        step_callbacks=[ctrl],
                    )
                cell_imgs[pair.pair_id] = res_p.image
                ev_p = evaluator.evaluate_image(
                    res_p.image,
                    obj_a=pair.obj_a,
                    obj_b=pair.obj_b,
                    color_a=pair.color_a_clean,
                    color_b=pair.color_b_clean,
                    expected_relation=pair.relation_clean,
                )
                outcomes.append(
                    compute_pair_causal_outcome(
                        pair_id=pair.pair_id,
                        task="color_binding",
                        clean=clean_evals[pair.pair_id],
                        corrupt=corrupt_evals[pair.pair_id],
                        patched=ev_p,
                        clean_relation=pair.relation_clean,
                        corrupt_relation=pair.relation_corrupt,
                    )
                )

            summ = summarize_causal_outcomes(outcomes, valid_only=True)
            tau_v = summ.tau_mean if summ.tau_mean is not None else 0.0
            sig_v = summ.sigma_mean if summ.sigma_mean is not None else 0.0
            flip_v = summ.flip_rate if summ.flip_rate is not None else 0.0

            tau_mat[li, wi] = tau_v
            sigma_mat[li, wi] = sig_v
            flip_mat[li, wi] = flip_v

            rows.append(
                {
                    "hook_point": hook_point,
                    "layer_group": l_lbl,
                    "layers": l_idxs,
                    "window": w_lbl,
                    "steps": s_idxs,
                    **summ.to_dict(),
                }
            )
            print(
                f"[E3:{hook_point}] {l_lbl:>7} x {w_lbl:<9} | tau_color={tau_v:.3f} "
                f"flip={flip_v:.2f} sigma_spatial={sig_v:.3f} det={summ.both_detected_rate:.2f}"
            )
            score = (tau_v * sig_v * summ.both_detected_rate) if summ.both_detected_rate >= 0.8 else -1.0
            if score > best_score:
                best_score = score
                best_label = f"{l_lbl}@{w_lbl}"
                best_images = cell_imgs

    pd.DataFrame(rows).to_csv(out_path / "results.csv", index=False)
    plot_layer_timestep_heatmap(
        tau_mat,
        layers=[g[1][0] for g in layer_groups],
        step_labels=[w[0] for w in WINDOWS],
        title=f"E3 Color-Binding Transfer (tau) [{hook_point}, N={len(valid_pairs)}]",
        cbar_label="Color Transfer (tau)",
        out_path=fig_dir / "heatmap_tau_color.png",
        cmap="magma",
    )
    plot_layer_timestep_heatmap(
        flip_mat,
        layers=[g[1][0] for g in layer_groups],
        step_labels=[w[0] for w in WINDOWS],
        title=f"E3 Color-Binding Flip Rate [{hook_point}, N={len(valid_pairs)}]",
        cbar_label="Color Flip Rate",
        out_path=fig_dir / "heatmap_flip_color.png",
        cmap="plasma",
    )
    plot_layer_timestep_heatmap(
        sigma_mat,
        layers=[g[1][0] for g in layer_groups],
        step_labels=[w[0] for w in WINDOWS],
        title=f"E3 Spatial Preservation (sigma) [{hook_point}, N={len(valid_pairs)}]",
        cbar_label="Specificity (sigma)",
        out_path=fig_dir / "heatmap_sigma_color.png",
        cmap="viridis",
    )

    if best_images:
        grid_rows = [
            (
                p.pair_id,
                [
                    ("Clean", clean_images[p.pair_id]),
                    ("Corrupt", corrupt_images[p.pair_id]),
                    (f"Peak {best_label}", best_images[p.pair_id]),
                ],
            )
            for p in valid_pairs[:6]
        ]
        save_counterfactual_grid(grid_rows, fig_dir / "color_binding_examples.png")

    summary = {
        "model_id": model_id,
        "hook_point": hook_point,
        "n_valid_pairs": len(valid_pairs),
        "peak_cell": best_label,
        "rows": rows,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 color-binding patching sweep")
    parser.add_argument("--hook-point", default="attn2_out")
    parser.add_argument("--layer-group-size", type=int, default=2)
    parser.add_argument("--num-pairs", type=int, default=12)
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--out-dir", default="artifacts/e3_color_binding")
    args = parser.parse_args()

    run_e3(
        hook_point=args.hook_point,  # type: ignore[arg-type]
        layer_group_size=args.layer_group_size,
        num_pairs=args.num_pairs,
        gen_device=args.gen_device,
        eval_device=args.eval_device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
