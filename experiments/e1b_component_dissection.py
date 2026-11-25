"""E1b: Single-layer and component dissection (attn2_out, attn1_out, ff_out, attn2_in_txt)."""

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
    build_spatial_lr_pairs,
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
from visualization.heatmaps import plot_layer_timestep_heatmap


def _parse_windows(window_spec: str) -> list[tuple[str, list[int]]]:
    if window_spec == "early_fine":
        return [
            ("S[0-1]", [0, 1]),
            ("S[2-3]", [2, 3]),
            ("S[4-5]", [4, 5]),
            ("S[6-7]", [6, 7]),
            ("S[8-11]", [8, 9, 10, 11]),
        ]
    return [
        ("W0[0-3]", [0, 1, 2, 3]),
        ("W1[4-7]", [4, 5, 6, 7]),
        ("W2[8-11]", [8, 9, 10, 11]),
    ]


def run_e1b(
    model_id: str = DEFAULT_MODEL_ID,
    hook_point: HookPoint = "attn2_out",
    layers: list[int] | None = None,
    layer_group_size: int = 1,
    window_mode: str = "early_coarse",
    num_pairs: int = 16,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
    out_dir: str = "artifacts/e1b_dissection",
) -> dict:
    t0 = time.time()
    out_path = Path(out_dir) / hook_point
    out_path.mkdir(parents=True, exist_ok=True)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Ensure rich benchmark exists
    rich_path = Path("artifacts/data/spatial_lr_rich.jsonl")
    if not rich_path.exists():
        pairs_all = build_spatial_lr_pairs(
            num_dev=60,
            num_test=80,
            num_ood=40,
            templates=("on_the_side_explicit", "side_by_side_scene"),
            base_seed=2000,
        )
        save_pairs_jsonl(pairs_all, rich_path)
    else:
        pairs_all = load_pairs_jsonl(rich_path)

    candidate_pairs = [p for p in pairs_all if p.split == "dev"]

    prompts = []
    for p in candidate_pairs:
        prompts.extend([p.clean_prompt, p.corrupt_prompt])

    runner = PixArtSigmaRunner(model_id=model_id, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    target_layers = layers if layers is not None else list(range(16))
    layer_groups: list[tuple[str, list[int]]] = []
    for i in range(0, len(target_layers), layer_group_size):
        grp = target_layers[i : i + layer_group_size]
        lbl = f"L{grp[0]}" if len(grp) == 1 else f"L{grp[0]}-{grp[-1]}"
        layer_groups.append((lbl, grp))

    windows = _parse_windows(window_mode)
    all_steps = sorted({s for _, steps in windows for s in steps})

    # Screen candidate pairs to collect num_pairs validated counterfactuals
    valid_pairs: list[CounterfactualPair] = []
    clean_caches: dict[str, ActivationCache] = {}
    clean_evals: dict[str, ImageEvalResult] = {}
    corrupt_evals: dict[str, ImageEvalResult] = {}

    print(f"[E1b:{hook_point}] Screening dev pairs to collect {num_pairs} valid counterfactuals...")
    for pair in candidate_pairs:
        if len(valid_pairs) >= num_pairs:
            break
        res_corrupt = runner.generate(
            prompt_entry=encoded[pair.corrupt_prompt],
            negative_entry=neg_entry,
            seed=pair.seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        ev_corrupt = evaluator.evaluate_image(
            res_corrupt.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_corrupt,
            color_b=pair.color_b_corrupt,
            expected_relation=pair.relation_corrupt,
        )
        if not (ev_corrupt.dx is not None and ev_corrupt.relation_pred == pair.relation_corrupt):
            continue

        hook_mgr = DiTHookManager(
            runner.transformer,
            layers=target_layers,
            hook_points=[hook_point],
            steps=all_steps,
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
            hook_mgr.cache = ActivationCache()

        ev_clean = evaluator.evaluate_image(
            res_clean.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )
        if not (
            ev_clean.dx is not None
            and ev_clean.relation_pred == pair.relation_clean
            and (ev_clean.dx - ev_corrupt.dx) > 0.04
        ):
            continue

        valid_pairs.append(pair)
        clean_caches[pair.pair_id] = cache
        clean_evals[pair.pair_id] = ev_clean
        corrupt_evals[pair.pair_id] = ev_corrupt

    print(f"[E1b:{hook_point}] Collected {len(valid_pairs)} validated counterfactual pairs.")
    if not valid_pairs:
        raise RuntimeError("No valid counterfactual pairs found.")

    tau_mat = np.full((len(layer_groups), len(windows)), np.nan, dtype=np.float64)
    sigma_mat = np.full((len(layer_groups), len(windows)), np.nan, dtype=np.float64)
    flip_mat = np.full((len(layer_groups), len(windows)), np.nan, dtype=np.float64)

    rows: list[dict] = []
    for li, (l_lbl, l_idxs) in enumerate(layer_groups):
        for wi, (w_lbl, s_idxs) in enumerate(windows):
            outcomes: list[PairCausalOutcome] = []
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
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
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
                        pair_id=pair.pair_id,
                        task="spatial_lr",
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

            row = {
                "hook_point": hook_point,
                "layer_group": l_lbl,
                "layers": l_idxs,
                "window": w_lbl,
                "steps": s_idxs,
                **summ.to_dict(),
            }
            rows.append(row)
            print(
                f"[E1b:{hook_point}] {l_lbl:>6} x {w_lbl:<8} | tau={tau_v:.3f} "
                f"[{summ.tau_ci_low:.3f},{summ.tau_ci_high:.3f}] flip={flip_v:.2f} "
                f"sigma={sig_v:.3f} det={summ.both_detected_rate:.2f}"
            )

    pd.DataFrame(rows).to_csv(out_path / "results.csv", index=False)

    plot_layer_timestep_heatmap(
        tau_mat,
        layers=[g[1][0] for g in layer_groups],
        step_labels=[w[0] for w in windows],
        title=f"E1b Spatial Transfer (tau) [{hook_point}, N={len(valid_pairs)}]",
        cbar_label="Transfer Score (tau)",
        out_path=fig_dir / "heatmap_tau.png",
        cmap="magma",
    )
    plot_layer_timestep_heatmap(
        flip_mat,
        layers=[g[1][0] for g in layer_groups],
        step_labels=[w[0] for w in windows],
        title=f"E1b Spatial Flip Rate [{hook_point}, N={len(valid_pairs)}]",
        cbar_label="Flip Rate",
        out_path=fig_dir / "heatmap_flip_rate.png",
        cmap="plasma",
    )
    plot_layer_timestep_heatmap(
        sigma_mat,
        layers=[g[1][0] for g in layer_groups],
        step_labels=[w[0] for w in windows],
        title=f"E1b Specificity (sigma) [{hook_point}, N={len(valid_pairs)}]",
        cbar_label="Specificity (sigma)",
        out_path=fig_dir / "heatmap_sigma.png",
        cmap="viridis",
    )

    summary = {
        "model_id": model_id,
        "hook_point": hook_point,
        "n_valid_pairs": len(valid_pairs),
        "valid_pair_ids": [p.pair_id for p in valid_pairs],
        "rows": rows,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E1b single-layer and component dissection")
    parser.add_argument("--hook-point", default="attn2_out")
    parser.add_argument("--layers", default="0-15")
    parser.add_argument("--layer-group-size", type=int, default=1)
    parser.add_argument("--window-mode", default="early_coarse", choices=["early_coarse", "early_fine"])
    parser.add_argument("--num-pairs", type=int, default=16)
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--out-dir", default="artifacts/e1b_dissection")
    args = parser.parse_args()

    if "-" in args.layers:
        a, b = args.layers.split("-", 1)
        layer_list = list(range(int(a), int(b) + 1))
    else:
        layer_list = [int(x) for x in args.layers.split(",")]

    run_e1b(
        hook_point=args.hook_point,  # type: ignore[arg-type]
        layers=layer_list,
        layer_group_size=args.layer_group_size,
        window_mode=args.window_mode,
        num_pairs=args.num_pairs,
        gen_device=args.gen_device,
        eval_device=args.eval_device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
