"""Day 7 Task 2: Full L0-27 x W0-4 E3 Color-Binding Scan + attn1_out / ff_out check."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import diffusers
import numpy as np
import torch
import transformers

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
from visualization.heatmaps import plot_layer_timestep_heatmap


SEEDS = [42, 123, 456]
WINDOWS: list[tuple[str, list[int]]] = [
    ("W0[0-3]", [0, 1, 2, 3]),
    ("W1[4-7]", [4, 5, 6, 7]),
    ("W2[8-11]", [8, 9, 10, 11]),
    ("W3[12-15]", [12, 13, 14, 15]),
    ("W4[16-19]", [16, 17, 18, 19]),
]


def _select_color_pairs(
    runner: PixArtSigmaRunner,
    encoded: dict,
    evaluator: SpatialColorEvaluator,
    candidates: list[CounterfactualPair],
    target_pairs: int = 4,
) -> list[tuple[CounterfactualPair, int]]:
    neg_entry = encoded[""]
    chosen: list[CounterfactualPair] = []
    for pair in candidates:
        if len(chosen) >= target_pairs:
            break
        valid_seeds = 0
        for sd in SEEDS:
            res_c = runner.generate(prompt_entry=encoded[pair.clean_prompt], negative_entry=neg_entry, seed=sd)
            ev_c = evaluator.evaluate_image(
                res_c.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_clean,
                color_b=pair.color_b_clean,
                expected_relation=pair.relation_clean,
                compute_embedding=False,
            )
            res_r = runner.generate(prompt_entry=encoded[pair.corrupt_prompt], negative_entry=neg_entry, seed=sd)
            ev_r = evaluator.evaluate_image(
                res_r.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_corrupt,
                color_b=pair.color_b_corrupt,
                expected_relation=pair.relation_corrupt,
                compute_embedding=False,
            )
            if not (ev_c.both_detected and ev_c.color_binding_correct and ev_r.both_detected and ev_r.color_binding_correct):
                continue
            ev_rc = evaluator.evaluate_image(
                res_r.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_clean,
                color_b=pair.color_b_clean,
                expected_relation=pair.relation_clean,
                compute_embedding=False,
            )
            mc = (ev_c.color_a_margin or 0.0) + (ev_c.color_b_margin or 0.0)
            mr = (ev_rc.color_a_margin or 0.0) + (ev_rc.color_b_margin or 0.0)
            if (mc - mr) > 0.02:
                valid_seeds += 1
        if valid_seeds >= 2:
            chosen.append(pair)
            print(f"[E3 Scan] Selected {pair.pair_id} ({valid_seeds}/3 valid seeds)")
    return [(p, sd) for p in chosen for sd in SEEDS]


def run_scan_part(
    layer_range: tuple[int, int],
    hook_point: HookPoint = "attn2_out",
    num_pairs: int = 4,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
    out_json: str = "experiments/e3/e3_scan_part.json",
    include_sublayer_checks: bool = False,
) -> dict:
    t0 = time.time()
    all_color = build_color_binding_pairs(
        num_dev=40, num_test=40, num_ood=12, templates=("fixed_lr_swap_colors",), base_seed=6000
    )
    test_candidates = [p for p in all_color if p.split == "test"]
    prompts = [p.clean_prompt for p in test_candidates] + [p.corrupt_prompt for p in test_candidates]

    runner = PixArtSigmaRunner(model_id=DEFAULT_MODEL_ID, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    instances = _select_color_pairs(runner, encoded, evaluator, test_candidates, target_pairs=num_pairs)
    target_layers = list(range(layer_range[0], layer_range[1] + 1))
    layer_groups = [(f"L{s}-{s+1}", [s, s + 1]) for s in range(layer_range[0], layer_range[1] + 1, 2)]

    hook_points_to_cache: list[HookPoint] = [hook_point]
    if include_sublayer_checks:
        hook_points_to_cache = sorted(set(hook_points_to_cache + ["attn1_out", "ff_out"]))  # type: ignore[assignment]

    clean_caches: dict[str, ActivationCache] = {}
    clean_evals: dict[str, ImageEvalResult] = {}
    corrupt_evals: dict[str, ImageEvalResult] = {}

    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        cache_layers = sorted(set(target_layers + ([2, 3, 6, 7, 10, 11] if include_sublayer_checks else [])))
        mgr = DiTHookManager(
            runner.transformer,
            layers=cache_layers,
            hook_points=hook_points_to_cache,
            steps=list(range(20)),
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
        res_r = runner.generate(prompt_entry=encoded[pair.corrupt_prompt], negative_entry=neg_entry, seed=sd)
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
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )

    rows: list[dict] = []
    for l_lbl, l_idxs in layer_groups:
        for w_lbl, s_idxs in WINDOWS:
            outcomes: list[PairCausalOutcome] = []
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
                        step_callbacks=[ctrl],
                    )
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
                        pair_id=inst_id,
                        task="color_binding",
                        clean=clean_evals[inst_id],
                        corrupt=corrupt_evals[inst_id],
                        patched=ev_p,
                        clean_relation=pair.relation_clean,
                        corrupt_relation=pair.relation_corrupt,
                    )
                )
            summ = summarize_causal_outcomes(outcomes, valid_only=True)
            row = {
                "hook_point": hook_point,
                "layer_group": l_lbl,
                "layers": l_idxs,
                "window": w_lbl,
                "steps": s_idxs,
                "tau": summ.tau_mean if summ.tau_mean is not None else 0.0,
                "tau_ci_low": summ.tau_ci_low,
                "tau_ci_high": summ.tau_ci_high,
                "sigma": summ.sigma_mean if summ.sigma_mean is not None else 0.0,
                "flip_rate": summ.flip_rate if summ.flip_rate is not None else 0.0,
                "detection_rate": summ.both_detected_rate,
                "n_valid": summ.n_valid,
            }
            rows.append(row)
            print(f"[E3 Scan] {l_lbl:>6} x {w_lbl:<9} | tau_color={row['tau']:.4f} flip={row['flip_rate']:.2f} sigma={row['sigma']:.3f}")

    sublayer_rows: list[dict] = []
    if include_sublayer_checks:
        sub_checks = [
            ("L10-11", [10, 11], "W1[4-7]", [4, 5, 6, 7]),
            ("L10-11", [10, 11], "W2[8-11]", [8, 9, 10, 11]),
            ("L6-7", [6, 7], "W2[8-11]", [8, 9, 10, 11]),
            ("L2-3", [2, 3], "W1[4-7]", [4, 5, 6, 7]),
        ]
        for l_lbl, l_idxs, w_lbl, s_idxs in sub_checks:
            for hp in ["attn1_out", "ff_out"]:
                outcomes = []
                for pair, sd in instances:
                    inst_id = f"{pair.pair_id}_s{sd}"
                    spec = InterventionSpec(
                        kind="patch",
                        hook_point=hp,  # type: ignore[arg-type]
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
                            patched=ev_p,
                            clean_relation=pair.relation_clean,
                            corrupt_relation=pair.relation_corrupt,
                        )
                    )
                summ = summarize_causal_outcomes(outcomes, valid_only=True)
                srow = {
                    "hook_point": hp,
                    "layer_group": l_lbl,
                    "layers": l_idxs,
                    "window": w_lbl,
                    "steps": s_idxs,
                    "tau": summ.tau_mean if summ.tau_mean is not None else 0.0,
                    "tau_ci_low": summ.tau_ci_low,
                    "tau_ci_high": summ.tau_ci_high,
                    "sigma": summ.sigma_mean if summ.sigma_mean is not None else 0.0,
                    "flip_rate": summ.flip_rate if summ.flip_rate is not None else 0.0,
                    "detection_rate": summ.both_detected_rate,
                    "n_valid": summ.n_valid,
                }
                sublayer_rows.append(srow)
                print(f"[E3 Sublayer] {hp:<10} {l_lbl:>6}@{w_lbl:<9} | tau_color={srow['tau']:.4f} flip={srow['flip_rate']:.2f}")

    payload = {
        "checkpoint": DEFAULT_MODEL_ID,
        "checkpoint_label": "512-MS",
        "host": socket.gethostname(),
        "pairs": [p.pair_id for p, _ in instances[:: len(SEEDS)]],
        "seeds": SEEDS,
        "n_valid": rows[0]["n_valid"] if rows else 0,
        "rows": rows,
        "sublayer_rows": sublayer_rows,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path(out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def merge_scan(
    part_paths: list[str],
    out_json: str = "experiments/e3/e3_color_scan.json",
    out_log: str = "experiments/e3/e3_color_scan_run_log.md",
    out_fig: str = "experiments/e3/images/e3_color_scan_tau.png",
) -> dict:
    parts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in part_paths]
    all_rows = sorted(
        [r for p in parts for r in p["rows"]],
        key=lambda r: (r["layers"][0], r["steps"][0]),
    )
    sublayer_rows = [r for p in parts for r in p.get("sublayer_rows", [])]

    layer_groups = sorted({(r["layers"][0], r["layer_group"]) for r in all_rows})
    w_labels = [w[0] for w in WINDOWS]
    mat = np.zeros((len(layer_groups), len(w_labels)), dtype=np.float64)
    l_to_i = {lbl: i for i, (_, lbl) in enumerate(layer_groups)}
    w_to_j = {w: j for j, w in enumerate(w_labels)}
    for r in all_rows:
        mat[l_to_i[r["layer_group"]], w_to_j[r["window"]]] = r["tau"]

    plot_layer_timestep_heatmap(
        mat,
        layers=[l0 for l0, _ in layer_groups],
        step_labels=w_labels,
        title="E3 Color-Binding Full Scan (tau) [attn2_out, 512-MS, test split]",
        cbar_label="Color Transfer (tau)",
        out_path=out_fig,
        cmap="magma",
    )

    top_all = sorted(all_rows, key=lambda r: r["tau"], reverse=True)[:10]
    outside_l1011 = [r for r in all_rows if not (r["layer_group"] == "L10-11" and r["window"] == "W1[4-7]")]
    top_outside = sorted(outside_l1011, key=lambda r: r["tau"], reverse=True)[:5]
    beats_l11h1 = [r for r in outside_l1011 + sublayer_rows if r["tau"] > 0.3820]

    payload = {
        "checkpoint": DEFAULT_MODEL_ID,
        "checkpoint_label": "512-MS",
        "host": parts[0]["host"],
        "pairs": parts[0]["pairs"],
        "seeds": parts[0]["seeds"],
        "n_valid": parts[0]["n_valid"],
        "rows": all_rows,
        "sublayer_rows": sublayer_rows,
        "top_10_attn2_cells": top_all,
        "top_5_outside_L10_11_W1": top_outside,
        "anything_outside_L10_11_beats_L11H01_0382": bool(beats_l11h1),
        "cells_beating_L11H01": beats_l11h1,
    }
    Path(out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# E3 Full L0-27 x W0-4 Color-Binding Scan Run Log",
        "",
        f"- Host: `{parts[0]['host']}`",
        f"- Checkpoint: `{DEFAULT_MODEL_ID}` (`512-MS`)",
        f"- Pairs: `{parts[0]['pairs']}`, Seeds: `{parts[0]['seeds']}`, Valid counterfactuals: `{parts[0]['n_valid']}`",
        f"- Reference single head `L11H01 @ W1[4-7]` (from `e3_full_results.json`): `tau = 0.3820`",
        "",
        "## Top 10 `attn2_out` Cells Across L0-27 x W0-4",
        "",
        "| Layer Group | Window | Tau (Color) | Flip Rate | Sigma | Detection Rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in top_all:
        lines.append(
            f"| {r['layer_group']} | {r['window']} | {r['tau']:.4f} | {r['flip_rate']:.4f} | {r['sigma']:.4f} | {r['detection_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Sublayer Checks (`attn1_out` and `ff_out`)",
            "",
            "| Hook Point | Layer Group | Window | Tau (Color) | Flip Rate | Sigma |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in sublayer_rows:
        lines.append(
            f"| {r['hook_point']} | {r['layer_group']} | {r['window']} | {r['tau']:.4f} | {r['flip_rate']:.4f} | {r['sigma']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Anything outside `L10-11 @ W1[4-7]` beating `L11H01` (`0.3820`): `{bool(beats_l11h1)}`",
        ]
    )
    Path(out_log).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[E3 Scan] Merged -> {out_json} & {out_log}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 full L0-27 x W0-4 color-binding scan")
    parser.add_argument("--layers", default="0-13")
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--include-sublayers", action="store_true")
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--out-json", default="experiments/e3/e3_scan_part.json")
    parser.add_argument("--merge", nargs="+", default=None)
    args = parser.parse_args()

    if args.merge:
        merge_scan(args.merge)
        return

    a, b = args.layers.split("-", 1)
    run_scan_part(
        layer_range=(int(a), int(b)),
        num_pairs=args.num_pairs,
        gen_device=args.gen_device,
        eval_device=args.eval_device,
        out_json=args.out_json,
        include_sublayer_checks=args.include_sublayers,
    )


if __name__ == "__main__":
    main()
