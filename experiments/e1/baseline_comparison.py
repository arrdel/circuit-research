"""Day 4 Task 3: Evaluate all 7 baselines on the E1 pilot pairs and compare against causal tau."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from benchmark.counterfactual_pairs import build_spatial_lr_pairs
from evaluation.causal_metrics import (
    compute_pair_causal_outcome,
    summarize_causal_outcomes,
)
from evaluation.spatial_eval import ImageEvalResult, SpatialColorEvaluator
from instrumentation.hooks import ActivationCache, DiTHookManager
from interpretability.baselines import (
    compute_roy_ablation_impact,
    score_heads_from_pair_caches,
    score_heads_revelio_probe,
)
from interventions.patching import (
    AttnMapInjectionController,
    InterventionController,
    InterventionSpec,
)
from models.pixart_sigma import DEFAULT_MODEL_ID, PixArtSigmaRunner


def _find_target_tokens(token_strings: list[str]) -> list[int]:
    idxs = [
        i
        for i, tok in enumerate(token_strings)
        if any(k in tok.replace("▁", "").lower() for k in ("left", "right"))
    ]
    return idxs if idxs else list(range(min(len(token_strings), 12)))


def _top_k_overlap(scores_a: dict[tuple[int, int], float], scores_b: dict[tuple[int, int], float], k: int = 5) -> dict:
    top_a = {key for key, _ in sorted(scores_a.items(), key=lambda x: x[1], reverse=True)[:k]}
    top_b = {key for key, _ in sorted(scores_b.items(), key=lambda x: x[1], reverse=True)[:k]}
    inter = sorted(top_a & top_b)
    return {
        "k": k,
        "top_causal": [f"L{l}H{h}" for l, h in sorted(top_a)],
        "top_baseline": [f"L{l}H{h}" for l, h in sorted(top_b)],
        "overlap_count": len(inter),
        "overlap_fraction": float(len(inter) / k),
        "overlapping_heads": [f"L{l}H{h}" for l, h in inter],
    }


def run_baseline_comparison(
    pilot_results_path: str = "experiments/e1/e1_pilot_results.json",
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
    out_json: str = "experiments/e1/baseline_comparison.json",
    out_md: str = "interpretability/baseline_comparison.md",
) -> dict:
    t0 = time.time()
    with Path(pilot_results_path).open("r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    model_id = pilot_data["checkpoint"]
    pilot_pair_ids = [p["pair_id"] for p in pilot_data["pairs"]]
    seeds = pilot_data["seeds"]
    peak_layers = pilot_data["peak_cell"]["layers"]  # e.g. [10, 11]
    peak_steps = pilot_data["peak_cell"]["steps"]    # e.g. [4, 5, 6, 7]

    # Build lookup of pilot per-(l, h) causal tau
    causal_tau: dict[tuple[int, int], float] = {
        (int(r["layer"]), int(r["head"])): float(r["tau"])
        for r in pilot_data["per_l_h_t_peak"]
    }

    all_explicit = build_spatial_lr_pairs(
        num_dev=24,
        num_test=12,
        num_ood=6,
        templates=("on_the_side_explicit",),
        base_seed=3000,
    )
    pair_map = {p.pair_id: p for p in all_explicit}
    instances = [(pair_map[pid], sd) for pid in pilot_pair_ids for sd in seeds]

    prompts = []
    for p in pair_map.values():
        prompts.extend([p.clean_prompt, p.corrupt_prompt])

    runner = PixArtSigmaRunner(model_id=model_id, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    num_heads = runner.num_heads
    clean_caches: dict[str, ActivationCache] = {}
    corrupt_caches: dict[str, ActivationCache] = {}
    clean_evals: dict[str, ImageEvalResult] = {}
    corrupt_evals: dict[str, ImageEvalResult] = {}

    print(f"[Baselines] Caching activations & attention maps for {len(instances)} pilot instances...")
    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        mgr_c = DiTHookManager(
            runner.transformer,
            layers=peak_layers,
            hook_points=["attn2_head_pre_proj", "attn2_in_txt"],
            steps=peak_steps,
            branch="cond",
            store_cpu=True,
            capture_attn2_probs=True,
        )
        with mgr_c.active() as c_cache:
            res_c = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=sd,
                step_callbacks=[mgr_c],
            )
            mgr_c.cache = ActivationCache()

        mgr_r = DiTHookManager(
            runner.transformer,
            layers=peak_layers,
            hook_points=["attn2_head_pre_proj", "attn2_in_txt"],
            steps=peak_steps,
            branch="cond",
            store_cpu=True,
            capture_attn2_probs=True,
        )
        with mgr_r.active() as r_cache:
            res_r = runner.generate(
                prompt_entry=encoded[pair.corrupt_prompt],
                negative_entry=neg_entry,
                seed=sd,
                step_callbacks=[mgr_r],
            )
            mgr_r.cache = ActivationCache()

        clean_caches[inst_id] = c_cache
        corrupt_caches[inst_id] = r_cache
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

    # Filter to valid counterfactual instances for Roy ablation & P2P
    valid_instances = [
        (p, sd)
        for p, sd in instances
        if clean_evals[f"{p.pair_id}_s{sd}"].dx is not None
        and corrupt_evals[f"{p.pair_id}_s{sd}"].dx is not None
        and clean_evals[f"{p.pair_id}_s{sd}"].relation_pred == p.relation_clean
        and corrupt_evals[f"{p.pair_id}_s{sd}"].relation_pred == p.relation_corrupt
    ]
    base_rel_acc = 1.0
    base_dx_mean = float(np.mean([clean_evals[f"{p.pair_id}_s{sd}"].dx for p, sd in valid_instances]))

    # Baselines 1-4: Zarei (attn_contribution), Maliha & Hougen (kl_spike), CIRCUIT spatial_grad, DAAM (box_mass)
    obs_accum: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        toks = _find_target_tokens(encoded[pair.clean_prompt].token_strings)
        ev_c = clean_evals[inst_id]
        scores = score_heads_from_pair_caches(
            runner.transformer,
            clean_caches[inst_id],
            corrupt_caches[inst_id],
            peak_layers,
            peak_steps,
            toks,
            boxes=[ev_c.box_a, ev_c.box_b],
        )
        for sc in scores:
            obs_accum.setdefault((sc.layer, sc.head), []).append(
                (sc.attn_contribution, sc.kl_spike, sc.spatial_gradient_abs, sc.daam_box_mass)
            )

    # Baseline 6: Revelio Diff-C-style linear probe on pooled activations
    probe_scores = score_heads_revelio_probe(
        [clean_caches[f"{p.pair_id}_s{sd}"] for p, sd in instances],
        [corrupt_caches[f"{p.pair_id}_s{sd}"] for p, sd in instances],
        layers=peak_layers,
        steps=peak_steps,
        num_heads=num_heads,
        hook_point="attn2_head_pre_proj",
    )

    # Baseline 5 (Roy mean-ablation impact) & Baseline 7 (Prompt-to-Prompt A_clean injection) per (l, h)
    mean_cache = ActivationCache.average(list(clean_caches.values()))
    roy_scores: dict[tuple[int, int], float] = {}
    p2p_scores: dict[tuple[int, int], float] = {}

    for l_idx in peak_layers:
        for h_idx in range(num_heads):
            # Roy ablation on clean runs
            abl_rel = []
            abl_dx = []
            for pair, sd in valid_instances:
                spec = InterventionSpec(
                    kind="mean_ablate",
                    hook_point="attn2_head_pre_proj",
                    layers=[l_idx],
                    steps=peak_steps,
                    heads=[h_idx],
                    source_cache=mean_cache,
                    branch="cond",
                )
                ctrl = InterventionController(runner.transformer, [spec])
                with ctrl.active():
                    res_a = runner.generate(
                        prompt_entry=encoded[pair.clean_prompt],
                        negative_entry=neg_entry,
                        seed=sd,
                        step_callbacks=[ctrl],
                    )
                ev_a = evaluator.evaluate_image(
                    res_a.image,
                    obj_a=pair.obj_a,
                    obj_b=pair.obj_b,
                    color_a=pair.color_a_clean,
                    color_b=pair.color_b_clean,
                    expected_relation=pair.relation_clean,
                    compute_embedding=False,
                )
                abl_rel.append(1.0 if ev_a.relation_correct else 0.0)
                if ev_a.dx is not None:
                    abl_dx.append(ev_a.dx)
            imp = compute_roy_ablation_impact(
                base_rel_acc,
                float(np.mean(abl_rel)),
                base_dx_mean,
                float(np.mean(abl_dx)) if abl_dx else None,
            )
            roy_scores[(l_idx, h_idx)] = imp["rel_acc_drop"] + 0.5 * max(0.0, imp["dx_drop"])

            # Prompt-to-Prompt A_clean map injection (without V patching)
            p2p_outcomes = []
            for pair, sd in instances:
                inst_id = f"{pair.pair_id}_s{sd}"
                p2p_ctrl = AttnMapInjectionController(
                    runner.transformer,
                    layers=[l_idx],
                    steps=peak_steps,
                    source_cache=clean_caches[inst_id],
                    heads=[h_idx],
                    branch="cond",
                )
                with p2p_ctrl.active():
                    res_p2p = runner.generate(
                        prompt_entry=encoded[pair.corrupt_prompt],
                        negative_entry=neg_entry,
                        seed=sd,
                        step_callbacks=[p2p_ctrl],
                    )
                ev_p2p = evaluator.evaluate_image(
                    res_p2p.image,
                    obj_a=pair.obj_a,
                    obj_b=pair.obj_b,
                    color_a=pair.color_a_corrupt,
                    color_b=pair.color_b_corrupt,
                    expected_relation=pair.relation_clean,
                )
                p2p_outcomes.append(
                    compute_pair_causal_outcome(
                        pair_id=inst_id,
                        task="spatial_lr",
                        clean=clean_evals[inst_id],
                        corrupt=corrupt_evals[inst_id],
                        patched=ev_p2p,
                        clean_relation=pair.relation_clean,
                        corrupt_relation=pair.relation_corrupt,
                    )
                )
            p2p_summ = summarize_causal_outcomes(p2p_outcomes, valid_only=True)
            p2p_scores[(l_idx, h_idx)] = float(p2p_summ.tau_mean or 0.0)
            print(
                f"[Baseline L{l_idx}H{h_idx:02d}] tau={causal_tau[(l_idx, h_idx)]:.3f} | "
                f"roy={roy_scores[(l_idx, h_idx)]:.3f} | p2p_tau={p2p_scores[(l_idx, h_idx)]:.3f} | "
                f"probe_auc={probe_scores[(l_idx, h_idx)][0]:.3f}"
            )

    # Also test layer-group Prompt-to-Prompt at L10-11 @ W1[4-7] (all 16 heads A_clean swap)
    p2p_group_outcomes = []
    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        p2p_ctrl = AttnMapInjectionController(
            runner.transformer,
            layers=peak_layers,
            steps=peak_steps,
            source_cache=clean_caches[inst_id],
            heads=None,
            branch="cond",
        )
        with p2p_ctrl.active():
            res_g = runner.generate(
                prompt_entry=encoded[pair.corrupt_prompt],
                negative_entry=neg_entry,
                seed=sd,
                step_callbacks=[p2p_ctrl],
            )
        ev_g = evaluator.evaluate_image(
            res_g.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_corrupt,
            color_b=pair.color_b_corrupt,
            expected_relation=pair.relation_clean,
        )
        p2p_group_outcomes.append(
            compute_pair_causal_outcome(
                pair_id=inst_id,
                task="spatial_lr",
                clean=clean_evals[inst_id],
                corrupt=corrupt_evals[inst_id],
                patched=ev_g,
                clean_relation=pair.relation_clean,
                corrupt_relation=pair.relation_corrupt,
            )
        )
    p2p_group_summ = summarize_causal_outcomes(p2p_group_outcomes, valid_only=True).to_dict()

    # Build unified per-head table & Spearman correlations against causal_tau
    grid_rows = []
    baseline_maps: dict[str, dict[tuple[int, int], float]] = {
        "1_zarei_attn_contribution": {},
        "2_maliha_kl_spike": {},
        "3_circuit_spatial_grad_wang_prior": {},
        "4_daam_box_mass": {},
        "5_roy_ablation_impact": {},
        "6_revelio_linear_probe_auc": {},
        "7_prompt_to_prompt_attn_map_tau": {},
    }

    for (l_idx, h_idx), tau_val in sorted(causal_tau.items()):
        obs = np.mean(obs_accum[(l_idx, h_idx)], axis=0)
        auc, acc = probe_scores[(l_idx, h_idx)]
        row = {
            "layer": l_idx,
            "head": h_idx,
            "causal_patching_tau": tau_val,
            "1_zarei_attn_contribution": float(obs[0]),
            "2_maliha_kl_spike": float(obs[1]),
            "3_circuit_spatial_grad_wang_prior": float(obs[2]),
            "4_daam_box_mass": float(obs[3]),
            "5_roy_ablation_impact": float(roy_scores[(l_idx, h_idx)]),
            "6_revelio_linear_probe_auc": float(auc),
            "revelio_probe_acc": float(acc),
            "7_prompt_to_prompt_attn_map_tau": float(p2p_scores[(l_idx, h_idx)]),
        }
        grid_rows.append(row)
        for bname in baseline_maps:
            baseline_maps[bname][(l_idx, h_idx)] = row[bname]

    keys_ordered = sorted(causal_tau.keys())
    y_tau = np.array([causal_tau[k] for k in keys_ordered], dtype=np.float64)

    comparison_summary = {}
    for bname, bmap in baseline_maps.items():
        x_b = np.array([bmap[k] for k in keys_ordered], dtype=np.float64)
        rho, pval = spearmanr(x_b, y_tau)
        ov5 = _top_k_overlap(causal_tau, bmap, k=5)
        comparison_summary[bname] = {
            "spearman_rho": float(0.0 if np.isnan(rho) else rho),
            "p_value": float(1.0 if np.isnan(pval) else pval),
            "top_5_overlap": ov5,
            "max_score": float(np.max(x_b)),
            "mean_score": float(np.mean(x_b)),
        }

    payload = {
        "checkpoint": model_id,
        "checkpoint_label": pilot_data["checkpoint_label"],
        "host": socket.gethostname(),
        "layers": peak_layers,
        "steps": peak_steps,
        "n_instances": len(instances),
        "n_valid_counterfactual": len(valid_instances),
        "revelio_note": "Implemented as Diff-C-style L2 logistic linear probe on spatially-pooled cached per-head activations rather than a full k-SAE.",
        "layer_group_patching_tau": pilot_data["peak_cell"]["tau"],
        "layer_group_prompt_to_prompt_tau": p2p_group_summ,
        "baseline_vs_causal_summary": comparison_summary,
        "per_head_table": grid_rows,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path(out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Write interpretability/baseline_comparison.md (no em dashes)
    md_lines = [
        "# Baseline Comparison Against E1 Pilot Causal Tau Map",
        "",
        f"- Checkpoint: `{model_id}` (`{pilot_data['checkpoint_label']}`)",
        f"- Grid evaluated: `{len(keys_ordered)}` heads across layers `{peak_layers}` at window `W1[4-7]` (`steps {peak_steps}`), `N={len(valid_instances)}` valid counterfactual instances.",
        f"- Layer-group activation patching (`attn2_out` at `L10-11 @ W1[4-7]`): `tau = {pilot_data['peak_cell']['tau']:.4f}`, `flip_rate = {pilot_data['peak_cell']['flip_rate']:.4f}`.",
        f"- Layer-group Prompt-to-Prompt (`A_clean` injection without `V` patching at `L10-11 @ W1[4-7]`): `tau = {(p2p_group_summ['tau_mean'] or 0.0):.4f}`, `flip_rate = {(p2p_group_summ['flip_rate'] or 0.0):.4f}`.",
        "- Note on Revelio baseline: implemented as a Diff-C-style cross-validated linear probe on spatially pooled cached per-head activations rather than training a full k-SAE.",
        "",
        "## Per-Head Rank Agreement with Causal Patching Tau",
        "",
        "| Baseline | Spearman rho | p-value | Top-5 Overlap | Overlapping Heads | Agreement Verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for bname, info in comparison_summary.items():
        rho = info["spearman_rho"]
        pval = info["p_value"]
        ov = info["top_5_overlap"]
        verdict = "Agrees" if (rho > 0.35 and ov["overlap_count"] >= 2) else ("Weak/Partial" if (rho > 0.15 or ov["overlap_count"] >= 1) else "Does not find circuit")
        md_lines.append(
            f"| {bname} | {rho:.3f} | {pval:.4f} | {ov['overlap_count']}/{ov['k']} | {', '.join(ov['overlapping_heads']) or 'none'} | {verdict} |"
        )

    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Observational metrics (Zarei attention contribution, Maliha KL spike, DAAM inside-box mass, Revelio linear probe, and CIRCUIT's spatial gradient diagnostic) measure where token signal correlates with prompt differences or spatial layout, whereas Roy ablation tests necessity and Prompt-to-Prompt tests attention-map transfer alone. Comparing all 7 against causal activation patching (`tau`) on the same `(layer, head, timestep)` grid tests directly which diagnostics recover the true causal circuit heads (`L10H15`, `L10H6`, `L10H13`, `L11H11`, `L11H4`) and whether attention maps alone suffice without value-stream activation patching.",
        ]
    )
    Path(out_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[Baselines] Saved {out_json} and {out_md}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all 7 baselines against E1 pilot tau")
    parser.add_argument("--pilot-json", default="experiments/e1/e1_pilot_results.json")
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--out-json", default="experiments/e1/baseline_comparison.json")
    parser.add_argument("--out-md", default="interpretability/baseline_comparison.md")
    args = parser.parse_args()
    run_baseline_comparison(
        pilot_results_path=args.pilot_json,
        gen_device=args.gen_device,
        eval_device=args.eval_device,
        out_json=args.out_json,
        out_md=args.out_md,
    )


if __name__ == "__main__":
    main()
