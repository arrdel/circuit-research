"""Compare all 7 baselines against E1 pilot or full-sweep causal tau map."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
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
from models.pixart_sigma import PixArtSigmaRunner


def _find_target_tokens(token_strings: list[str]) -> list[int]:
    idxs = [
        i
        for i, tok in enumerate(token_strings)
        if any(k in tok.replace("▁", "").lower() for k in ("left", "right"))
    ]
    return idxs if idxs else list(range(min(len(token_strings), 12)))


def _top_k_overlap(
    scores_a: dict[tuple[int, int], float],
    scores_b: dict[tuple[int, int], float],
    k: int = 5,
) -> dict:
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


def _summarize_subset(
    causal_tau: dict[tuple[int, int], float],
    grid_rows: list[dict],
    baseline_keys: list[str],
) -> dict:
    keys_ordered = sorted(causal_tau.keys())
    y_tau = np.array([causal_tau[k] for k in keys_ordered], dtype=np.float64)
    row_by_key = {(int(r["layer"]), int(r["head"])): r for r in grid_rows}

    summary = {}
    for bname in baseline_keys:
        bmap = {k: float(row_by_key[k][bname]) for k in keys_ordered}
        x_b = np.array([bmap[k] for k in keys_ordered], dtype=np.float64)
        rho, pval = spearmanr(x_b, y_tau)
        ov5 = _top_k_overlap(causal_tau, bmap, k=5)
        summary[bname] = {
            "spearman_rho": float(0.0 if np.isnan(rho) else rho),
            "p_value": float(1.0 if np.isnan(pval) else pval),
            "top_5_overlap": ov5,
            "max_score": float(np.max(x_b)),
            "mean_score": float(np.mean(x_b)),
        }
    return summary


def run_baseline_comparison(
    results_path: str = "experiments/e1/e1_full_results.json",
    layers_filter: list[int] | None = None,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
    out_json: str = "experiments/e1/baseline_comparison_full.json",
    out_md: str | None = "interpretability/baseline_comparison.md",
) -> dict:
    t0 = time.time()
    with Path(results_path).open("r", encoding="utf-8") as f:
        src_data = json.load(f)

    model_id = src_data["checkpoint"]
    pair_ids = [p["pair_id"] for p in src_data["pairs"]]
    seeds = src_data["seeds"]
    head_list = src_data.get("per_l_h_t", src_data.get("per_l_h_t_peak", []))
    peak_steps = src_data["peak_cell"]["steps"]

    all_causal_tau: dict[tuple[int, int], float] = {
        (int(r["layer"]), int(r["head"])): float(r["tau"])
        for r in head_list
    }
    target_layers = sorted({l for l, _ in all_causal_tau.keys()})
    if layers_filter is not None:
        target_layers = [l for l in target_layers if l in layers_filter]
    causal_tau = {k: v for k, v in all_causal_tau.items() if k[0] in target_layers}

    all_explicit = build_spatial_lr_pairs(
        num_dev=24,
        num_test=36,
        num_ood=12,
        templates=("on_the_side_explicit",),
        base_seed=3000,
    )
    pair_map = {p.pair_id: p for p in all_explicit}
    instances = [(pair_map[pid], sd) for pid in pair_ids for sd in seeds]

    prompts = []
    for p, _ in instances[:: len(seeds)]:
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

    print(f"[Baselines] Caching activations & attn maps for {len(instances)} instances across layers {target_layers}...")
    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        mgr_c = DiTHookManager(
            runner.transformer,
            layers=target_layers,
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
            layers=target_layers,
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

    # Baselines 1-4
    obs_accum: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        toks = _find_target_tokens(encoded[pair.clean_prompt].token_strings)
        ev_c = clean_evals[inst_id]
        scores = score_heads_from_pair_caches(
            runner.transformer,
            clean_caches[inst_id],
            corrupt_caches[inst_id],
            target_layers,
            peak_steps,
            toks,
            boxes=[ev_c.box_a, ev_c.box_b],
        )
        for sc in scores:
            obs_accum.setdefault((sc.layer, sc.head), []).append(
                (sc.attn_contribution, sc.kl_spike, sc.spatial_gradient_abs, sc.daam_box_mass)
            )

    # Baseline 6: Revelio Diff-C linear probe
    probe_scores = score_heads_revelio_probe(
        [clean_caches[f"{p.pair_id}_s{sd}"] for p, sd in instances],
        [corrupt_caches[f"{p.pair_id}_s{sd}"] for p, sd in instances],
        layers=target_layers,
        steps=peak_steps,
        num_heads=num_heads,
        hook_point="attn2_head_pre_proj",
    )

    # Baselines 5 & 7 per (l, h)
    mean_cache = ActivationCache.average(list(clean_caches.values()))
    roy_scores: dict[tuple[int, int], float] = {}
    p2p_scores: dict[tuple[int, int], float] = {}

    for l_idx in target_layers:
        for h_idx in range(num_heads):
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
                f"[Baseline L{l_idx:02d}H{h_idx:02d}] tau={causal_tau[(l_idx, h_idx)]:.3f} | "
                f"roy={roy_scores[(l_idx, h_idx)]:.3f} | p2p_tau={p2p_scores[(l_idx, h_idx)]:.3f} | "
                f"daam={np.mean(obs_accum[(l_idx, h_idx)], axis=0)[3]:.3f}"
            )

    # Layer-group Prompt-to-Prompt checks
    p2p_group_results: dict[str, dict] = {}
    for grp_name, grp_layers in [("L10-11", [10, 11]), ("L2-3", [2, 3])]:
        if not all(l in target_layers for l in grp_layers):
            continue
        p2p_outcomes = []
        for pair, sd in instances:
            inst_id = f"{pair.pair_id}_s{sd}"
            p2p_ctrl = AttnMapInjectionController(
                runner.transformer,
                layers=grp_layers,
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
            p2p_outcomes.append(
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
        p2p_group_results[grp_name] = summarize_causal_outcomes(p2p_outcomes, valid_only=True).to_dict()

    baseline_keys = [
        "1_zarei_attn_contribution",
        "2_maliha_kl_spike",
        "3_circuit_spatial_grad_wang_prior",
        "4_daam_box_mass",
        "5_roy_ablation_impact",
        "6_revelio_linear_probe_auc",
        "7_prompt_to_prompt_attn_map_tau",
    ]
    grid_rows = []
    for (l_idx, h_idx), tau_val in sorted(causal_tau.items()):
        obs = np.mean(obs_accum[(l_idx, h_idx)], axis=0)
        auc, acc = probe_scores[(l_idx, h_idx)]
        grid_rows.append(
            {
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
        )

    payload = {
        "checkpoint": model_id,
        "checkpoint_label": src_data["checkpoint_label"],
        "source_results": results_path,
        "host": socket.gethostname(),
        "layers": target_layers,
        "steps": peak_steps,
        "n_instances": len(instances),
        "n_valid_counterfactual": len(valid_instances),
        "revelio_note": "Implemented as Diff-C-style L2 logistic linear probe on spatially-pooled cached per-head activations rather than a full k-SAE.",
        "layer_group_patching_tau": {
            "L10-11": src_data["peak_cell"]["tau"],
            "L2-3": src_data["l2_3_replication"]["tau"] if src_data.get("l2_3_replication") else None,
        },
        "layer_group_prompt_to_prompt": p2p_group_results,
        "baseline_vs_causal_summary": _summarize_subset(causal_tau, grid_rows, baseline_keys),
        "by_layer_group_summary": {
            "L2-3": _summarize_subset(
                {k: v for k, v in causal_tau.items() if k[0] in (2, 3)},
                [r for r in grid_rows if r["layer"] in (2, 3)],
                baseline_keys,
            )
            if any(l in target_layers for l in (2, 3))
            else {},
            "L10-11": _summarize_subset(
                {k: v for k, v in causal_tau.items() if k[0] in (10, 11)},
                [r for r in grid_rows if r["layer"] in (10, 11)],
                baseline_keys,
            )
            if any(l in target_layers for l in (10, 11))
            else {},
        },
        "per_head_table": grid_rows,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path(out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if out_md is not None:
        write_comparison_markdown(payload, out_md=out_md)
    print(f"[Baselines] Saved {out_json}")
    return payload


def merge_partial_results(
    part_paths: list[str],
    results_path: str,
    out_json: str,
    out_md: str,
) -> dict:
    with Path(results_path).open("r", encoding="utf-8") as f:
        src_data = json.load(f)
    parts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in part_paths]
    grid_rows = sorted(
        [r for p in parts for r in p["per_head_table"]],
        key=lambda r: (r["layer"], r["head"]),
    )
    causal_tau = {(int(r["layer"]), int(r["head"])): float(r["causal_patching_tau"]) for r in grid_rows}
    baseline_keys = list(parts[0]["baseline_vs_causal_summary"].keys())
    p2p_groups = {}
    for p in parts:
        p2p_groups.update(p.get("layer_group_prompt_to_prompt", {}))

    payload = {
        "checkpoint": src_data["checkpoint"],
        "checkpoint_label": src_data["checkpoint_label"],
        "source_results": results_path,
        "host": parts[0]["host"],
        "layers": sorted({int(r["layer"]) for r in grid_rows}),
        "steps": parts[0]["steps"],
        "n_instances": parts[0]["n_instances"],
        "n_valid_counterfactual": parts[0]["n_valid_counterfactual"],
        "revelio_note": parts[0]["revelio_note"],
        "layer_group_patching_tau": {
            "L10-11": src_data["peak_cell"]["tau"],
            "L2-3": src_data["l2_3_replication"]["tau"] if src_data.get("l2_3_replication") else None,
        },
        "layer_group_prompt_to_prompt": p2p_groups,
        "baseline_vs_causal_summary": _summarize_subset(causal_tau, grid_rows, baseline_keys),
        "by_layer_group_summary": {
            "L2-3": _summarize_subset(
                {k: v for k, v in causal_tau.items() if k[0] in (2, 3)},
                [r for r in grid_rows if r["layer"] in (2, 3)],
                baseline_keys,
            ),
            "L10-11": _summarize_subset(
                {k: v for k, v in causal_tau.items() if k[0] in (10, 11)},
                [r for r in grid_rows if r["layer"] in (10, 11)],
                baseline_keys,
            ),
        },
        "per_head_table": grid_rows,
        "elapsed_sec": round(sum(p["elapsed_sec"] for p in parts), 2),
    }
    Path(out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_comparison_markdown(payload, out_md=out_md)
    print(f"[Baselines] Merged {part_paths} -> {out_json} & {out_md}")
    return payload


def write_comparison_markdown(full_payload: dict, out_md: str = "interpretability/baseline_comparison.md") -> None:
    pilot_path = Path("experiments/e1/baseline_comparison.json")
    pilot_payload = json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.exists() else None

    lines = [
        "# Baseline Comparison Against E1 Causal Tau Maps",
        "",
        "## 1. Full Sweep Comparison (`test` split, 64 heads across L2, L3, L10, L11 @ W1[4-7])",
        "",
        f"- Checkpoint: `{full_payload['checkpoint']}` (`{full_payload['checkpoint_label']}`)",
        f"- Valid counterfactuals: `N={full_payload['n_valid_counterfactual']}` (from `18` instances, `6` test pairs x `3` seeds).",
        f"- `L10-11 @ W1[4-7]`: activation patching `tau = {full_payload['layer_group_patching_tau']['L10-11']:.4f}` vs Prompt-to-Prompt `A_clean` injection `tau = {full_payload['layer_group_prompt_to_prompt']['L10-11']['tau_mean']:.4f}` (flip rate `{(full_payload['layer_group_prompt_to_prompt']['L10-11']['flip_rate'] or 0.0):.4f}`).",
        f"- `L2-3 @ W1[4-7]`: activation patching `tau = {full_payload['layer_group_patching_tau']['L2-3']:.4f}` vs Prompt-to-Prompt `A_clean` injection `tau = {full_payload['layer_group_prompt_to_prompt']['L2-3']['tau_mean']:.4f}` (flip rate `{(full_payload['layer_group_prompt_to_prompt']['L2-3']['flip_rate'] or 0.0):.4f}`).",
        "- Note on Revelio baseline: implemented as a Diff-C-style cross-validated linear probe on spatially pooled cached per-head activations rather than training a full k-SAE.",
        "",
        "### All 64 Heads (`L2, L3, L10, L11`)",
        "",
        "| Baseline | Spearman rho | p-value | Top-5 Overlap | Overlapping Heads | Verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for bname, info in full_payload["baseline_vs_causal_summary"].items():
        rho = info["spearman_rho"]
        pval = info["p_value"]
        ov = info["top_5_overlap"]
        verdict = "Agrees" if (rho > 0.35 and ov["overlap_count"] >= 2) else ("Weak/Partial" if (rho > 0.15 or ov["overlap_count"] >= 1) else "Does not find circuit")
        lines.append(f"| {bname} | {rho:.3f} | {pval:.4f} | {ov['overlap_count']}/{ov['k']} | {', '.join(ov['overlapping_heads']) or 'none'} | {verdict} |")

    for grp in ["L2-3", "L10-11"]:
        sub = full_payload["by_layer_group_summary"].get(grp, {})
        if not sub:
            continue
        lines.extend(
            [
                "",
                f"### Subset: `{grp}` (32 heads)",
                "",
                "| Baseline | Spearman rho | p-value | Top-5 Overlap | Overlapping Heads |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for bname, info in sub.items():
            ov = info["top_5_overlap"]
            lines.append(
                f"| {bname} | {info['spearman_rho']:.3f} | {info['p_value']:.4f} | {ov['overlap_count']}/{ov['k']} | {', '.join(ov['overlapping_heads']) or 'none'} |"
            )

    if pilot_payload is not None:
        lines.extend(
            [
                "",
                "## 2. Pilot Comparison Reference (`dev` split, 32 heads in L10-11 @ W1[4-7], N=6 valid)",
                "",
                f"- Layer-group patching `tau = {pilot_payload['layer_group_patching_tau']:.4f}` vs Prompt-to-Prompt `tau = {pilot_payload['layer_group_prompt_to_prompt_tau']['tau_mean']:.4f}`.",
                "",
                "| Baseline | Spearman rho | p-value | Top-5 Overlap |",
                "| --- | --- | --- | --- |",
            ]
        )
        for bname, info in pilot_payload["baseline_vs_causal_summary"].items():
            ov = info["top_5_overlap"]
            lines.append(f"| {bname} | {info['spearman_rho']:.3f} | {info['p_value']:.4f} | {ov['overlap_count']}/{ov['k']} |")

    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all 7 baselines against E1 tau map")
    parser.add_argument("--results-json", default="experiments/e1/e1_full_results.json")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--merge", nargs="+", default=None)
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--out-json", default="experiments/e1/baseline_comparison_full.json")
    parser.add_argument("--out-md", default="interpretability/baseline_comparison.md")
    args = parser.parse_args()

    if args.merge:
        merge_partial_results(args.merge, args.results_json, args.out_json, args.out_md)
        return

    layers_filter = [int(x) for x in args.layers.split(",")] if args.layers else None
    run_baseline_comparison(
        results_path=args.results_json,
        layers_filter=layers_filter,
        gen_device=args.gen_device,
        eval_device=args.eval_device,
        out_json=args.out_json,
        out_md=args.out_md,
    )


if __name__ == "__main__":
    main()
