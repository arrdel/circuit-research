"""E2: Head-level causal discovery, observational baseline comparison, and held-out necessity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from benchmark.counterfactual_pairs import CounterfactualPair, load_pairs_jsonl
from evaluation.causal_metrics import (
    compute_pair_causal_outcome,
    summarize_causal_outcomes,
)
from evaluation.spatial_eval import ImageEvalResult, SpatialColorEvaluator
from instrumentation.hooks import ActivationCache, DiTHookManager
from interpretability.baselines import score_heads_from_pair_caches
from interventions.patching import InterventionController, InterventionSpec
from models.pixart_sigma import DEFAULT_MODEL_ID, PixArtSigmaRunner
from visualization.heatmaps import plot_layer_timestep_heatmap


def _find_relation_token_indices(token_strings: list[str]) -> list[int]:
    """Locate token indices corresponding to spatial relation cues ('left', 'right')."""
    idxs: list[int] = []
    for i, tok in enumerate(token_strings):
        clean_t = tok.replace("▁", "").lower().strip()
        if any(k in clean_t for k in ("left", "right")):
            idxs.append(i)
    return idxs if idxs else list(range(min(len(token_strings), 12)))


def _collect_valid_pairs(
    pairs: list[CounterfactualPair],
    runner: PixArtSigmaRunner,
    encoded: dict,
    evaluator: SpatialColorEvaluator,
    target_layers: list[int],
    steps: list[int],
    max_pairs: int,
    capture_obs: bool = False,
) -> tuple[
    list[CounterfactualPair],
    dict[str, ActivationCache],
    dict[str, ActivationCache],
    dict[str, ImageEvalResult],
    dict[str, ImageEvalResult],
]:
    neg_entry = encoded[""]
    valid: list[CounterfactualPair] = []
    clean_caches: dict[str, ActivationCache] = {}
    corrupt_caches: dict[str, ActivationCache] = {}
    clean_evals: dict[str, ImageEvalResult] = {}
    corrupt_evals: dict[str, ImageEvalResult] = {}

    hook_points = ["attn2_head_pre_proj"]
    if capture_obs:
        hook_points.append("attn2_in_txt")

    for pair in pairs:
        if len(valid) >= max_pairs:
            break

        if capture_obs:
            mgr_r = DiTHookManager(
                runner.transformer,
                layers=target_layers,
                hook_points=hook_points,
                steps=steps,
                branch="cond",
                store_cpu=True,
                capture_attn2_probs=True,
            )
            with mgr_r.active() as r_cache:
                res_r = runner.generate(
                    prompt_entry=encoded[pair.corrupt_prompt],
                    negative_entry=neg_entry,
                    seed=pair.seed,
                    step_callbacks=[mgr_r],
                )
                mgr_r.cache = ActivationCache()
        else:
            r_cache = ActivationCache()
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
        if not (ev_r.dx is not None and ev_r.relation_pred == pair.relation_corrupt):
            continue

        mgr_c = DiTHookManager(
            runner.transformer,
            layers=target_layers,
            hook_points=hook_points,
            steps=steps,
            branch="cond",
            store_cpu=True,
            capture_attn2_probs=capture_obs,
        )
        with mgr_c.active() as c_cache:
            res_c = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=pair.seed,
                step_callbacks=[mgr_c],
            )
            mgr_c.cache = ActivationCache()

        ev_c = evaluator.evaluate_image(
            res_c.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )
        if not (
            ev_c.dx is not None
            and ev_c.relation_pred == pair.relation_clean
            and (ev_c.dx - ev_r.dx) > 0.04
        ):
            continue

        valid.append(pair)
        clean_caches[pair.pair_id] = c_cache
        corrupt_caches[pair.pair_id] = r_cache
        clean_evals[pair.pair_id] = ev_c
        corrupt_evals[pair.pair_id] = ev_r

    return valid, clean_caches, corrupt_caches, clean_evals, corrupt_evals


def _eval_head_subset_patching(
    pairs: list[CounterfactualPair],
    head_specs: list[tuple[int, int]],
    steps: list[int],
    runner: PixArtSigmaRunner,
    encoded: dict,
    evaluator: SpatialColorEvaluator,
    clean_caches: dict[str, ActivationCache],
    clean_evals: dict[str, ImageEvalResult],
    corrupt_evals: dict[str, ImageEvalResult],
) -> dict:
    neg_entry = encoded[""]
    by_layer: dict[int, list[int]] = {}
    for l_idx, h_idx in head_specs:
        by_layer.setdefault(l_idx, []).append(h_idx)

    outcomes = []
    for pair in pairs:
        specs = [
            InterventionSpec(
                kind="patch",
                hook_point="attn2_head_pre_proj",
                layers=[l_idx],
                steps=steps,
                heads=h_list,
                source_cache=clean_caches[pair.pair_id],
                branch="cond",
            )
            for l_idx, h_list in by_layer.items()
        ]
        ctrl = InterventionController(runner.transformer, specs)
        with ctrl.active():
            res_p = runner.generate(
                prompt_entry=encoded[pair.corrupt_prompt],
                negative_entry=neg_entry,
                seed=pair.seed,
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
    return summarize_causal_outcomes(outcomes, valid_only=True).to_dict()


def _eval_head_subset_ablation(
    pairs: list[CounterfactualPair],
    head_specs: list[tuple[int, int]],
    steps: list[int],
    runner: PixArtSigmaRunner,
    encoded: dict,
    evaluator: SpatialColorEvaluator,
    mean_cache: ActivationCache,
    mode: str = "mean_ablate",
) -> dict:
    neg_entry = encoded[""]
    by_layer: dict[int, list[int]] = {}
    for l_idx, h_idx in head_specs:
        by_layer.setdefault(l_idx, []).append(h_idx)

    rel_ok_list: list[float] = []
    det_ok_list: list[float] = []
    color_ok_list: list[float] = []

    for pair in pairs:
        specs = [
            InterventionSpec(
                kind=mode,  # type: ignore[arg-type]
                hook_point="attn2_head_pre_proj",
                layers=[l_idx],
                steps=steps,
                heads=h_list,
                source_cache=mean_cache if mode == "mean_ablate" else None,
                branch="cond",
            )
            for l_idx, h_list in by_layer.items()
        ]
        ctrl = InterventionController(runner.transformer, specs)
        with ctrl.active():
            res = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=pair.seed,
                step_callbacks=[ctrl],
            )
        ev = evaluator.evaluate_image(
            res.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
            compute_embedding=False,
        )
        det_ok_list.append(1.0 if ev.both_detected else 0.0)
        rel_ok_list.append(1.0 if ev.relation_correct else 0.0)
        if ev.color_binding_correct is not None:
            color_ok_list.append(1.0 if ev.color_binding_correct else 0.0)

    return {
        "n_pairs": len(pairs),
        "rel_acc": float(np.mean(rel_ok_list)),
        "both_detected_rate": float(np.mean(det_ok_list)),
        "color_acc": float(np.mean(color_ok_list)) if color_ok_list else None,
    }


def run_e2(
    model_id: str = DEFAULT_MODEL_ID,
    pairs_path: str = "artifacts/data/spatial_lr_rich.jsonl",
    layers: list[int] | None = None,
    steps: list[int] | None = None,
    num_dev_pairs: int = 12,
    num_test_pairs: int = 20,
    top_k: int = 6,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
    out_dir: str = "artifacts/e2_head_discovery",
) -> dict:
    t0 = time.time()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    target_layers = layers or [2, 3, 10, 11]
    target_steps = steps or [0, 1, 2, 3, 4, 5, 6, 7]

    all_pairs = load_pairs_jsonl(pairs_path)
    dev_pool = [p for p in all_pairs if p.split == "dev"]
    test_pool = [p for p in all_pairs if p.split in ("test", "ood")]

    prompts = []
    for p in dev_pool + test_pool:
        prompts.extend([p.clean_prompt, p.corrupt_prompt])

    runner = PixArtSigmaRunner(model_id=model_id, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    print(f"[E2] Collecting dev pairs (target={num_dev_pairs}) & test/ood pairs (target={num_test_pairs})...")
    dev_pairs, dev_c_caches, dev_r_caches, dev_c_evals, dev_r_evals = _collect_valid_pairs(
        dev_pool, runner, encoded, evaluator, target_layers, target_steps, num_dev_pairs, capture_obs=True
    )
    test_pairs, test_c_caches, _, test_c_evals, test_r_evals = _collect_valid_pairs(
        test_pool, runner, encoded, evaluator, target_layers, target_steps, num_test_pairs, capture_obs=False
    )
    print(f"[E2] Valid pairs: dev={len(dev_pairs)}, test/ood={len(test_pairs)}")

    # 1. Observational baselines on dev pairs
    obs_acc: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for pair in dev_pairs:
        rel_toks = _find_relation_token_indices(encoded[pair.clean_prompt].token_strings)
        scores = score_heads_from_pair_caches(
            runner.transformer,
            dev_c_caches[pair.pair_id],
            dev_r_caches[pair.pair_id],
            target_layers,
            target_steps,
            rel_toks,
        )
        for sc in scores:
            obs_acc.setdefault((sc.layer, sc.head), []).append(
                (sc.attn_contribution, sc.kl_spike, sc.spatial_gradient_abs)
            )

    # 2. Causal single-head patching across all (layer, head) on dev pairs
    num_heads = runner.num_heads
    tau_mat = np.zeros((len(target_layers), num_heads), dtype=np.float64)
    head_rows: list[dict] = []

    for li, l_idx in enumerate(target_layers):
        for h_idx in range(num_heads):
            summ = _eval_head_subset_patching(
                dev_pairs,
                [(l_idx, h_idx)],
                target_steps,
                runner,
                encoded,
                evaluator,
                dev_c_caches,
                dev_c_evals,
                dev_r_evals,
            )
            tau_v = summ["tau_mean"] or 0.0
            tau_mat[li, h_idx] = tau_v
            obs_vals = np.mean(obs_acc[(l_idx, h_idx)], axis=0)
            row = {
                "layer": l_idx,
                "head": h_idx,
                "tau_dev": tau_v,
                "flip_dev": summ["flip_rate"] or 0.0,
                "sigma_dev": summ["sigma_mean"] or 0.0,
                "det_dev": summ["both_detected_rate"],
                "attn_contribution": float(obs_vals[0]),
                "kl_spike": float(obs_vals[1]),
                "spatial_gradient_abs": float(obs_vals[2]),
            }
            head_rows.append(row)
            print(
                f"[E2 Head] L{l_idx:02d}H{h_idx:02d} | tau={tau_v:.3f} flip={row['flip_dev']:.2f} "
                f"contrib={row['attn_contribution']:.3f} kl={row['kl_spike']:.3f} grad={row['spatial_gradient_abs']:.3f}"
            )

    df_heads = pd.DataFrame(head_rows)
    df_heads.to_csv(out_path / "head_scores.csv", index=False)

    # Plot Layer x Head causal tau map
    plot_layer_timestep_heatmap(
        tau_mat,
        layers=target_layers,
        step_labels=[f"H{h}" for h in range(num_heads)],
        title=f"E2 Causal Head Transfer (tau) on Dev [steps {target_steps[0]}-{target_steps[-1]}]",
        cbar_label="Transfer Score (tau)",
        out_path=fig_dir / "head_tau_heatmap.png",
        cmap="magma",
    )

    # Spearman correlations between observational baselines and causal tau_dev
    corr_stats = {}
    for col in ["attn_contribution", "kl_spike", "spatial_gradient_abs"]:
        rho, pval = spearmanr(df_heads[col].values, df_heads["tau_dev"].values)
        corr_stats[col] = {"spearman_rho": float(rho), "p_value": float(pval)}

    # 3. Evaluate Top-K subsets on HELD-OUT test/ood pairs (Sufficiency + Necessity)
    def _top_k_by(col: str, ascending: bool = False) -> list[tuple[int, int]]:
        sub = df_heads.sort_values(col, ascending=ascending).head(top_k)
        return [(int(r.layer), int(r.head)) for r in sub.itertuples()]

    head_sets = {
        f"top{top_k}_causal_tau": _top_k_by("tau_dev"),
        f"top{top_k}_attn_contrib_Zarei": _top_k_by("attn_contribution"),
        f"top{top_k}_kl_spike_Maliha": _top_k_by("kl_spike"),
        f"top{top_k}_spatial_grad_Wang": _top_k_by("spatial_gradient_abs"),
        f"bottom{top_k}_control": _top_k_by("tau_dev", ascending=True),
    }

    mean_cache = ActivationCache.average(list(dev_c_caches.values()))
    held_out_eval: dict[str, dict] = {}
    for name, h_list in head_sets.items():
        suff = _eval_head_subset_patching(
            test_pairs,
            h_list,
            target_steps,
            runner,
            encoded,
            evaluator,
            test_c_caches,
            test_c_evals,
            test_r_evals,
        )
        nec = _eval_head_subset_ablation(
            test_pairs,
            h_list,
            target_steps,
            runner,
            encoded,
            evaluator,
            mean_cache,
            mode="mean_ablate",
        )
        held_out_eval[name] = {
            "heads": h_list,
            "sufficiency_patching": suff,
            "necessity_mean_ablate": nec,
        }
        print(
            f"[E2 Held-out] {name:<30} | suff_tau={suff['tau_mean']:.3f} suff_flip={suff['flip_rate']:.2f} "
            f"| ablate_rel_acc={nec['rel_acc']:.2f} (baseline=1.00)"
        )

    summary = {
        "model_id": model_id,
        "layers": target_layers,
        "steps": target_steps,
        "n_dev_valid": len(dev_pairs),
        "n_test_ood_valid": len(test_pairs),
        "observational_vs_causal_spearman": corr_stats,
        "held_out_comparison": held_out_eval,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 head-level discovery and baseline comparison")
    parser.add_argument("--layers", default="2,3,10,11")
    parser.add_argument("--steps", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-dev", type=int, default=12)
    parser.add_argument("--num-test", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--out-dir", default="artifacts/e2_head_discovery")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    steps = [int(x) for x in args.steps.split(",")]

    run_e2(
        layers=layers,
        steps=steps,
        num_dev_pairs=args.num_dev,
        num_test_pairs=args.num_test,
        top_k=args.top_k,
        gen_device=args.gen_device,
        eval_device=args.eval_device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
