"""Day 7 Task 3(b): Same-instance paired comparison of Prompt-to-Prompt vs activation patching."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from benchmark.counterfactual_pairs import build_spatial_lr_pairs
from evaluation.causal_metrics import (
    compute_pair_causal_outcome,
    summarize_causal_outcomes,
)
from evaluation.spatial_eval import SpatialColorEvaluator
from instrumentation.hooks import ActivationCache, DiTHookManager
from interventions.patching import (
    AttnMapInjectionController,
    InterventionController,
    InterventionSpec,
)
from models.pixart_sigma import DEFAULT_MODEL_ID, PixArtSigmaRunner


def run_paired_comparison(
    results_json: str = "experiments/e1/e1_full_results.json",
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:0",
    out_json: str = "experiments/e1/p2p_vs_patching_paired.json",
) -> dict:
    t0 = time.time()
    src = json.loads(Path(results_json).read_text(encoding="utf-8"))
    pair_ids = [p["pair_id"] for p in src["pairs"]]
    seeds = src["seeds"]
    peak_steps = src["peak_cell"]["steps"]

    all_explicit = build_spatial_lr_pairs(
        num_dev=24, num_test=36, num_ood=12, templates=("on_the_side_explicit",), base_seed=3000
    )
    pair_map = {p.pair_id: p for p in all_explicit}
    instances = [(pair_map[pid], sd) for pid in pair_ids for sd in seeds]

    prompts = []
    for p, _ in instances[:: len(seeds)]:
        prompts.extend([p.clean_prompt, p.corrupt_prompt])

    runner = PixArtSigmaRunner(model_id=DEFAULT_MODEL_ID, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    # Baseline clean/corrupt evaluations using unmodified default SDPA forward pass
    clean_evals = {}
    corrupt_evals = {}
    clean_caches = {}

    for pair, sd in instances:
        inst_id = f"{pair.pair_id}_s{sd}"
        # Capture attn2_out AND attn2_probs in ONE clean pass on [2, 3, 10, 11]
        # First get baseline unmodified clean/corrupt evals so validity mask is 100% shared!
        res_c_base = runner.generate(prompt_entry=encoded[pair.clean_prompt], negative_entry=neg_entry, seed=sd)
        res_r_base = runner.generate(prompt_entry=encoded[pair.corrupt_prompt], negative_entry=neg_entry, seed=sd)
        clean_evals[inst_id] = evaluator.evaluate_image(
            res_c_base.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )
        corrupt_evals[inst_id] = evaluator.evaluate_image(
            res_r_base.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_corrupt,
            color_b=pair.color_b_corrupt,
            expected_relation=pair.relation_corrupt,
        )

        mgr = DiTHookManager(
            runner.transformer,
            layers=[2, 3, 10, 11],
            hook_points=["attn2_out"],
            steps=peak_steps,
            branch="cond",
            store_cpu=True,
            capture_attn2_probs=True,
        )
        with mgr.active() as c_cache:
            _ = runner.generate(
                prompt_entry=encoded[pair.clean_prompt],
                negative_entry=neg_entry,
                seed=sd,
                step_callbacks=[mgr],
                decode_image=False,
            )
            mgr.cache = ActivationCache()
        clean_caches[inst_id] = c_cache

    shared_valid_ids = [
        f"{p.pair_id}_s{sd}"
        for p, sd in instances
        if clean_evals[f"{p.pair_id}_s{sd}"].dx is not None
        and corrupt_evals[f"{p.pair_id}_s{sd}"].dx is not None
        and clean_evals[f"{p.pair_id}_s{sd}"].relation_pred == p.relation_clean
        and corrupt_evals[f"{p.pair_id}_s{sd}"].relation_pred == p.relation_corrupt
        and (clean_evals[f"{p.pair_id}_s{sd}"].dx - corrupt_evals[f"{p.pair_id}_s{sd}"].dx) > 0.04
    ]
    print(f"[Paired P2P vs Patching] Shared valid counterfactuals: {len(shared_valid_ids)}/{len(instances)}")

    comparisons = {}
    for grp_lbl, layers in [("L10-11", [10, 11]), ("L2-3", [2, 3])]:
        patch_outcomes = []
        p2p_outcomes = []
        for pair, sd in instances:
            inst_id = f"{pair.pair_id}_s{sd}"

            spec = InterventionSpec(
                kind="patch",
                hook_point="attn2_out",
                layers=layers,
                steps=peak_steps,
                source_cache=clean_caches[inst_id],
                branch="cond",
            )
            ctrl = InterventionController(runner.transformer, [spec])
            with ctrl.active():
                res_patch = runner.generate(
                    prompt_entry=encoded[pair.corrupt_prompt],
                    negative_entry=neg_entry,
                    seed=sd,
                    step_callbacks=[ctrl],
                )
            ev_patch = evaluator.evaluate_image(
                res_patch.image,
                obj_a=pair.obj_a,
                obj_b=pair.obj_b,
                color_a=pair.color_a_corrupt,
                color_b=pair.color_b_corrupt,
                expected_relation=pair.relation_clean,
            )
            patch_outcomes.append(
                compute_pair_causal_outcome(
                    pair_id=inst_id,
                    task="spatial_lr",
                    clean=clean_evals[inst_id],
                    corrupt=corrupt_evals[inst_id],
                    patched=ev_patch,
                    clean_relation=pair.relation_clean,
                    corrupt_relation=pair.relation_corrupt,
                )
            )

            p2p = AttnMapInjectionController(
                runner.transformer,
                layers=layers,
                steps=peak_steps,
                source_cache=clean_caches[inst_id],
                branch="cond",
            )
            with p2p.active():
                res_p2p = runner.generate(
                    prompt_entry=encoded[pair.corrupt_prompt],
                    negative_entry=neg_entry,
                    seed=sd,
                    step_callbacks=[p2p],
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

        s_patch = summarize_causal_outcomes(patch_outcomes, valid_only=True).to_dict()
        s_p2p = summarize_causal_outcomes(p2p_outcomes, valid_only=True).to_dict()
        comparisons[grp_lbl] = {
            "activation_patching_attn2_out": s_patch,
            "prompt_to_prompt_attn_map": s_p2p,
        }
        print(
            f"[Paired {grp_lbl} N={s_patch['n_valid']}] Patch tau={s_patch['tau_mean']:.4f} "
            f"(flip={s_patch['flip_rate']:.4f}, sigma={s_patch['sigma_mean']:.4f}) | "
            f"P2P tau={s_p2p['tau_mean']:.4f} (flip={s_p2p['flip_rate']:.4f}, sigma={s_p2p['sigma_mean']:.4f})"
        )

    payload = {
        "checkpoint": DEFAULT_MODEL_ID,
        "checkpoint_label": "512-MS",
        "n_instances": len(instances),
        "n_shared_valid": len(shared_valid_ids),
        "shared_valid_ids": shared_valid_ids,
        "why_previous_n_differed": (
            "In baseline_comparison_full.json, baseline clean/corrupt images were generated with "
            "CrossAttentionCaptureProcessor (float32 matmul attention) rather than default bfloat16 SDPA, "
            "shifting 1-2 borderline test detections/dx values relative to e1_full_results.json (N=12 in L2-3, N=15 in L10-11 vs N=13 in e1_full_results). "
            "Here both methods share identical unmodified SDPA baseline masks (N=13)."
        ),
        "comparisons": comparisons,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path(out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired P2P vs activation patching check")
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    args = parser.parse_args()
    run_paired_comparison(gen_device=args.gen_device, eval_device=args.eval_device)


if __name__ == "__main__":
    main()
