"""E0: Baseline PixArt-Sigma spatial-relation and color-binding sanity check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from benchmark.counterfactual_pairs import (
    CounterfactualPair,
    build_color_binding_pairs,
    build_spatial_lr_pairs,
    save_pairs_jsonl,
)
from evaluation.spatial_eval import SpatialColorEvaluator
from models.pixart_sigma import DEFAULT_MODEL_ID, PixArtSigmaRunner
from visualization.heatmaps import save_counterfactual_grid


def run_e0(
    model_id: str = DEFAULT_MODEL_ID,
    num_spatial_dev: int = 24,
    num_color_dev: int = 18,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    gen_device: str = "cuda:0",
    eval_device: str = "cuda:1",
    out_dir: str = "artifacts/e0_baseline",
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build and save deterministic counterfactual benchmark datasets
    spatial_pairs = build_spatial_lr_pairs(num_dev=num_spatial_dev, num_test=64, num_ood=32)
    color_pairs = build_color_binding_pairs(num_dev=num_color_dev, num_test=64, num_ood=32)

    data_dir = Path("artifacts/data")
    save_pairs_jsonl(spatial_pairs, data_dir / "spatial_lr.jsonl")
    save_pairs_jsonl(color_pairs, data_dir / "color_binding.jsonl")

    dev_spatial = [p for p in spatial_pairs if p.split == "dev"]
    dev_color = [p for p in color_pairs if p.split == "dev"]
    dev_all: list[CounterfactualPair] = dev_spatial + dev_color

    # 2. Pre-encode & cache all dev prompts once using sharded T5-XXL
    all_prompts: list[str] = []
    for p in dev_all:
        all_prompts.append(p.clean_prompt)
        all_prompts.append(p.corrupt_prompt)

    runner = PixArtSigmaRunner(model_id=model_id, device=gen_device, dtype=torch.bfloat16)
    encoded = runner.encode_and_cache_prompts(all_prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    # Use cuda:0 for eval as fallback if only 1 GPU exists
    if torch.cuda.device_count() < 2:
        eval_device = gen_device
    evaluator = SpatialColorEvaluator(device=eval_device)
    evaluator.load()

    records: list[dict] = []
    sample_grid_rows: list[tuple[str, list[tuple[str, object]]]] = []

    for idx, pair in enumerate(dev_all):
        res_clean = runner.generate(
            prompt_entry=encoded[pair.clean_prompt],
            negative_entry=neg_entry,
            seed=pair.seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        res_corrupt = runner.generate(
            prompt_entry=encoded[pair.corrupt_prompt],
            negative_entry=neg_entry,
            seed=pair.seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        ev_clean = evaluator.evaluate_image(
            res_clean.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_clean,
            color_b=pair.color_b_clean,
            expected_relation=pair.relation_clean,
        )
        ev_corrupt = evaluator.evaluate_image(
            res_corrupt.image,
            obj_a=pair.obj_a,
            obj_b=pair.obj_b,
            color_a=pair.color_a_corrupt,
            color_b=pair.color_b_corrupt,
            expected_relation=pair.relation_corrupt,
        )

        if idx < 6 or (pair.task == "color_binding" and len(sample_grid_rows) < 10):
            res_clean.image.save(img_dir / f"{pair.pair_id}_clean.png")
            res_corrupt.image.save(img_dir / f"{pair.pair_id}_corrupt.png")
            sample_grid_rows.append(
                (
                    pair.pair_id,
                    [
                        (f"Clean ({ev_clean.relation_pred})", res_clean.image),
                        (f"Corrupt ({ev_corrupt.relation_pred})", res_corrupt.image),
                    ],
                )
            )

        if pair.task == "spatial_lr":
            valid_cf = (
                ev_clean.dx is not None
                and ev_corrupt.dx is not None
                and ev_clean.relation_pred == pair.relation_clean
                and ev_corrupt.relation_pred == pair.relation_corrupt
                and (ev_clean.dx - ev_corrupt.dx) > 0.04
            )
        else:
            valid_cf = bool(ev_clean.color_binding_correct) and bool(ev_corrupt.color_binding_correct)

        records.append(
            {
                "pair_id": pair.pair_id,
                "task": pair.task,
                "template_id": pair.template_id,
                "obj_a": pair.obj_a,
                "obj_b": pair.obj_b,
                "color_a_clean": pair.color_a_clean,
                "color_b_clean": pair.color_b_clean,
                "clean_both_det": ev_clean.both_detected,
                "corrupt_both_det": ev_corrupt.both_detected,
                "clean_dx": ev_clean.dx,
                "corrupt_dx": ev_corrupt.dx,
                "clean_rel_pred": ev_clean.relation_pred,
                "corrupt_rel_pred": ev_corrupt.relation_pred,
                "clean_rel_ok": ev_clean.relation_correct,
                "corrupt_rel_ok": ev_corrupt.relation_correct,
                "clean_color_ok": ev_clean.color_binding_correct,
                "corrupt_color_ok": ev_corrupt.color_binding_correct,
                "valid_counterfactual": valid_cf,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(out_path / "details.csv", index=False)

    if sample_grid_rows:
        save_counterfactual_grid(sample_grid_rows[:8], out_path / "sample_pairs_grid.png")

    summary: dict = {"model_id": model_id, "num_inference_steps": num_inference_steps}
    for task_name in ["spatial_lr", "color_binding"]:
        sub = df[df["task"] == task_name]
        if sub.empty:
            continue
        task_stats = {
            "n_pairs": int(len(sub)),
            "clean_both_det_rate": float(sub["clean_both_det"].mean()),
            "corrupt_both_det_rate": float(sub["corrupt_both_det"].mean()),
            "clean_color_acc": float(sub["clean_color_ok"].dropna().mean())
            if sub["clean_color_ok"].notna().any()
            else None,
            "corrupt_color_acc": float(sub["corrupt_color_ok"].dropna().mean())
            if sub["corrupt_color_ok"].notna().any()
            else None,
            "valid_counterfactual_count": int(sub["valid_counterfactual"].sum()),
            "valid_counterfactual_rate": float(sub["valid_counterfactual"].mean()),
            "valid_pair_ids": sub.loc[sub["valid_counterfactual"], "pair_id"].tolist(),
        }
        if task_name == "spatial_lr":
            task_stats["clean_rel_acc"] = float(sub["clean_rel_ok"].fillna(False).mean())
            task_stats["corrupt_rel_acc"] = float(sub["corrupt_rel_ok"].fillna(False).mean())
            by_tpl = {}
            for tpl, grp in sub.groupby("template_id"):
                by_tpl[str(tpl)] = {
                    "n": int(len(grp)),
                    "clean_rel_acc": float(grp["clean_rel_ok"].fillna(False).mean()),
                    "corrupt_rel_acc": float(grp["corrupt_rel_ok"].fillna(False).mean()),
                    "valid_rate": float(grp["valid_counterfactual"].mean()),
                }
            task_stats["by_template"] = by_tpl
        summary[task_name] = task_stats

    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E0 baseline sanity check on PixArt-Sigma")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--num-spatial-dev", type=int, default=24)
    parser.add_argument("--num-color-dev", type=int, default=18)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--gen-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:1")
    parser.add_argument("--out-dir", default="artifacts/e0_baseline")
    args = parser.parse_args()

    run_e0(
        model_id=args.model_id,
        num_spatial_dev=args.num_spatial_dev,
        num_color_dev=args.num_color_dev,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        gen_device=args.gen_device,
        eval_device=args.eval_device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
