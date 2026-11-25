"""Day 1 Task 2: Reproduce PixArt-Sigma 1024-MS generation across 5 counterfactual prompt families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.spatial_eval import SpatialColorEvaluator
from models.pixart_sigma import PixArtSigmaRunner
from visualization.heatmaps import save_counterfactual_grid


MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
SAMPLER_NAME = "DPMSolverMultistepScheduler"
NUM_STEPS = 20
GUIDANCE_SCALE = 4.5
HEIGHT = 1024
WIDTH = 1024
SEEDS = [42, 123, 456]

PROMPT_FAMILIES: list[tuple[str, str]] = [
    ("p1_red_cube_left_blue_sphere", "Red cube left of blue sphere."),
    ("p2_blue_cube_left_red_sphere", "Blue cube left of red sphere."),
    ("p3_red_cube_right_blue_sphere", "Red cube right of blue sphere."),
    ("p4_red_cube_left_blue_sphere_green_cyl", "Red cube left of blue sphere, green cylinder in front."),
    ("p5_two_red_cubes_one_blue_sphere", "Two red cubes and one blue sphere."),
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_reproduction(
    gen_device: str = "cuda:2",
    eval_device: str = "cuda:3",
    out_dir: str = "experiments/e0",
) -> dict:
    out_path = Path(out_dir)
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.device_count() <= 2:
        gen_device = "cuda:0"
        eval_device = "cuda:0" if torch.cuda.device_count() < 2 else "cuda:1"

    prompts = [p for _, p in PROMPT_FAMILIES]
    runner = PixArtSigmaRunner(
        model_id=MODEL_ID,
        device=gen_device,
        dtype=torch.bfloat16,
    )
    encoded = runner.encode_and_cache_prompts(prompts)
    neg_entry = encoded[""]
    runner.load_denoiser()

    evaluator = SpatialColorEvaluator(device=eval_device, box_threshold=0.18, text_threshold=0.18)
    evaluator.load()

    records: list[dict] = []
    grid_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []

    for fam_id, prompt in PROMPT_FAMILIES:
        row_imgs: list[tuple[str, Image.Image]] = []
        for seed in SEEDS:
            res = runner.generate(
                prompt_entry=encoded[prompt],
                negative_entry=neg_entry,
                seed=seed,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT,
                width=WIDTH,
            )
            rel_img_path = Path("experiments/e0/images") / f"{fam_id}_seed{seed}.png"
            res.image.save(rel_img_path)
            sha = _file_sha256(rel_img_path)

            # Quantitative detector + SigLIP diagnostic
            if fam_id in ("p1_red_cube_left_blue_sphere", "p3_red_cube_right_blue_sphere"):
                ev = evaluator.evaluate_image(
                    res.image,
                    obj_a="cube",
                    obj_b="sphere",
                    color_a="red",
                    color_b="blue",
                    expected_relation="left" if fam_id == "p1_red_cube_left_blue_sphere" else "right",
                    compute_embedding=False,
                )
            elif fam_id == "p2_blue_cube_left_red_sphere":
                ev = evaluator.evaluate_image(
                    res.image,
                    obj_a="cube",
                    obj_b="sphere",
                    color_a="blue",
                    color_b="red",
                    expected_relation="left",
                    compute_embedding=False,
                )
            else:
                ev = evaluator.evaluate_image(
                    res.image,
                    obj_a="cube",
                    obj_b="sphere",
                    color_a="red",
                    color_b="blue",
                    expected_relation="left" if "left" in prompt.lower() else "none",
                    compute_embedding=False,
                )

            records.append(
                {
                    "family_id": fam_id,
                    "prompt": prompt,
                    "seed": seed,
                    "sampler": SAMPLER_NAME,
                    "steps": NUM_STEPS,
                    "guidance_scale": GUIDANCE_SCALE,
                    "resolution": f"{WIDTH}x{HEIGHT}",
                    "output_path": str(rel_img_path),
                    "sha256": sha,
                    "both_cube_sphere_detected": ev.both_detected,
                    "dx_sphere_minus_cube": round(ev.dx, 4) if ev.dx is not None else None,
                    "relation_pred": ev.relation_pred,
                    "relation_correct": ev.relation_correct,
                    "color_a_margin": round(ev.color_a_margin, 4) if ev.color_a_margin is not None else None,
                    "color_b_margin": round(ev.color_b_margin, 4) if ev.color_b_margin is not None else None,
                    "color_binding_correct": ev.color_binding_correct,
                }
            )
            row_imgs.append((f"seed={seed}", res.image))
        grid_rows.append((fam_id, row_imgs))

    # Determinism / reproduction check: rerun p1 with seed 42
    recheck_prompt = PROMPT_FAMILIES[0][1]
    recheck_seed = SEEDS[0]
    res_recheck = runner.generate(
        prompt_entry=encoded[recheck_prompt],
        negative_entry=neg_entry,
        seed=recheck_seed,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
    )
    recheck_path = Path("experiments/e0/images") / f"{PROMPT_FAMILIES[0][0]}_seed{recheck_seed}_recheck.png"
    res_recheck.image.save(recheck_path)
    orig_path = Path("experiments/e0/images") / f"{PROMPT_FAMILIES[0][0]}_seed{recheck_seed}.png"

    arr_orig = np.asarray(Image.open(orig_path))
    arr_recheck = np.asarray(Image.open(recheck_path))
    max_abs_diff = int(np.max(np.abs(arr_orig.astype(np.int32) - arr_recheck.astype(np.int32))))
    sha_orig = _file_sha256(orig_path)
    sha_recheck = _file_sha256(recheck_path)
    repro_passed = (max_abs_diff == 0) and (sha_orig == sha_recheck)

    save_counterfactual_grid(grid_rows, out_path / "e0_grid.png", tile_size=256)

    summary = {
        "model_id": MODEL_ID,
        "sampler": SAMPLER_NAME,
        "steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "dtype": "bfloat16",
        "reproduction_check": {
            "prompt": recheck_prompt,
            "seed": recheck_seed,
            "orig_path": str(orig_path),
            "recheck_path": str(recheck_path),
            "orig_sha256": sha_orig,
            "recheck_sha256": sha_recheck,
            "max_abs_pixel_diff": max_abs_diff,
            "passed": repro_passed,
        },
        "runs": records,
    }
    with (out_path / "e0_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write run-log.md (no em dashes)
    lines = [
        "# E0 PixArt-Sigma Reproduction Run Log",
        "",
        "## Fixed Generation Settings",
        "",
        f"- Model checkpoint: `{MODEL_ID}`",
        f"- Sampler: `{SAMPLER_NAME}`",
        f"- Inference steps: `{NUM_STEPS}`",
        f"- Guidance scale: `{GUIDANCE_SCALE}`",
        f"- Resolution: `{WIDTH}x{HEIGHT}`",
        "- Precision: `torch.bfloat16`",
        f"- Seeds: `{SEEDS}`",
        "",
        "## Reproduction Check",
        "",
        f"- Prompt: `{recheck_prompt}`",
        f"- Seed: `{recheck_seed}`",
        f"- Original output: `{orig_path}` (sha256: `{sha_orig[:16]}...`)",
        f"- Recheck output: `{recheck_path}` (sha256: `{sha_recheck[:16]}...`)",
        f"- Max absolute pixel difference: `{max_abs_diff}`",
        f"- Status: `{'PASS' if repro_passed else 'FAIL'}`",
        "",
        "## Generated Runs",
        "",
        "| Family | Prompt | Seed | Sampler | Steps | Guidance | Output Path |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        lines.append(
            f"| {r['family_id']} | {r['prompt']} | {r['seed']} | {r['sampler']} | "
            f"{r['steps']} | {r['guidance_scale']} | `{r['output_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Automated Spatial and Color Diagnostics (GroundingDINO-tiny + SigLIP-base)",
            "",
            "| Family | Seed | Both Detected | dx (sphere_x - cube_x) | Pred Rel | Rel OK | Color Binding OK |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in records:
        lines.append(
            f"| {r['family_id']} | {r['seed']} | {r['both_cube_sphere_detected']} | "
            f"{r['dx_sphere_minus_cube']} | {r['relation_pred']} | {r['relation_correct']} | "
            f"{r['color_binding_correct']} |"
        )

    (out_path / "run-log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[E0-Day1] Reproduction passed: {repro_passed}, max_abs_diff={max_abs_diff}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 1 Task 2: PixArt-Sigma 1024-MS reproduction")
    parser.add_argument("--gen-device", default="cuda:2")
    parser.add_argument("--eval-device", default="cuda:3")
    parser.add_argument("--out-dir", default="experiments/e0")
    args = parser.parse_args()
    run_reproduction(gen_device=args.gen_device, eval_device=args.eval_device, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
