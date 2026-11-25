# E0 PixArt-Sigma Reproduction Run Log

## Fixed Generation Settings

- Model checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`
- Sampler: `DPMSolverMultistepScheduler`
- Inference steps: `20`
- Guidance scale: `4.5`
- Resolution: `1024x1024`
- Precision: `torch.bfloat16`
- Seeds: `[42, 123, 456]`

## Reproduction Check

- Prompt: `Red cube left of blue sphere.`
- Seed: `42`
- Original output: `experiments/e0/images/p1_red_cube_left_blue_sphere_seed42.png` (sha256: `0f825cd7463f0f00...`)
- Recheck output: `experiments/e0/images/p1_red_cube_left_blue_sphere_seed42_recheck.png` (sha256: `0f825cd7463f0f00...`)
- Max absolute pixel difference: `0`
- Status: `PASS`

## Generated Runs

| Family | Prompt | Seed | Sampler | Steps | Guidance | Output Path |
| --- | --- | --- | --- | --- | --- | --- |
| p1_red_cube_left_blue_sphere | Red cube left of blue sphere. | 42 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p1_red_cube_left_blue_sphere_seed42.png` |
| p1_red_cube_left_blue_sphere | Red cube left of blue sphere. | 123 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p1_red_cube_left_blue_sphere_seed123.png` |
| p1_red_cube_left_blue_sphere | Red cube left of blue sphere. | 456 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p1_red_cube_left_blue_sphere_seed456.png` |
| p2_blue_cube_left_red_sphere | Blue cube left of red sphere. | 42 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p2_blue_cube_left_red_sphere_seed42.png` |
| p2_blue_cube_left_red_sphere | Blue cube left of red sphere. | 123 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p2_blue_cube_left_red_sphere_seed123.png` |
| p2_blue_cube_left_red_sphere | Blue cube left of red sphere. | 456 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p2_blue_cube_left_red_sphere_seed456.png` |
| p3_red_cube_right_blue_sphere | Red cube right of blue sphere. | 42 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p3_red_cube_right_blue_sphere_seed42.png` |
| p3_red_cube_right_blue_sphere | Red cube right of blue sphere. | 123 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p3_red_cube_right_blue_sphere_seed123.png` |
| p3_red_cube_right_blue_sphere | Red cube right of blue sphere. | 456 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p3_red_cube_right_blue_sphere_seed456.png` |
| p4_red_cube_left_blue_sphere_green_cyl | Red cube left of blue sphere, green cylinder in front. | 42 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p4_red_cube_left_blue_sphere_green_cyl_seed42.png` |
| p4_red_cube_left_blue_sphere_green_cyl | Red cube left of blue sphere, green cylinder in front. | 123 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p4_red_cube_left_blue_sphere_green_cyl_seed123.png` |
| p4_red_cube_left_blue_sphere_green_cyl | Red cube left of blue sphere, green cylinder in front. | 456 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p4_red_cube_left_blue_sphere_green_cyl_seed456.png` |
| p5_two_red_cubes_one_blue_sphere | Two red cubes and one blue sphere. | 42 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p5_two_red_cubes_one_blue_sphere_seed42.png` |
| p5_two_red_cubes_one_blue_sphere | Two red cubes and one blue sphere. | 123 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p5_two_red_cubes_one_blue_sphere_seed123.png` |
| p5_two_red_cubes_one_blue_sphere | Two red cubes and one blue sphere. | 456 | DPMSolverMultistepScheduler | 20 | 4.5 | `experiments/e0/images/p5_two_red_cubes_one_blue_sphere_seed456.png` |

## Automated Spatial and Color Diagnostics (GroundingDINO-tiny + SigLIP-base)

| Family | Seed | Both Detected | dx (sphere_x - cube_x) | Pred Rel | Rel OK | Color Binding OK |
| --- | --- | --- | --- | --- | --- | --- |
| p1_red_cube_left_blue_sphere | 42 | True | -0.4102 | right | False | True |
| p1_red_cube_left_blue_sphere | 123 | True | -0.4358 | right | False | True |
| p1_red_cube_left_blue_sphere | 456 | True | 0.3172 | left | True | True |
| p2_blue_cube_left_red_sphere | 42 | True | 0.3413 | left | True | True |
| p2_blue_cube_left_red_sphere | 123 | True | 0.4419 | left | True | True |
| p2_blue_cube_left_red_sphere | 456 | True | -0.1165 | right | False | False |
| p3_red_cube_right_blue_sphere | 42 | True | -0.3774 | right | True | True |
| p3_red_cube_right_blue_sphere | 123 | True | -0.3687 | right | True | True |
| p3_red_cube_right_blue_sphere | 456 | True | 0.2795 | left | False | True |
| p4_red_cube_left_blue_sphere_green_cyl | 42 | True | -0.2248 | right | False | True |
| p4_red_cube_left_blue_sphere_green_cyl | 123 | True | 0.4106 | left | True | False |
| p4_red_cube_left_blue_sphere_green_cyl | 456 | True | -0.365 | right | False | False |
| p5_two_red_cubes_one_blue_sphere | 42 | True | -0.296 | right | None | True |
| p5_two_red_cubes_one_blue_sphere | 123 | True | -0.3103 | right | None | True |
| p5_two_red_cubes_one_blue_sphere | 456 | True | 0.4094 | left | None | True |
