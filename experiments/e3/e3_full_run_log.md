# E3 Color-Binding Full Sweep Run Log

## Execution Environment

- Host: `ai-lab2`
- GPUs visible: `['NVIDIA GeForce RTX 4090']` (gen: `cuda:0`, eval: `cuda:0`)
- Python: `3.10.12`
- Torch: `2.6.0+cu124`, Diffusers: `0.41.0`, Transformers: `5.20.2`

## Model and Generation Settings

- Checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`)
- Split: `test` (fresh object-disjoint color-binding pairs, template `fixed_lr_swap_colors`)
- Sampler: `DPMSolverMultistepScheduler`
- Steps: `20`
- Guidance scale: `4.5`
- Precision: `torch.bfloat16`
- Hook point: `attn2_out` (conditional CFG branch only)
- Seeds: `[42, 123, 456]`
- Total (pair, seed) instances: `15` (valid counterfactuals: `15`)

## Fresh Test Pairs

- `color_binding_test_0001`: Clean = `A yellow clock on the left and a red bottle on the right, clean studio photo` | Corrupt = `A red clock on the left and a yellow bottle on the right, clean studio photo`
- `color_binding_test_0013`: Clean = `A yellow dog on the left and a red clock on the right, clean studio photo` | Corrupt = `A red dog on the left and a yellow clock on the right, clean studio photo`
- `color_binding_test_0016`: Clean = `A red cup on the left and a blue dog on the right, clean studio photo` | Corrupt = `A blue cup on the left and a red dog on the right, clean studio photo`
- `color_binding_test_0019`: Clean = `A yellow cup on the left and a red apple on the right, clean studio photo` | Corrupt = `A red cup on the left and a yellow apple on the right, clean studio photo`
- `color_binding_test_0034`: Clean = `A red car on the left and a blue bottle on the right, clean studio photo` | Corrupt = `A blue car on the left and a red bottle on the right, clean studio photo`

## Peak Cell Results

| Cell | Tau (Color) | 95% CI | Flip Rate | Sigma | Spatial Pres. | Detection Rate |
| --- | --- | --- | --- | --- | --- | --- |
| L10-11 @ W1[4-7] | 0.3535 | [0.168, 0.568] | 0.3333 | 0.9816 | 1.0000 | 1.0000 |
| L2-3 @ W1[4-7] | 0.0275 | [0.009, 0.050] | 0.0667 | 0.9931 | 1.0000 | 1.0000 |

- Peak single head `(l, h, t)`: `L11H1 @ W1[4-7]` (`tau = 0.3820`, `flip_rate = 0.3333`, `sigma = 0.9844`)
- Confirms E3 Pilot (`L10-11` > `L2-3` and beats `random_head`): `True`

## Controls

| Control | Description | Tau (Color) | Flip Rate | Sigma | Spatial Pres. |
| --- | --- | --- | --- | --- | --- |
| random_head | Patch single random head L3H10 at W1[4-7] | 0.0114 | 0.0667 | 0.9957 | 1.0000 |
| wrong_timestep | Patch L10-11 at late window W4[16-19] | 0.0054 | 0.0000 | 0.9988 | 1.0000 |
| same_prompt_noop | Patch corrupt cache back into corrupt at L10-11@W1[4-7] | 0.0000 | 0.0667 | 1.0000 | 1.0000 |
| seed_variation | Patch clean cache from shifted seed (s+999) into corrupt at L10-11@W1[4-7] | 0.3658 | 0.3333 | 0.9794 | 1.0000 |
