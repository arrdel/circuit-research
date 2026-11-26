# E3 Color-Binding Pilot Run Log

## Execution Environment

- Host: `ai-lab2`
- GPUs visible: `['NVIDIA GeForce RTX 4090']` (gen: `cuda:0`, eval: `cuda:0`)
- Python: `3.10.12`
- Torch: `2.6.0+cu124`, Diffusers: `0.41.0`, Transformers: `5.20.2`

## Model and Generation Settings

- Checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`)
- Sampler: `DPMSolverMultistepScheduler`
- Steps: `20`
- Guidance scale: `4.5`
- Precision: `torch.bfloat16`
- Hook point: `attn2_out` (conditional CFG branch only)
- Seeds: `[42, 123, 456]`
- Total (pair, seed) instances: `9` (valid counterfactuals: `7`)

## Color-Binding Counterfactual Pairs

- `color_binding_dev_0001`: Clean = `A blue umbrella on the left and a red banana on the right, clean studio photo` | Corrupt = `A red umbrella on the left and a blue banana on the right, clean studio photo`
- `color_binding_dev_0027`: Clean = `A yellow bicycle on the left and a red chair on the right, clean studio photo` | Corrupt = `A red bicycle on the left and a yellow chair on the right, clean studio photo`
- `color_binding_dev_0030`: Clean = `A red bicycle on the left and a blue book on the right, clean studio photo` | Corrupt = `A blue bicycle on the left and a red book on the right, clean studio photo`

## Peak Cell Results and Gate

| Cell | Tau (Color) | Flip Rate | Sigma (Spatial Pres.) | Detection Rate |
| --- | --- | --- | --- | --- |
| L10-11 @ W1[4-7] | 0.4027 | 0.1429 | 0.9872 | 1.0000 |
| L2-3 @ W1[4-7] | 0.0768 | 0.0000 | 0.9943 | 1.0000 |

- Peak single head `(l, h, t)`: `L11H1 @ W1[4-7]` (`tau = 0.2457`, `flip_rate = 0.1429`, `sigma = 0.9917`)
- Random-head control (`Patch single random head L3H10 at W1[4-7]`): `tau = 0.0080`
- **E3 Pilot Gate (`peak_tau > random_head_tau`): `PASS`**

## Four Controls

| Control | Description | Tau (Color) | Flip Rate | Sigma | Detection Rate |
| --- | --- | --- | --- | --- | --- |
| random_head | Patch single random head L3H10 at W1[4-7] | 0.0080 | 0.0000 | 0.9968 | 1.0000 |
| wrong_timestep | Patch L10-11 at late window W4[16-19] | 0.0049 | 0.0000 | 0.9986 | 1.0000 |
| same_prompt_noop | Patch corrupt cache back into corrupt at L10-11@W1[4-7] | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| seed_variation | Patch clean cache from shifted seed (s+999) into corrupt at L10-11@W1[4-7] | 0.3114 | 0.1429 | 0.9375 | 1.0000 |
