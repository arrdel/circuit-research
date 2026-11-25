# E1 Pilot Run Log

## Execution Environment

- Host: `ai-lab2`
- GPUs visible: `['NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090']` (gen: `cuda:0`, eval: `cuda:1`)
- Python: `3.10.12`
- Torch: `2.6.0+cu124`, Diffusers: `0.41.0`, Transformers: `5.20.2`

## Model and Generation Settings

- Checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`)
- Sampler: `DPMSolverMultistepScheduler`
- Steps: `20`
- Guidance scale: `4.5`
- Precision: `torch.bfloat16`
- Hook point (L x T sweep): `attn2_out` (conditional CFG branch only)
- Seeds: `[42, 123, 456]`
- Total (pair, seed) instances: `9` (valid counterfactuals: `6`)

## Counterfactual Pairs Tested (Explicit Template)

- `spatial_lr_dev_0002`: Clean = `A green dog on the left and a purple cup on the right, clean studio photo` | Corrupt = `A green dog on the right and a purple cup on the left, clean studio photo`
- `spatial_lr_dev_0004`: Clean = `A purple clock on the left and a yellow book on the right, clean studio photo` | Corrupt = `A purple clock on the right and a yellow book on the left, clean studio photo`
- `spatial_lr_dev_0010`: Clean = `A green book on the left and a red banana on the right, clean studio photo` | Corrupt = `A green book on the right and a red banana on the left, clean studio photo`

## Peak Cell and Gate Result

- Peak (L, T) cell: `L10-11 @ W1[4-7]` | `tau = 0.7104`, `sigma = 0.9263`, `flip_rate = 0.6667`, `detection_rate = 1.0000`
- Peak single head `(l, h, t)`: `L10H15 @ W1[4-7]` | `tau = 0.2521`, `sigma = 0.9291`, `flip_rate = 0.1667`
- Random-head control (`Patch single random head L3H10 at W1[4-7]`): `tau = 0.0141`
- **Pilot Gate (`peak_tau > random_head_tau`): `PASS`**

## Four Controls Summary

| Control | Description | Tau | Flip Rate | Sigma | Detection Rate |
| --- | --- | --- | --- | --- | --- |
| random_head | Patch single random head L3H10 at W1[4-7] | 0.0141 | 0.0000 | 0.9931 | 1.0000 |
| wrong_timestep | Patch L10-11 at late window W4[16-19] | 0.0003 | 0.0000 | 0.9983 | 1.0000 |
| same_prompt_noop | Patch corrupt cache back into corrupt at L10-11@W1[4-7] | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| seed_variation | Patch clean cache from shifted seed (s+999) into corrupt at L10-11@W1[4-7] | 0.4980 | 0.5000 | 0.8148 | 1.0000 |
