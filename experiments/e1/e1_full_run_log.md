# E1 Full Sweep Run Log

## Execution Environment

- Host: `ai-lab2`
- GPUs visible: `['NVIDIA GeForce RTX 4090']` (gen: `cuda:0`, eval: `cuda:0`)
- Python: `3.10.12`
- Torch: `2.6.0+cu124`, Diffusers: `0.41.0`, Transformers: `5.20.2`

## Model and Generation Settings

- Checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`)
- Split: `test` (explicit template `on_the_side_explicit`, disjoint objects from `dev`)
- Sampler: `DPMSolverMultistepScheduler`
- Steps: `20`
- Guidance scale: `4.5`
- Precision: `torch.bfloat16`
- Hook point: `attn2_out` (conditional CFG branch only)
- Seeds: `[42, 123, 456]`
- Total (pair, seed) instances: `18` (valid counterfactuals: `13`)

## Test Pairs

- `spatial_lr_test_0001`: Clean = `A yellow dog on the left and a green book on the right, clean studio photo` | Corrupt = `A yellow dog on the right and a green book on the left, clean studio photo`
- `spatial_lr_test_0011`: Clean = `A purple umbrella on the left and a green apple on the right, clean studio photo` | Corrupt = `A purple umbrella on the right and a green apple on the left, clean studio photo`
- `spatial_lr_test_0013`: Clean = `A green cup on the left and a blue banana on the right, clean studio photo` | Corrupt = `A green cup on the right and a blue banana on the left, clean studio photo`
- `spatial_lr_test_0016`: Clean = `A red car on the left and a yellow apple on the right, clean studio photo` | Corrupt = `A red car on the right and a yellow apple on the left, clean studio photo`
- `spatial_lr_test_0020`: Clean = `A red cat on the left and a blue apple on the right, clean studio photo` | Corrupt = `A red cat on the right and a blue apple on the left, clean studio photo`
- `spatial_lr_test_0026`: Clean = `A green banana on the left and a yellow bottle on the right, clean studio photo` | Corrupt = `A green banana on the right and a yellow bottle on the left, clean studio photo`

## Summary Findings

- Peak cell: `L10-11 @ W1[4-7]` | `tau = 0.6903`, `sigma = 0.9349`, `flip_rate = 0.6923`, `detection_rate = 0.9231`
- L2-3 @ W1[4-7] replication: `tau = 0.5522` (`flip_rate = 0.5385`, `sigma = 0.9099`)

## Controls

| Control | Description | Tau | Flip Rate | Sigma | Color Preserve |
| --- | --- | --- | --- | --- | --- |
| random_head | Patch single random head L3H10 at W1[4-7] | 0.0042 | 0.0000 | 0.9722 | 0.8333 |
| wrong_timestep | Patch L10-11 at late window W4[16-19] | 0.0005 | 0.0000 | 0.9986 | 1.0000 |
| same_prompt_noop | Patch corrupt cache back into corrupt at L10-11@W1[4-7] | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| seed_variation | Patch clean cache from shifted seed (s+999) into corrupt at L10-11@W1[4-7] | 0.7063 | 0.6923 | 0.9100 | 0.8000 |
