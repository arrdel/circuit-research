# E1 1024-MS Confirmation and Sublayer Comparison Run Log

## Execution Environment

- Host: `ai-lab2`
- GPUs visible: `['NVIDIA GeForce RTX 4090']` (gen: `cuda:0`, eval: `cuda:0`)
- Python: `3.10.12`
- Torch: `2.6.0+cu124`, Diffusers: `0.41.0`, Transformers: `5.20.2`

## Part A: 1024-MS Confirmation (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`)

- Pairs: `['spatial_lr_test_0000', 'spatial_lr_test_0001', 'spatial_lr_test_0003']`, Seeds: `[42, 123, 456]`, Random-head control: `L2H10@W1[4-7]`

| Condition | Tau | Flip Rate | Sigma | Detection Rate | Color Preserve |
| --- | --- | --- | --- | --- | --- |
| L10-11_W1[4-7] | 0.4901 | 0.4286 | 0.8402 | 1.0000 | 0.5714 |
| L2-3_W1[4-7] | 0.4991 | 0.4286 | 0.8412 | 1.0000 | 0.5000 |
| ctrl_random_head | 0.0001 | 0.0000 | 0.9984 | 1.0000 | 1.0000 |
| ctrl_wrong_timestep | 0.0002 | 0.0000 | 0.9989 | 1.0000 | 1.0000 |
| ctrl_same_prompt_noop | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| ctrl_seed_variation | 0.3356 | 0.2857 | 0.7905 | 1.0000 | 0.4286 |

## Part B: Sublayer Comparison (`PixArt-alpha/PixArt-Sigma-XL-2-512-MS`)

- Pairs: `['spatial_lr_test_0001', 'spatial_lr_test_0011', 'spatial_lr_test_0013', 'spatial_lr_test_0016']`, Seeds: `[42, 123, 456]`

| Cell | Hook Point | Tau | Flip Rate | Sigma | Detection Rate |
| --- | --- | --- | --- | --- | --- |
| L10-11_W1[4-7] | attn2_out | 0.7912 | 0.7778 | 0.9751 | 1.0000 |
| L10-11_W1[4-7] | attn1_out | 0.7191 | 0.6667 | 0.9062 | 1.0000 |
| L10-11_W1[4-7] | ff_out | 0.1402 | 0.1111 | 0.9486 | 1.0000 |
| L10-11_W1[4-7] | attn2_in_txt | 0.1503 | 0.1111 | 0.8825 | 1.0000 |
| L2-3_W1[4-7] | attn2_out | 0.6795 | 0.6667 | 0.8750 | 1.0000 |
| L2-3_W1[4-7] | attn1_out | 0.0723 | 0.1111 | 0.9116 | 1.0000 |
| L2-3_W1[4-7] | ff_out | 0.0753 | 0.1111 | 0.8109 | 0.8889 |
| L2-3_W1[4-7] | attn2_in_txt | 0.6300 | 0.5556 | 0.9121 | 1.0000 |
| L6-7_W2[8-11] | attn2_out | 0.2166 | 0.2222 | 0.8830 | 1.0000 |
| L6-7_W2[8-11] | attn1_out | 0.9424 | 0.8889 | 0.8979 | 1.0000 |
| L6-7_W2[8-11] | ff_out | 0.0174 | 0.0000 | 0.9552 | 1.0000 |
| L6-7_W2[8-11] | attn2_in_txt | 0.0008 | 0.0000 | 0.9617 | 1.0000 |
