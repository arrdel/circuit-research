# E3 Full L0-27 x W0-4 Color-Binding Scan Run Log

- Host: `ai-lab2`
- Checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`)
- Pairs: `['color_binding_test_0001', 'color_binding_test_0013', 'color_binding_test_0016', 'color_binding_test_0019']`, Seeds: `[42, 123, 456]`, Valid counterfactuals: `12`
- Reference single head `L11H01 @ W1[4-7]` (from `e3_full_results.json`): `tau = 0.3820`

## Top 10 `attn2_out` Cells Across L0-27 x W0-4

| Layer Group | Window | Tau (Color) | Flip Rate | Sigma | Detection Rate |
| --- | --- | --- | --- | --- | --- |
| L10-11 | W1[4-7] | 0.4090 | 0.4167 | 0.9854 | 1.0000 |
| L10-11 | W0[0-3] | 0.2053 | 0.0833 | 0.9555 | 1.0000 |
| L12-13 | W0[0-3] | 0.1016 | 0.0833 | 0.9923 | 1.0000 |
| L4-5 | W0[0-3] | 0.0992 | 0.0000 | 0.9902 | 1.0000 |
| L12-13 | W1[4-7] | 0.0986 | 0.0000 | 0.9907 | 1.0000 |
| L10-11 | W2[8-11] | 0.0953 | 0.0833 | 0.9948 | 1.0000 |
| L4-5 | W1[4-7] | 0.0698 | 0.0833 | 0.9933 | 1.0000 |
| L22-23 | W1[4-7] | 0.0634 | 0.0000 | 0.9958 | 1.0000 |
| L26-27 | W0[0-3] | 0.0589 | 0.0833 | 0.9972 | 1.0000 |
| L22-23 | W0[0-3] | 0.0519 | 0.0000 | 0.9958 | 1.0000 |

## Sublayer Checks (`attn1_out` and `ff_out`)

| Hook Point | Layer Group | Window | Tau (Color) | Flip Rate | Sigma |
| --- | --- | --- | --- | --- | --- |
| attn1_out | L10-11 | W1[4-7] | 0.1162 | 0.0833 | 0.9857 |
| ff_out | L10-11 | W1[4-7] | 0.1521 | 0.0833 | 0.9039 |
| attn1_out | L10-11 | W2[8-11] | 0.0150 | 0.0000 | 0.9861 |
| ff_out | L10-11 | W2[8-11] | 0.0149 | 0.0000 | 0.9876 |
| attn1_out | L6-7 | W2[8-11] | 0.3621 | 0.1667 | 0.9503 |
| ff_out | L6-7 | W2[8-11] | 0.0246 | 0.0833 | 0.9922 |
| attn1_out | L2-3 | W1[4-7] | 0.1264 | 0.1667 | 0.8931 |
| ff_out | L2-3 | W1[4-7] | 0.0467 | 0.0000 | 0.6522 |

- Anything outside `L10-11 @ W1[4-7]` beating `L11H01` (`0.3820`): `False`
