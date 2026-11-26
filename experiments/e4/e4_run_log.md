# E4 Generalization Run Log

## Execution Environment

- Host: `ai-lab2`
- GPUs visible: `['NVIDIA GeForce RTX 4090']` (gen: `cuda:0`, eval: `cuda:0`)
- Python: `3.10.12`
- Torch: `2.6.0+cu124`, Diffusers: `0.41.0`, Transformers: `5.20.2`

## Part (a.1): Terse-Template Rescue (`512-MS`, `to_the_side_of` failing instances)

- Checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`), `N = 8` failing terse instances.

| Intervention | Tau (Rescue) | Flip Rate | Sigma | Detection Rate |
| --- | --- | --- | --- | --- |
| L3H02_W1 | 0.0297 | 0.0000 | 0.9948 | 1.0000 |
| L10_11_coalition_W1 | 0.4683 | 0.6250 | 0.8725 | 0.8750 |
| L10_11_full_W1 | 0.4493 | 0.6250 | 0.8760 | 0.8750 |
| L6_7_attn1_W2 | 0.8478 | 1.0000 | 0.7966 | 1.0000 |
| combined_circuit | 0.8438 | 1.0000 | 0.8866 | 1.0000 |
| random_head | 0.0027 | 0.0000 | 0.9986 | 1.0000 |

## Part (a.2): E0 Exact Failure Rescue (`1024-MS`)

| Case ID | Terse Base dx (rel) | Donor dx (rel) | L3H02 dx (rel) | L10-11 dx (rel) | L6-7 attn1 dx (rel) | Combined dx (rel) |
| --- | --- | --- | --- | --- | --- | --- |
| e0_p1_s42 | -0.3848685696721077 (right) | 0.44215109944343567 (left) | -0.41929565370082855 (right) | 0.4203472435474396 (left) | 0.43372149020433426 (left) | 0.4370746985077858 (left) |
| e0_p1_s123 | -0.43704185634851456 (right) | 0.44922542572021484 (left) | -0.4287644401192665 (right) | 0.44798801094293594 (left) | 0.46515610069036484 (left) | 0.4693758487701416 (left) |
| e0_p4_s42 | -0.23616454005241394 (right) | 0.44823259115219116 (left) | -0.29128260165452957 (right) | -0.37045567482709885 (right) | 0.07085592299699783 (left) | 0.434342660009861 (left) |
| e0_p5_s42 | -0.2959994301199913 (right) | 0.4445189833641052 (left) | -0.2951895073056221 (right) | -0.20000425726175308 (right) | 0.4328829348087311 (left) | 0.43444931507110596 (left) |

## Part (b.1): Held-Out OOD Spatial (`512-MS`)

- Pairs: `['spatial_lr_ood_0000', 'spatial_lr_ood_0001', 'spatial_lr_ood_0002']`

| Condition | Tau | Flip Rate | Sigma | Detection Rate |
| --- | --- | --- | --- | --- |
| L10-11_W1 | 0.9901 | 1.0000 | 0.9159 | 1.0000 |
| L2-3_W1 | 0.8196 | 0.7778 | 0.9158 | 1.0000 |
| L3H02_W1 | 0.6485 | 0.6667 | 0.8785 | 1.0000 |
| L6-7_attn1_W2 | 0.9968 | 1.0000 | 0.8408 | 1.0000 |
| ctrl_random_head | 0.0005 | 0.0000 | 0.9986 | 1.0000 |
| ctrl_wrong_timestep | 0.0043 | 0.0000 | 0.9976 | 1.0000 |
| ctrl_same_prompt_noop | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| ctrl_seed_variation | 0.8953 | 0.8889 | 0.9159 | 1.0000 |

## Part (b.2): Held-Out OOD Color Binding (`512-MS`)

- Pairs: `['color_binding_ood_0001', 'color_binding_ood_0002', 'color_binding_ood_0003']`

| Condition | Tau | Flip Rate | Sigma | Detection Rate |
| --- | --- | --- | --- | --- |
| L10-11_W1 | 0.1128 | 0.0000 | 0.9890 | 1.0000 |
| L2-3_W1 | 0.0315 | 0.0000 | 0.9923 | 1.0000 |
| L11H01_W1 | 0.0791 | 0.0000 | 0.9914 | 1.0000 |
| ctrl_random_head | 0.0398 | 0.0000 | 0.9989 | 1.0000 |
| ctrl_wrong_timestep | 0.0107 | 0.0000 | 0.9986 | 1.0000 |
| ctrl_same_prompt_noop | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| ctrl_seed_variation | 0.2780 | 0.1429 | 0.9886 | 1.0000 |
