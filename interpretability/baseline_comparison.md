# Baseline Comparison Against E1 Causal Tau Maps

## 1. Full Sweep Comparison (`test` split, 64 heads across L2, L3, L10, L11 @ W1[4-7])

- Checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`)
- Valid counterfactuals in `baseline_comparison_full.json`: `N=12` overall/`L2-3` (`N=15` for `L10-11` P2P) from `18` instances (`6` test pairs x `3` seeds). This differed from `e1_full_results.json` (`N=13`) because capturing attention maps via `CrossAttentionCaptureProcessor` (`float32` matmul) shifted 1-2 borderline baseline detections relative to default `bfloat16` SDPA.
- Same-instance paired re-report under shared unmodified SDPA baseline masks (`experiments/e1/p2p_vs_patching_paired.json`, `N=13` shared valid):
  - `L10-11 @ W1[4-7]`: activation patching `tau = 0.6845` (`flip = 0.6923`, `sigma = 0.9355`) vs Prompt-to-Prompt `tau = 0.6857` (`flip = 0.6923`, `sigma = 0.9788`).
  - `L2-3 @ W1[4-7]`: activation patching `tau = 0.5540` (`flip = 0.4615`, `sigma = 0.9361`) vs Prompt-to-Prompt `tau = 0.5494` (`flip = 0.5385`, `sigma = 0.9332`).
- Note on Revelio baseline: implemented as a Diff-C-style cross-validated linear probe on spatially pooled cached per-head activations rather than training a full k-SAE.

### All 64 Heads (`L2, L3, L10, L11`)

| Baseline | Spearman rho | p-value | Top-5 Overlap | Overlapping Heads | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1_zarei_attn_contribution | 0.232 | 0.0651 | 1/5 | L3H2 | Weak/Partial |
| 2_maliha_kl_spike | -0.159 | 0.2089 | 0/5 | none | Does not find circuit |
| 3_circuit_spatial_grad_wang_prior | 0.187 | 0.1386 | 1/5 | L3H2 | Weak/Partial |
| 4_daam_box_mass | 0.370 | 0.0026 | 2/5 | L10H8, L10H15 | Agrees |
| 5_roy_ablation_impact | 0.383 | 0.0018 | 2/5 | L3H2, L10H15 | Agrees |
| 6_revelio_linear_probe_auc | -0.087 | 0.4939 | 1/5 | L3H2 | Weak/Partial |
| 7_prompt_to_prompt_attn_map_tau | 0.483 | 0.0001 | 4/5 | L3H2, L10H6, L10H15, L11H11 | Agrees |

### Subset: `L2-3` (32 heads)

| Baseline | Spearman rho | p-value | Top-5 Overlap | Overlapping Heads |
| --- | --- | --- | --- | --- |
| 1_zarei_attn_contribution | 0.125 | 0.4967 | 3/5 | L3H2, L3H7, L3H8 |
| 2_maliha_kl_spike | -0.553 | 0.0010 | 0/5 | none |
| 3_circuit_spatial_grad_wang_prior | 0.098 | 0.5941 | 2/5 | L3H1, L3H2 |
| 4_daam_box_mass | 0.352 | 0.0485 | 1/5 | L3H8 |
| 5_roy_ablation_impact | 0.320 | 0.0746 | 2/5 | L3H1, L3H2 |
| 6_revelio_linear_probe_auc | 0.156 | 0.3949 | 2/5 | L3H2, L3H7 |
| 7_prompt_to_prompt_attn_map_tau | 0.228 | 0.2094 | 3/5 | L3H1, L3H2, L3H8 |

### Subset: `L10-11` (32 heads)

| Baseline | Spearman rho | p-value | Top-5 Overlap | Overlapping Heads |
| --- | --- | --- | --- | --- |
| 1_zarei_attn_contribution | 0.250 | 0.1670 | 0/5 | none |
| 2_maliha_kl_spike | 0.257 | 0.1557 | 1/5 | L10H15 |
| 3_circuit_spatial_grad_wang_prior | 0.230 | 0.2057 | 2/5 | L10H15, L11H1 |
| 4_daam_box_mass | 0.424 | 0.0155 | 3/5 | L10H6, L10H8, L10H15 |
| 5_roy_ablation_impact | 0.404 | 0.0218 | 2/5 | L10H15, L11H1 |
| 6_revelio_linear_probe_auc | -0.090 | 0.6261 | 1/5 | L10H8 |
| 7_prompt_to_prompt_attn_map_tau | 0.637 | 0.0001 | 3/5 | L10H6, L10H15, L11H11 |

## 2. Pilot Comparison Reference (`dev` split, 32 heads in L10-11 @ W1[4-7], N=6 valid)

- Layer-group patching `tau = 0.7104` vs Prompt-to-Prompt `tau = 0.8261`.

| Baseline | Spearman rho | p-value | Top-5 Overlap |
| --- | --- | --- | --- |
| 1_zarei_attn_contribution | 0.022 | 0.9033 | 0/5 |
| 2_maliha_kl_spike | 0.224 | 0.2186 | 0/5 |
| 3_circuit_spatial_grad_wang_prior | 0.314 | 0.0799 | 2/5 |
| 4_daam_box_mass | 0.413 | 0.0188 | 3/5 |
| 5_roy_ablation_impact | 0.026 | 0.8867 | 2/5 |
| 6_revelio_linear_probe_auc | -0.056 | 0.7599 | 0/5 |
| 7_prompt_to_prompt_attn_map_tau | 0.568 | 0.0007 | 4/5 |
