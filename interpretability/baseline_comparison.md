# Baseline Comparison Against E1 Pilot Causal Tau Map

- Checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`)
- Grid evaluated: `32` heads across layers `[10, 11]` at window `W1[4-7]` (`steps [4, 5, 6, 7]`), `N=6` valid counterfactual instances.
- Layer-group activation patching (`attn2_out` at `L10-11 @ W1[4-7]`): `tau = 0.7104`, `flip_rate = 0.6667`.
- Layer-group Prompt-to-Prompt (`A_clean` injection without `V` patching at `L10-11 @ W1[4-7]`): `tau = 0.8261`, `flip_rate = 0.8333`.
- Note on Revelio baseline: implemented as a Diff-C-style cross-validated linear probe on spatially pooled cached per-head activations rather than training a full k-SAE.

## Per-Head Rank Agreement with Causal Patching Tau

| Baseline | Spearman rho | p-value | Top-5 Overlap | Overlapping Heads | Agreement Verdict |
| --- | --- | --- | --- | --- | --- |
| 1_zarei_attn_contribution | 0.022 | 0.9033 | 0/5 | none | Does not find circuit |
| 2_maliha_kl_spike | 0.224 | 0.2186 | 0/5 | none | Weak/Partial |
| 3_circuit_spatial_grad_wang_prior | 0.314 | 0.0799 | 2/5 | L10H15, L11H1 | Weak/Partial |
| 4_daam_box_mass | 0.413 | 0.0188 | 3/5 | L10H6, L10H15, L11H4 | Agrees |
| 5_roy_ablation_impact | 0.026 | 0.8867 | 2/5 | L11H4, L11H11 | Weak/Partial |
| 6_revelio_linear_probe_auc | -0.056 | 0.7599 | 0/5 | none | Does not find circuit |
| 7_prompt_to_prompt_attn_map_tau | 0.568 | 0.0007 | 4/5 | L10H6, L10H15, L11H4, L11H11 | Agrees |

## Interpretation

Observational metrics (Zarei attention contribution, Maliha KL spike, DAAM inside-box mass, Revelio linear probe, and CIRCUIT's spatial gradient diagnostic) measure where token signal correlates with prompt differences or spatial layout, whereas Roy ablation tests necessity and Prompt-to-Prompt tests attention-map transfer alone. Comparing all 7 against causal activation patching (`tau`) on the same `(layer, head, timestep)` grid tests directly which diagnostics recover the true causal circuit heads (`L10H15`, `L10H6`, `L10H13`, `L11H11`, `L11H4`) and whether attention maps alone suffice without value-stream activation patching.
