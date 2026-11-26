# CIRCUIT First-Milestone Write-Up (Draft)

Status: Draft for Muse and Adele review (Week 1 milestone summary). Every figure below traces directly to saved JSONs and images in `experiments/e0/`, `experiments/e1/`, and `experiments/e3/`.

## 1. Research Question and Experimental Setup

We investigate whether pretrained Diffusion Transformers (PixArt-Sigma, 0.6B DiT, 28 transformer blocks, 16 heads per block, T5-XXL text conditioning) develop temporally localized, causally identifiable computational mechanisms for spatial relationships (`left` vs `right`) and color-attribute binding.

All patching interventions operate on the conditional Classifier-Free Guidance (CFG) branch using deterministic `DPMSolverMultistepScheduler` (20 steps, guidance scale `4.5`, `torch.bfloat16`), with explicit checkpoint labeling (`512-MS` = `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`; `1024-MS` = `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`). Because E0 (`experiments/e0/failures.md`) showed that terse `"Red cube left of blue sphere"` vs `"Red cube right of blue sphere"` prompts are preposition-insensitive on `1024-MS` at matched seeds, E1 evaluates on explicit positional counterfactuals (`"A {color_a} {obj_a} on the left and a {color_b} {obj_b} on the right"` vs swapped) where clean and corrupt reliably differ, reserving terse `"left of / right of"` for E4 generalization.

## 2. Main Findings

### 2.1 Temporal and Layer Localization of Spatial Transfer (`512-MS`)

Across both the `dev` pilot (`N=6` valid counterfactuals, `experiments/e1/e1_pilot_results.json`) and the held-out `test` split (`N=13` valid counterfactuals from `18` instances across `6` object-disjoint pairs x `3` seeds, `experiments/e1/e1_full_results.json`), cross-attention output patching (`attn2_out`) is sharply localized to **early denoising steps `4-7` (`W1[4-7]`)** and **layers `L2-3` and `L10-11`**, with near-zero transfer (`tau <= 0.082`) above `L14` or in late steps `W4[16-19]`:

- **`L10-11 @ W1[4-7]` (Primary Peak):**
  - Pilot (`512-MS`, `dev`): `tau = 0.7104`, `flip_rate = 0.6667`, `sigma = 0.9263`, `detection_rate = 1.0000`.
  - Full Sweep (`512-MS`, `test`): `tau = 0.6903` (`95% CI [0.4643, 0.8956]`), `flip_rate = 0.6923`, `sigma = 0.9349` (`95% CI [0.8492, 0.9814]`), `detection_rate = 0.9231`, `color_preserve_rate = 1.0000`.
- **`L2-3 @ W1[4-7]` (Replicated Secondary Peak):**
  - Pilot (`512-MS`, `dev`): `tau = 0.6648` (`95% CI [0.3330, 0.9160]`), `flip_rate = 0.5000`, `sigma = 0.8176`.
  - Full Sweep (`512-MS`, `test`): `tau = 0.5522` (`95% CI [0.3123, 0.7907]`), `flip_rate = 0.5385`, `sigma = 0.9099`, `detection_rate = 1.0000`.
- **Controls on Full Sweep (`512-MS`, `test`):**
  - `random_head` (`L3H10 @ W1[4-7]`): `tau = 0.0042` (`95% CI [0.0001, 0.0118]`), `flip_rate = 0.0000`, `sigma = 0.9722`, `color_preserve_rate = 0.8333`.
  - `wrong_timestep` (`L10-11 @ W4[16-19]`): `tau = 0.0005` (`95% CI [0.0002, 0.0009]`), `flip_rate = 0.0000`, `sigma = 0.9986`, `color_preserve_rate = 1.0000`.
  - `same_prompt_noop` (corrupt into corrupt at `L10-11 @ W1[4-7]`): `tau = 0.0000`, `flip_rate = 0.0000`, `sigma = 1.0000`, `color_preserve_rate = 1.0000`.

### 2.2 Two-Regime Head Structure (H2 Refinement)

Per-head patching (`attn2_head_pre_proj`, `experiments/e1/head_analysis.md`, `experiments/e1/e1_full_results.json`) reveals two distinct regimes rather than a single uniform H2 pattern:

1. **Single-head text-reading bottleneck (`L3H02`):** Within `L2-3 @ W1[4-7]` (`32` heads), one single head **`L3H02`** reaches `tau = 0.5468` (`95% CI [0.3047, 0.7865]`), `flip_rate = 0.5385`, `sigma = 0.9113`, accounting for **99.0%** of the entire `L2-3` layer group (`0.5522`). All other 31 heads in `L2-3` have `tau <= 0.0955`.
2. **Distributed 6-head routing coalition (`L10-11`):** Within `L10-11 @ W1[4-7]` (`32` heads), single heads cap at `L10H15` (`tau = 0.3129`, `95% CI [0.1143, 0.5364]`, `flip_rate = 0.2308`, `sigma = 0.9884`) and `L10H06` (`tau = 0.2585`, `flip_rate = 0.2308`), followed by `L11H11` (`0.1710`), `L10H08` (`0.1357`), `L11H01` (`0.1123`), and `L11H13` (`0.0940`), while 20 of 32 heads remain below `0.02`. Together these heads compose to the layer-group `tau = 0.6903`.

### 2.3 Cross-Attention-to-Self-Attention Handoff (`512-MS`)

Sublayer comparison (`experiments/e1/e1_1024_check.json`, `4` test pairs x `3` seeds = `12` instances) shows a three-stage cross-modal to intra-image handoff:

1. **`L2-3 @ W1[4-7]` (Cross-Attention Text Reading):** `attn2_out` (`tau = 0.6795`) and `attn2_in_txt` (`tau = 0.6300`) carry the spatial transfer, while `attn1_out` (`0.0723`) and `ff_out` (`0.0753`) do not.
2. **`L10-11 @ W1[4-7]` (Cross-Modal Query Routing + Self-Attention):** `attn2_out` (`tau = 0.7912`, `flip_rate = 0.7778`, `sigma = 0.9751`) and `attn1_out` (`tau = 0.7191`, `flip_rate = 0.6667`, `sigma = 0.9062`) dominate, whereas swapping text inputs alone (`attn2_in_txt`) drops sharply to `0.1503` (and `ff_out` to `0.1402`).
3. **`L6-7 @ W2[8-11]` (Self-Attention Consolidation):** As `attn2_out` fades (`0.2166`) and `attn2_in_txt` vanishes (`0.0008`), downstream self-attention `attn1_out` reaches **`tau = 0.9424`** (`flip_rate = 0.8889`, `sigma = 0.8979`).

### 2.4 `1024-MS` Confirmation

On `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (`1024-MS`, `3` test pairs x `3` seeds, `7` valid counterfactuals, `experiments/e1/e1_1024_check.json`), spatial transfer replicates at both peak cells at reduced magnitude:
- `L10-11 @ W1[4-7]`: `tau = 0.4901`, `flip_rate = 0.4286`, `sigma = 0.8402`, `detection_rate = 1.0000`, `color_preserve_rate = 0.5714`.
- `L2-3 @ W1[4-7]`: `tau = 0.4991`, `flip_rate = 0.4286`, `sigma = 0.8412`, `detection_rate = 1.0000`, `color_preserve_rate = 0.5000`.
- Controls: `ctrl_random_head` (`L2H10`: `tau = 0.0001`), `ctrl_wrong_timestep` (`tau = 0.0002`), `ctrl_same_prompt_noop` (`tau = 0.0000`).

### 2.5 Seed-Invariance of Spatial Layout vs. Fragility of Color

Shifted-seed patching (`seed_variation`, patching `clean` activations from `s + 999` into `corrupt` at `L10-11 @ W1[4-7]`) consistently transfers spatial layout while degrading object color preservation:
- Pilot (`512-MS`, `dev`): `tau = 0.4980`, `flip_rate = 0.5000`, `sigma = 0.8148`, `color_preserve_rate = 0.4000`.
- Full Sweep (`512-MS`, `test`): `tau = 0.7063` (`95% CI [0.4742, 0.9003]`), `flip_rate = 0.6923`, `sigma = 0.9100`, `color_preserve_rate = 0.8000`.
- `1024-MS` Check (`test`): `tau = 0.3356`, `flip_rate = 0.2857`, `sigma = 0.7905`, `color_preserve_rate = 0.4286`.

Spatial left/right routing in `L10-11` is therefore substantially seed-invariant (prompt-driven), whereas faithful attribute binding depends on matched latent geometry; we make no claim of matched-seed specificity under mismatched seeds.

### 2.6 Seven-Baseline Comparison (`512-MS`)

Across both the pilot (`32` heads in `L10-11`, `N=6` valid, `experiments/e1/baseline_comparison.json`) and the full sweep (`64` heads across `L2, L3, L10, L11`, `N=12` valid, `experiments/e1/baseline_comparison_full.json`, `interpretability/baseline_comparison.md`):

- **Prompt-to-Prompt (`A_clean` attention-map injection without `V` patching):** On the pilot, layer-group P2P outperformed activation patching (`tau = 0.8261`, `flip = 0.8333` vs patching `0.7104`). On the full `test` split (`N=12`), P2P and patching were comparable at `L10-11 @ W1[4-7]` (`0.6702` vs `0.6903`) and `L2-3 @ W1[4-7]` (`0.5956` vs `0.5522`), while per-head P2P correlated strongly with causal patching across all 64 heads (`Spearman rho = 0.483, p = 0.0001`, `4/5` top-5 overlap: `L3H2, L10H6, L10H15, L11H11`; `rho = 0.637` inside `L10-11`).
- **DAAM inside-box grounding mass:** Significantly correlated with causal patching (`rho = 0.370, p = 0.0026` over 64 heads; `rho = 0.424` in `L10-11`).
- **Roy destructive ablation vs. patching asymmetry:** On the `N=6` pilot, single-head mean ablation had `rho = 0.026` (`p = 0.8867`) and assigned `L10H15` `0.000` impact due to head redundancy. At `N=12` across 64 heads, ablation impact rose to `rho = 0.383` (`p = 0.0018`, `2/5` overlap: `L3H2, L10H15`), though still missing secondary coalition heads such as `L10H6` (`0.082`) and `L11H11`.
- **Observational baselines failing to locate the circuit:** Zarei attention contribution (`rho = 0.232, p = 0.0651` all 64; `rho = 0.022` pilot), Maliha KL spike (`rho = -0.159, p = 0.2089` all 64; significantly negative `-0.553` in `L2-3`), CIRCUIT spatial gradient diagnostic (`rho = 0.187, p = 0.1386`), and Revelio Diff-C-style spatially pooled linear probe (`rho = -0.087, p = 0.4939`) fail to rank the causal circuit reliably.

What activation patching adds beyond Prompt-to-Prompt attention-map injection: P2P shows that attention probability matrices carry the spatial relation signal, whereas causal activation patching provides the full 2D `(layer, timestep)` localization map, sublayer attribution (`attn2_in_txt` vs `attn2_out` vs `attn1_out` vs `ff_out`), single-head vs coalition resolution (`L3H2` vs `L10-11`), and matched counterfactual vs ablation necessity tests.

### 2.7 E3 Color-Binding Pilot (`512-MS`, `dev`)

On 3 color-binding pairs x 3 seeds (`N=7` valid counterfactuals, `experiments/e3/e3_pilot_results.json`), `L10-11 @ W1[4-7]` achieved `tau_color = 0.4027`, `flip_rate = 0.1429`, `sigma = 0.9872`, beating `random_head` (`L3H10`: `tau_color = 0.0080`), `wrong_timestep` (`0.0049`), and `same_prompt_noop` (`0.0000`) -> **Pilot Gate: PASS**. Crucially, `L2-3 @ W1[4-7]` (`tau_color = 0.0768`, `flip = 0.0000`) and spatial master head `L3H02` (`tau_color = 0.009`) show near-zero color-binding transfer, whereas `L11H01 @ W1[4-7]` reaches `tau_color = 0.2457` (`flip = 0.1429`), giving initial evidence of functional dissociation between spatial-relation and color-binding heads.

## 3. Honest Limitations and Artifacts

1. **Imperfect flip rate (`0.6923` on `512-MS` test, `0.4286` on `1024-MS`):** Transfer is substantial and statistically robust against controls, not 100%.
2. **Observed per-instance artifacts (`experiments/e1/e1_full_grid.png`):**
   - `spatial_lr_test_0001` (`seed=42`): patching `L10-11 @ W1[4-7]` did not flip the yellow dog to the left, keeping the corrupt layout while introducing a yellow mug on top of the green books.
   - `spatial_lr_test_0011` (`seed=42`): patching flipped the purple umbrella back to the left, but the right-hand apple turned red instead of green (attribute corruption during layout transfer).
3. **Small sample size on `1024-MS`:** The `1024-MS` confirmation tested 3 pairs x 3 seeds (`N=7` valid) at `L10-11` and `L2-3` rather than a full 14-layer-group x 5-window sweep.
4. **Prompt-to-Prompt competitiveness:** On the `N=6` pilot, layer-group `A_clean` injection (`0.8261`) exceeded `attn2_out` activation patching (`0.7104`), and on `N=12` test pairs P2P (`0.6702`) remained essentially tied with patching (`0.6903`).
5. **E3 color-binding pilot partial flip rate:** Although `L10-11 @ W1[4-7]` passed the pilot gate (`tau_color = 0.4027` vs `0.0080`), binary color-binding flip rate was only `0.1429`, indicating either distributed/later color-binding layers outside `L10-11 @ W1[4-7]` or weaker single-block sufficiency for color binding.
