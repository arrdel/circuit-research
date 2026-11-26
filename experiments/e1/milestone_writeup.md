# CIRCUIT First-Milestone Write-Up (Week 1 Finalized Draft)

Status: Finalized Week 1 draft for Muse and Adele review. Every figure below traces directly to saved JSONs and images in `experiments/e0/`, `experiments/e1/`, and `experiments/e3/`.

## 1. Research Question and Experimental Setup

We investigate whether pretrained Diffusion Transformers (PixArt-Sigma, 0.6B DiT, 28 transformer blocks, 16 heads per block, T5-XXL text conditioning) develop temporally localized, causally identifiable computational mechanisms for spatial relationships (`left` vs `right`) and color-attribute binding.

All patching interventions operate on the conditional Classifier-Free Guidance (CFG) branch using deterministic `DPMSolverMultistepScheduler` (20 steps, guidance scale `4.5`, `torch.bfloat16`), with explicit checkpoint labeling (`512-MS` = `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`; `1024-MS` = `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`). Because E0 (`experiments/e0/failures.md`) established that terse `"Red cube left of blue sphere"` vs `"Red cube right of blue sphere"` prompts are preposition-insensitive on `1024-MS` at matched seeds, E1 uses explicit positional counterfactuals (`"A {color_a} {obj_a} on the left and a {color_b} {obj_b} on the right"` vs swapped) where clean and corrupt reliably differ, and E3 uses color-swapped counterfactuals with fixed left/right positions (`"A {color_a} {obj_a} on the left and a {color_b} {obj_b} on the right"` vs swapped colors), reserving terse `"left of / right of"` for E4 generalization.

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

Per-head patching (`attn2_head_pre_proj`, `experiments/e1/head_analysis.md`, `experiments/e1/e1_full_results.json`) reveals two distinct regimes across the spatial circuit:

1. **Single-head text-reading bottleneck (`L3H02`):** Within `L2-3 @ W1[4-7]` (`32` heads), one single head **`L3H02`** reaches `tau = 0.5468` (`95% CI [0.3047, 0.7865]`), `flip_rate = 0.5385`, `sigma = 0.9113`, accounting for **99.0%** of the entire `L2-3` layer group (`0.5522`). All other 31 heads in `L2-3` have `tau <= 0.0955`.
2. **Distributed 6-head routing coalition (`L10-11`):** Within `L10-11 @ W1[4-7]` (`32` heads), single heads cap at `L10H15` (`tau = 0.3129`, `95% CI [0.1143, 0.5364]`, `flip_rate = 0.2308`, `sigma = 0.9884`) and `L10H06` (`tau = 0.2585`, `flip_rate = 0.2308`), followed by `L11H11` (`0.1710`), `L10H08` (`0.1357`), `L11H01` (`0.1123`), and `L11H13` (`0.0940`), while 20 of 32 heads remain below `0.02`. Together these heads compose to `tau = 0.6903`.

### 2.3 Cross-Attention-to-Self-Attention Handoff (`512-MS`)

Sublayer comparison (`experiments/e1/e1_1024_check.json`, `4` test pairs x `3` seeds = `12` instances) shows a three-stage cross-modal to intra-image handoff:

1. **`L2-3 @ W1[4-7]` (Cross-Attention Text Reading):** `attn2_out` (`tau = 0.6795`) and `attn2_in_txt` (`tau = 0.6300`) carry the spatial transfer, while `attn1_out` (`0.0723`) and `ff_out` (`0.0753`) do not.
2. **`L10-11 @ W1[4-7]` (Cross-Modal Query Routing + Self-Attention):** `attn2_out` (`tau = 0.7912`, `flip_rate = 0.7778`, `sigma = 0.9751`) and `attn1_out` (`tau = 0.7191`, `flip_rate = 0.6667`, `sigma = 0.9062`) dominate, whereas swapping text inputs alone (`attn2_in_txt`) drops to `0.1503` (and `ff_out` to `0.1402`).
3. **`L6-7 @ W2[8-11]` (Self-Attention Consolidation):** As `attn2_out` fades (`0.2166`) and `attn2_in_txt` vanishes (`0.0008`), downstream self-attention `attn1_out` reaches **`tau = 0.9424`** (`flip_rate = 0.8889`, `sigma = 0.8979`).

### 2.4 `1024-MS` Confirmation

On `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (`1024-MS`, `3` test pairs x `3` seeds, `7` valid counterfactuals, `experiments/e1/e1_1024_check.json`), spatial transfer replicates at both peak cells at reduced magnitude:
- `L10-11 @ W1[4-7]`: `tau = 0.4901`, `flip_rate = 0.4286`, `sigma = 0.8402`, `detection_rate = 1.0000`, `color_preserve_rate = 0.5714`.
- `L2-3 @ W1[4-7]`: `tau = 0.4991`, `flip_rate = 0.4286`, `sigma = 0.8412`, `detection_rate = 1.0000`, `color_preserve_rate = 0.5000`.
- Controls: `ctrl_random_head` (`L2H10`: `tau = 0.0001`), `ctrl_wrong_timestep` (`tau = 0.0002`), `ctrl_same_prompt_noop` (`tau = 0.0000`).

### 2.5 Seed-Invariance of Spatial Layout vs. Fragility of Color

Shifted-seed patching (`seed_variation`, `s + 999` into `s` at `L10-11 @ W1[4-7]`) consistently transfers spatial layout while degrading object color preservation in E1:
- E1 Pilot (`512-MS`, `dev`): `tau = 0.4980`, `flip_rate = 0.5000`, `sigma = 0.8148`, `color_preserve_rate = 0.4000`.
- E1 Full Sweep (`512-MS`, `test`): `tau = 0.7063` (`95% CI [0.4742, 0.9003]`), `flip_rate = 0.6923`, `sigma = 0.9100`, `color_preserve_rate = 0.8000`.
- E1 `1024-MS` Check (`test`): `tau = 0.3356`, `flip_rate = 0.2857`, `sigma = 0.7905`, `color_preserve_rate = 0.4286`.

Spatial left/right routing in `L10-11` is therefore substantially seed-invariant (prompt-driven), whereas faithful attribute binding depends on matched latent geometry; we make no claim of matched-seed specificity under mismatched seeds.

### 2.6 Seven-Baseline Comparison (`512-MS`, Pilot vs. Full Sweep)

Across the pilot (`32` heads in `L10-11`, `N=6` valid, `experiments/e1/baseline_comparison.json`) and the full `test` sweep (`64` heads across `L2, L3, L10, L11`, `N=12` valid, `experiments/e1/baseline_comparison_full.json`, `interpretability/baseline_comparison.md`):

- **Prompt-to-Prompt (`A_clean` attention-map injection without `V` patching):** On the `N=6` pilot, layer-group P2P outperformed activation patching (`tau = 0.8261`, `flip = 0.8333` vs patching `0.7104`). In `experiments/e1/baseline_comparison_full.json`, `N=12` overall (`L2-3`) vs `N=15` (`L10-11` P2P) differed from `e1_full_results.json` (`N=13`) because capturing attention maps via `CrossAttentionCaptureProcessor` (`float32` matmul) shifted 1-2 borderline baseline detections relative to default `bfloat16` SDPA. Re-running both methods on the **exact shared `N=13` valid set** (`experiments/e1/p2p_vs_patching_paired.json`) shows parity: at `L10-11 @ W1[4-7]`, patching `tau = 0.6845` (`95% CI [0.4503, 0.8939]`, `flip = 0.6923`, `sigma = 0.9355`) vs P2P `tau = 0.6857` (`95% CI [0.4509, 0.8942]`, `flip = 0.6923`, `sigma = 0.9788`); at `L2-3 @ W1[4-7]`, patching `tau = 0.5540` (`flip = 0.4615`) vs P2P `tau = 0.5494` (`flip = 0.5385`). Across all 64 heads (`N=12`), per-head P2P was the strongest baseline predictor of causal patching (`Spearman rho = 0.483, p = 0.0001`, `4/5` top-5 overlap: `L3H2, L10H6, L10H15, L11H11`; `rho = 0.637` within `L10-11`).
- **DAAM inside-box grounding mass:** Significantly correlated across 64 heads (`rho = 0.370, p = 0.0026`, `2/5` overlap: `L10H8, L10H15`; `rho = 0.424` in `L10-11`, `rho = 0.352` in `L2-3`), though in `L2-3` it ranked `L3H5` (`0.554`) above `L3H2` (`0.413`).
- **Roy destructive ablation vs. patching asymmetry:** On the `N=6` pilot, single-head mean ablation had `rho = 0.026` (`p = 0.8867`) and assigned `L10H15` `0.000` impact due to head redundancy. At `N=12` over 64 heads, ablation correlation improved (`rho = 0.383, p = 0.0018`, `2/5` overlap: `L3H2, L10H15`), but still assigned near-zero or weak impact to secondary coalition heads (`L10H6: 0.082`).
- **Observational baselines failing to locate the circuit:** Zarei attention contribution (`rho = 0.232, p = 0.0651` over 64 heads; `0.022` pilot), Maliha KL spike (`rho = -0.159, p = 0.2089`; significantly negative `-0.553` in `L2-3`), CIRCUIT spatial gradient diagnostic (`rho = 0.187, p = 0.1386`), and Revelio Diff-C linear probe (`rho = -0.087, p = 0.4939`) all failed to rank the full 64-head circuit significantly.

**Framing beyond steering and P2P (`arXiv:2611.26627`, `arXiv:2208.01626`):** Fixed activation steering (`2611.26627`) applies an input-independent global shift across all prompts, whereas Prompt-to-Prompt (`2208.01626`) swaps attention probability maps. CIRCUIT does not claim patching is the only way to transfer a spatial relation; rather, input-dependent counterfactual patching uniquely exposes *where*, *when*, and *which heads and sublayers* implement compositional routing (`L3H2` reading text K/V, the `L10-11` coalition, `L11H1` color binding, and the `attn2`-to-`attn1` handoff).

### 2.7 E3 Color-Binding Pilot and Full Sweep: Head-Level Double Dissociation (`512-MS`)

Both the E3 pilot (`dev`, `N=7` valid, `experiments/e3/e3_pilot_results.json`) and E3 full sweep (`test`, `N=15` valid from 5 fresh object-disjoint pairs x 3 seeds, `experiments/e3/e3_full_results.json`) confirm a clean **head-level dissociation** between spatial layout and color-attribute binding:

- **Pilot (`512-MS`, `dev`, `N=7`):** `L10-11 @ W1[4-7]` `tau_color = 0.4027` (`95% CI [0.226, 0.586]`, `flip = 0.1429`, `sigma = 0.9872`) vs `L2-3 @ W1[4-7]` `tau_color = 0.0768` (`flip = 0.0000`), beating `random_head` (`0.0080`), `wrong_timestep` (`0.0049`), and `same_prompt_noop` (`0.0000`) -> Gate **PASS**. Peak single head: `L11H01` (`tau_color = 0.2457`).
- **Full Sweep (`512-MS`, `test`, `N=15`):** Confirms the pilot in the exact same direction:
  - `L10-11 @ W1[4-7]`: `tau_color = 0.3535` (`95% CI [0.168, 0.568]`), `flip_rate = 0.3333`, `sigma = 0.9816`, **`spatial_preserve_rate = 1.0000`** (`15/15` kept fixed left/right positions intact).
  - `L2-3 @ W1[4-7]`: `tau_color = 0.0275` (`95% CI [0.009, 0.050]`), `flip_rate = 0.0667`, `sigma = 0.9931`, `spatial_preserve_rate = 1.0000`.
  - Controls: `random_head` (`L3H10`: `0.0114`, `flip = 0.0667`), `wrong_timestep` (`W4[16-19]`: `0.0054`, `flip = 0.0000`), `same_prompt_noop` (`0.0000`), `seed_variation` (`s+999`: `0.3658`, `flip = 0.3333`, `sigma = 0.9794`, `spatial_preserve_rate = 1.0000`).
  - **Single-head color-binding peak (`L11H01`):** Across all 64 heads in `L2, L3, L10, L11`, **`L11H01 @ W1[4-7]`** reaches **`tau_color = 0.3820` (`flip_rate = 0.3333`, `sigma = 0.9844`)**, slightly exceeding the full `L10-11` layer group (`0.3535`), while 62 other heads remain `<= 0.043` and `L11H13` reaches `tau_color = 0.0511` (`flip_rate = 0.0000`)!
  - **Full `L0-27 x W0-4` Color Scan (`experiments/e3/e3_color_scan.json`, `N=12` valid):** Confirms no color-binding hotspot exists outside `L10-11 @ W1[4-7]` (`tau_color = 0.4090`, `flip = 0.4167`): all other 69 `attn2_out` cells remain `<= 0.2053` (`L10-11 @ W0[0-3]`) and `<= 0.1016` outside `L10-11`, so nothing beats `L11H01` (`0.3820`). Sublayer checks show downstream `L6-7 attn1_out @ W2[8-11]` reaches `tau_color = 0.3621` (`flip = 0.1667`), whereas `ff_out` everywhere is `<= 0.1521`.
- **Double Dissociation Summary (`512-MS`, `test`):**
  - `L3H02 @ W1[4-7]`: `tau_spatial = 0.5468` (`flip = 0.5385`) vs `tau_color = 0.009` (`flip = 0.0000`).
  - `L10H15 @ W1[4-7]`: `tau_spatial = 0.3129` (`flip = 0.2308`) vs `tau_color = 0.012` (`flip = 0.0000`).
  - `L11H01 @ W1[4-7]`: `tau_color = 0.3820` (`flip = 0.3333`) vs `tau_spatial = 0.1123` (`flip = 0.0769`).

## 3. Honest Limitations and Per-Instance Artifacts

1. **Imperfect binary flip rates:** Spatial flip rate is `0.6923` (`512-MS` test) and `0.4286` (`1024-MS`); color-binding flip rate is `0.1429` (`512-MS` E3 pilot) and `0.3333` (`512-MS` E3 full).
2. **Observed per-instance artifacts (`experiments/e1/e1_full_grid.png`):**
   - `spatial_lr_test_0001` (`seed=42`): patching `L10-11 @ W1[4-7]` did not flip the yellow dog to the left, keeping the corrupt layout while introducing a yellow mug on top of the green books.
   - `spatial_lr_test_0011` (`seed=42`): patching flipped the purple umbrella back to the left, but the right-hand apple turned red instead of green.
3. **Small sample size on `1024-MS`:** The `1024-MS` confirmation tested 3 pairs x 3 seeds (`N=7` valid) at `L10-11` and `L2-3` rather than a full 14-layer-group x 5-window sweep.
4. **Prompt-to-Prompt competitiveness:** P2P (`A_clean` injection) beat patching on the `N=6` pilot (`0.8261` vs `0.7104`) and matched it on the `N=12` test split (`0.6702` vs `0.6903`).
5. **Partial E3 exploration outside `W1[4-7]`:** Because E3 tested `L10-11` and `L2-3` at `W1[4-7]` (`tau_color = 0.3535`, `flip = 0.3333`), it remains open whether later layers (`L12-27`) or `W2[8-11]` carry additional color-binding capacity beyond `L11H01`.

## 4. Week 4 Go/No-Go Decision Note

**Decision: GO (Advance to E4 held-out-combination & terse-template generalization, plus E5 compositional repair/steering).**

**Evidence basis:**
1. Both opening operations passed all pilot and full-sweep causal gates on object-disjoint `test` splits with tight controls (`random_head <= 0.014`, `wrong_timestep <= 0.005`, `same_prompt_noop = 0.000`).
2. Spatial relation transfer replicated across resolutions (`512-MS` `0.6903` and `1024-MS` `0.4901`).
3. We uncovered concrete, falsifiable mechanistic structure: a cross-attention-to-self-attention handoff (`L3H2` -> `L10-11` -> `L6-7 attn1_out = 0.9424`), head-level double dissociation between spatial (`L3H02`, `L10H15`) and color binding (`L11H01`), and a clear 7-baseline separation showing why observational metrics (`Zarei`, `Maliha`, `Revelio` probe) miss the circuit while `P2P` (`rho = 0.483`) and `DAAM` (`rho = 0.370`) partially align.

**Targeted priorities for E4/E5 based on these limitations:**
- In E4, test OOD objects (`ood` split) AND test whether injecting `L3H02` + `L10-11` / `L6-7 attn1_out` rescues terse preposition-insensitive `"left of / right of"` failures from E0 (`p1` vs `p3`) and multi-object E0 failures (`p4`, `p5`).
- Check a full `L0-27 x W0-4` scan for E3 color binding (or `attn1_out` / `ff_out` for color) to test whether additional color-binding capacity sits outside `L10-11 @ W1[4-7]`.
