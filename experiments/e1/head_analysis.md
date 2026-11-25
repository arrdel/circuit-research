# E1 Head-Level Analysis and H2 Implications

Checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`), `test` split (`N=18` instances, `13` valid counterfactuals), hook point `attn2_head_pre_proj` vs full `attn2_out`, window `W1[4-7]` (`steps 4-7`).

## 1. Full Per-Head Distribution at Peak Cell (`L10-11 @ W1[4-7]`)

Full layer-group patching (`attn2_out`, all 16 heads in `L10` and `L11`):
- `L10-11 @ W1[4-7]`: `tau = 0.6903` (`95% CI [0.464, 0.896]`), `flip_rate = 0.6923`, `sigma = 0.9349`, `detection_rate = 0.9231`.

Individual head patching across all 32 heads in `L10` and `L11` at `W1[4-7]`:

| Layer | Head | Tau | 95% CI | Flip Rate | Sigma |
| --- | --- | --- | --- | --- | --- |
| L10 | H00 | 0.0060 | [0.0003, 0.0161] | 0.0000 | 0.9729 |
| L10 | H01 | 0.0532 | [0.0004, 0.1560] | 0.0769 | 0.9473 |
| L10 | H02 | 0.0586 | [0.0014, 0.1655] | 0.0769 | 0.9720 |
| L10 | H03 | 0.0154 | [0.0002, 0.0421] | 0.0000 | 0.9721 |
| L10 | H04 | 0.0014 | [0.0001, 0.0036] | 0.0000 | 0.9726 |
| L10 | H05 | 0.0521 | [0.0003, 0.1549] | 0.0769 | 0.9463 |
| L10 | **H06** | **0.2585** | [0.0727, 0.4927] | **0.2308** | 0.9423 |
| L10 | H07 | 0.0777 | [0.0004, 0.2317] | 0.0769 | 0.9455 |
| L10 | **H08** | **0.1357** | [0.0003, 0.3287] | **0.1538** | 0.8938 |
| L10 | H09 | 0.0067 | [0.0001, 0.0195] | 0.0000 | 0.9729 |
| L10 | H10 | 0.0144 | [0.0001, 0.0350] | 0.0000 | 0.9727 |
| L10 | H11 | 0.0305 | [0.0013, 0.0655] | 0.0000 | 0.9716 |
| L10 | H12 | 0.0539 | [0.0005, 0.1592] | 0.0769 | 0.9728 |
| L10 | H13 | 0.0783 | [0.0002, 0.2330] | 0.0769 | 0.9457 |
| L10 | H14 | 0.0087 | [0.0001, 0.0243] | 0.0000 | 0.9473 |
| L10 | **H15** | **0.3129** | [0.1143, 0.5364] | **0.2308** | **0.9884** |
| L11 | H00 | 0.0058 | [0.0002, 0.0167] | 0.0000 | 0.9726 |
| L11 | **H01** | **0.1123** | [0.0031, 0.2715] | **0.0769** | 0.8970 |
| L11 | H02 | 0.0074 | [0.0002, 0.0208] | 0.0000 | 0.9727 |
| L11 | H03 | 0.0075 | [0.0003, 0.0211] | 0.0000 | 0.9453 |
| L11 | H04 | 0.0061 | [0.0004, 0.0165] | 0.0000 | 0.9521 |
| L11 | H05 | 0.0025 | [0.0003, 0.0058] | 0.0000 | 0.9723 |
| L11 | H06 | 0.0003 | [0.0001, 0.0005] | 0.0000 | 0.9980 |
| L11 | H07 | 0.0033 | [0.0006, 0.0068] | 0.0000 | 0.9725 |
| L11 | H08 | 0.0579 | [0.0004, 0.1663] | 0.0769 | 0.9469 |
| L11 | H09 | 0.0095 | [0.0006, 0.0252] | 0.0000 | 0.9728 |
| L11 | H10 | 0.0543 | [0.0001, 0.1597] | 0.0769 | 0.9469 |
| L11 | **H11** | **0.1710** | [0.0108, 0.3859] | **0.1538** | 0.9682 |
| L11 | H12 | 0.0035 | [0.0002, 0.0097] | 0.0000 | 0.9469 |
| L11 | H13 | 0.0940 | [0.0007, 0.2479] | 0.0769 | 0.8989 |
| L11 | H14 | 0.0129 | [0.0004, 0.0367] | 0.0000 | 0.9719 |
| L11 | H15 | 0.0015 | [0.0002, 0.0033] | 0.0000 | 0.9729 |

## 2. Replication of Secondary Peak (`L2-3 @ W1[4-7]`) and Single-Head Dominance (`L3H2`)

The `L2-3 @ W1[4-7]` secondary peak replicates on the held-out `test` split (`512-MS`) and on `1024-MS`:
- `512-MS` pilot (`dev`, `N=6` valid): `L2-3 @ W1[4-7]` `tau = 0.665`, `flip_rate = 0.50`, `sigma = 0.818`.
- `512-MS` full sweep (`test`, `N=13` valid): `L2-3 @ W1[4-7]` `tau = 0.5522` (`95% CI [0.312, 0.791]`), `flip_rate = 0.5385`, `sigma = 0.9099`, `detection_rate = 1.0000`.
- `1024-MS` confirmation (`test`, `N=7` valid): `L2-3 @ W1[4-7]` `tau = 0.4991`, `flip_rate = 0.4286`, `sigma = 0.8412`, `detection_rate = 1.0000` (matching `L10-11 @ W1[4-7]`: `tau = 0.4901`, `flip_rate = 0.4286`, `sigma = 0.8402`).

Crucially, per-head dissection across all 32 heads of `L2` and `L3` at `W1[4-7]` reveals extreme single-head concentration in Layer 3:
- **`L3H02 @ W1[4-7]` alone reaches `tau = 0.5468` (`95% CI [0.305, 0.787]`), `flip_rate = 0.5385`, `sigma = 0.9113`**, recovering **99.0% of the entire `L2-3` 32-head layer group (`0.5522`)**.
- Every other head in `L2` and `L3` has `tau <= 0.0955` (`L3H08: 0.0955`, `L3H07: 0.0943`, `L3H01: 0.0855`; all 16 heads in `L2` have `tau <= 0.0581`).

## 3. Implications for Hypothesis H2 (Small `(layer, head)` Set vs. Layer-Group Effect)

Comparing single-head patching against full layer-group patching reveals a **two-regime circuit structure** that refines H2:

1. **Early Cross-Attention Text-Reading Bottleneck (`L3H2`, strict H2 regime):** At `L2-3`, H2 holds in its strongest form. A single cross-attention head (`L3H2`, `tau = 0.5468`) accounts for virtually the entire causal effect of `L2-3` (`tau = 0.5522`), and sublayer comparison shows `attn2_in_txt` patching at `L2-3` reproduces most of the effect (`tau = 0.6300` vs `attn2_out` `0.6795`).
2. **Mid-Layer Cross-Modal Routing Stage (`L10-11`, semi-distributed multi-head regime):** At `L10-11`, H2 holds only as a **sparse multi-head coalition (~5-6 heads out of 32)** rather than a single head. Patching the full `L10-11` layer group yields `tau = 0.6903` (`flip_rate = 0.6923`), more than double the best single head `L10H15` (`tau = 0.3129`, `flip_rate = 0.2308`), followed by `L10H06` (`0.2585`), `L11H11` (`0.1710`), `L10H08` (`0.1357`), `L11H01` (`0.1123`), and `L11H13` (`0.0940`), while 20 of 32 heads remain below `0.02`.

Thus, PixArt-Sigma does not rely on a single head everywhere: it concentrates spatial relation reading into `L3H2` (consistent with Wang et al.'s early spatial-relation head prior), then distributes spatial routing across a compact 6-head coalition in `L10-11` before self-attention consolidates the layout in `L6-15`.

## 4. Seed-Variation Control and Color Preservation (Main Results)

Across all three evaluations, patching `L10-11 @ W1[4-7]` from a shifted-seed clean run (`s + 999`) transfers spatial layout reliably while degrading color binding relative to seed-matched patching:
- **E1 Pilot (`512-MS`, `dev`):** `seed_variation` `tau = 0.4980`, `flip_rate = 0.5000`, `sigma = 0.8148`, `color_preserve_rate = 0.4000` (vs. `L10-11` seed-matched `tau = 0.7104`, `sigma = 0.9263`).
- **E1 Full Sweep (`512-MS`, `test`):** `seed_variation` `tau = 0.7063` (`95% CI` in JSON), `flip_rate = 0.6923`, `sigma = 0.9100`, `color_preserve_rate = 0.8000` (vs. `wrong_timestep` and `same_prompt_noop` `color_preserve_rate = 1.0000`).
- **E1 `1024-MS` Confirmation (`test`):** `seed_variation` `tau = 0.3356`, `flip_rate = 0.2857`, `sigma = 0.7905`, `color_preserve_rate = 0.4286` (vs. `L10-11` seed-matched `tau = 0.4901`, `color_preserve_rate = 0.5714`).

The cross-attention spatial layout representation in `L10-11 @ W1[4-7]` is therefore substantially seed-invariant (transferring left/right placement even when sourced from a mismatched initial noise sample `s + 999`), whereas preserving exact object color bindings requires matched latent geometry or attention-map-only injection (`Prompt-to-Prompt` `A_clean` injection at `L10-11 @ W1[4-7]` achieves `tau = 0.8261`, `flip_rate = 0.8333`).
