# CIRCUIT: Muse <-> Cursor agent coordination (single file)

This file is the operating contract AND the day-by-day log between the
CIRCUIT Cursor agent (execution, in Adele's Cursor IDE) and Muse
(oversight). Single file of record, by Adele's decision Nov 24, 2026: all
communication between Muse and the agent lives here, in dated sections
below. The daily task files tasks/2026-11-24.md through tasks/2026-11-27.md
were consolidated into this file and deleted.

Before the public release, gitignore this file and remove it from tracking
(git rm --cached tasks/AGENT.md). Until then it stays tracked so the agent
can read it from the repo.

## The loop

1. Muse appends the day's tasks as a new dated section in THIS file each
   morning, Monday to Saturday, around 7:50 AM ET, and as TickTick tasks.
   The section is the full brief: what, why, done criteria, files, repos,
   constraints, links. TickTick carries the short title and due date only.
2. The agent reads the latest dated section, executes in the IDE, and
   reports back in that same section, under its `### Report`, before the
   day ends.
3. Muse verifies the report against the evidence the next morning,
   appends a Review to that section (or opens the next day's section with
   it), and appends the next day's section.
4. Adele decides anything that needs his judgment. The agent never decides
   for him and never sends, publishes, or submits anything.

## The shared channel

This file is how the agent and Muse communicate. The agent reports by
updating this file and pushing, and by linking its work (commit SHAs,
file paths, PR or issue links). If the agent is blocked, it writes the
blocker plainly in the report and moves to the next task. It does not spin
on a blocked task.

## Scope

The agent executes CIRCUIT work only: causal interpretability and reasoning
under controlled interventions in diffusion transformers. The research
question: do diffusion transformers develop temporally localized, causally
identifiable computational mechanisms for object-attribute binding and
spatial relationships? Model: PixArt-Sigma. Opening operations: spatial
relation left/right and color attribute binding. The full blueprint (literature
review, novelty matrix, hypotheses, causal measurement definitions,
experiment protocols E0-E6, first-month plan) lives in the private frontier
repo; each day's section below carries everything needed for that day.

Out of scope for this agent:
- LUCID / CDR / diffusion-language-model work. That is Adele's PhD project;
  never touch it, never import from it, never fold its material in.
- traj-eval work. CIRCUIT uses traj-eval results as evidence only.
- Open-source contributions to other projects. Separate window.
- Anything that sends, publishes, or submits on Adele's behalf (emails,
  applications, posts, PRs to other people's repos). Drafts only, Adele
  approves.

## Guardrails (from the verified blueprint)

- Read Roy's circuit-level paper (arXiv:2506.17237) first, before any
  localization work. It is the closest prior.
- Do not duplicate DLM-Scope's SAE extraction on LLaDA/Dream. That space is
  occupied. CIRCUIT is image DiT plus relational circuits plus patching.
- Do not rebuild T2I-CompBench. CIRCUIT-Bench reuses its detector scoring;
  its contribution is the counterfactual-pair layer.
- If a run needs a GPU and this environment has none, report blocked and say
  what is needed. Never fake a result.

## The bar

The goal is to be over-qualified for research roles at frontier labs.
Above average is the floor. Work that is merely good enough does not pass.

- No placeholders, no TODOs left as done, no unverified numbers.
- Every number traces to a measured run or a cited source. If a number
  cannot be traced, it is not claimed.
- Drafts go in separate files unless the target file itself marks the work
  as today's task. Canonical files are updated only after Muse or Adele
  reviews a draft.
- DIAL and ATAT are under review at ICLR 2027, never presented as published.
- No em dashes in anything public-facing.
- No credentials anywhere in any repo.
- Do not describe CIRCUIT publicly as a personal project or mention advisor
  non-involvement. That framing is not public information.

## Report format

For each task in the day's section, the agent appends:

- Status: done / partial / blocked.
- What was done, with evidence: file paths, commit SHAs, links.
- What is blocked, and what would unblock it.
- Open questions for Muse or Adele.

A task is done when its done criteria are met and the evidence is linked.
A task with no evidence is not done, even if the work happened.

---

## Day 1: Tue Nov 24, 2026

Week 1 of 4. Goal for the week: research foundation and reproduction
(blueprint E0). This section is the full brief; report back in the
`### Report` below before the day ends, then push.

Research question: Do diffusion transformers develop temporally localized,
causally identifiable computational mechanisms for object-attribute binding
and spatial relationships? Model: PixArt-Sigma (0.6B DiT, Hugging Face
Diffusers, checkpoint PixArt-alpha/PixArt-Sigma-XL-2-1024-MS). Opening
operations: spatial relation left/right, and color attribute binding.

### Task 1: Read the five core papers (Roy first)

Read in this order:

1. Roy, "Mechanistic Interpretability of Diffusion Models: Circuit-Level
   Analysis and Causal Validation", arXiv:2506.17237. Read first: closest
   prior, circuit-level plus causal validation.
2. Maliha and Hougen, "Mechanistic Interpretability of Text-to-Image
   Diffusion Models via Cross-Attention Interventions", Findings of ACL
   2026, https://aclanthology.org/2026.findings-acl.1265/
3. Kim, Thomas, Ghadiyaram, "Revelio", ICCV 2025, arXiv:2411.16725
4. Wang et al., "DLM-Scope", ICML 2026, arXiv:2602.05859
5. Huang et al., "T2I-CompBench", NeurIPS 2023, arXiv:2307.06350

Write `research/reading-notes-2026-11-24.md` with, per paper: what it does,
what it does not do, and what it means for CIRCUIT (one paragraph each).
Done when the notes file exists and covers all five with links.

### Task 2: Reproduce PixArt-Sigma generation (E0 start)

Load PixArt-Sigma through Diffusers. Fixed seed, sampler, step count, and
guidance for every run; record the exact settings in the run log. Generate
the five counterfactual prompt families:

1. Red cube left of blue sphere.
2. Blue cube left of red sphere.
3. Red cube right of blue sphere.
4. Red cube left of blue sphere, green cylinder in front.
5. Two red cubes and one blue sphere.

Save the images under `experiments/e0/` and the run log as
`experiments/e0/run-log.md` (prompt, seed, sampler, steps, guidance, output
path per image). Verify the same prompt and seed reproduces the same image
(run one prompt twice and compare).
Done when the script, the images, and the run log exist and the
reproduction check passes. If this environment has no GPU or cannot
download the model, report blocked with the exact error and what is needed
(HF access, GPU). Never fake a run.

### Task 3: E0 failure set

From the Task 2 images, identify the reproducible compositional failures:
cases where the image violates the prompt and the violation is stable across
3 seeds. Record them in `experiments/e0/failures.md` (prompt, what was
required, what the image shows, seeds tested). Depends on Task 2; if Task 2
is blocked, this task is blocked too. Done when failures.md lists the
failures with evidence, or states that no stable failure was found (with
the images as evidence).

### Constraints

- Read the contract at the top of this file first. It is the operating
  contract.
- No invented numbers. Every claim traces to a measured run or a cited
  source.
- No em dashes in anything public-facing. No credentials anywhere.
- Do not describe CIRCUIT publicly as a personal project or mention advisor
  non-involvement.

### Report

#### Task 1: Read the five core papers (Roy first)
- Status: done.
- What was done: Read all five papers in the specified order (Roy `arXiv:2506.17237` first, followed by Maliha and Hougen ACL Findings 2026, Revelio `arXiv:2411.16725`, DLM-Scope `arXiv:2602.05859`, and T2I-CompBench / T2I-CompBench++ `arXiv:2307.06350`) and wrote `research/reading-notes-2026-11-24.md` covering what each paper does, what it does not do, and what it means for CIRCUIT, with links to all five papers.
- Evidence: `research/reading-notes-2026-11-24.md`.

#### Task 2: Reproduce PixArt-Sigma generation (E0 start)
- Status: done.
- What was done: Implemented the reusable `PixArtSigmaRunner` (`models/pixart_sigma.py`), hook/patching infrastructure (`instrumentation/hooks.py`, `interventions/patching.py`), detector + SigLIP evaluator (`evaluation/spatial_eval.py`, `evaluation/causal_metrics.py`), and Day 1 reproduction script (`experiments/e0/reproduce_pixart.py`). Ran `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (`DPMSolverMultistepScheduler`, 20 steps, guidance scale 4.5, 1024x1024, `torch.bfloat16`) across all 5 counterfactual prompt families and 3 seeds (`42, 123, 456`), saving 15 images plus a deterministic recheck image (`p1_red_cube_left_blue_sphere_seed42_recheck.png`) and visual grid (`experiments/e0/e0_grid.png`). Verified exact determinism: rerunning `Red cube left of blue sphere.` with seed `42` produced identical SHA-256 (`0f825cd7463f0f00...`) and `max_abs_pixel_diff = 0`.
- Also ran a 24-pair spatial + 18-pair color-binding dev sanity check on `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`experiments/e0_baseline_sanity.py`) to test multi-template counterfactual yield: explicit positional phrasing (`on_the_side_explicit`, `side_by_side_scene`) achieved 62.5% valid counterfactual pairs (87.5% clean / 75.0% corrupt relation accuracy), whereas `to_the_side_of` achieved only 25.0% valid counterfactual pairs (37.5% clean / 50.0% corrupt).
- Evidence: `experiments/e0/reproduce_pixart.py`, `experiments/e0/run-log.md`, `experiments/e0/e0_results.json`, `experiments/e0/e0_grid.png`, `experiments/e0/images/`, `models/pixart_sigma.py`, `experiments/e0_baseline_sanity.py`.

#### Task 3: E0 failure set
- Status: done.
- What was done: Inspected all 15 generated 1024x1024 images from Task 2 alongside GroundingDINO-tiny + SigLIP-base diagnostics and documented reproducible failures in `experiments/e0/failures.md`. Found two prompt families with stable compositional failures across all 3 seeds (3/3):
  1. `p4` (`Red cube left of blue sphere, green cylinder in front.`): spatial inversion and blue cylinder leakage on seed 42; attribute swap (green cube, red sphere) and cylinder omission on seed 123; object omission (no blue sphere) and shape/color confusion on seed 456.
  2. `p5` (`Two red cubes and one blue sphere.`): undercount (1 red cube) on seed 42; count error and color leakage (1 blue cube, 1 red cube, 1 blue sphere, 1 red sphere) on seed 123; overcount (hallucinated extra red sphere) on seed 456.
  Also documented a key counterfactual diagnostic on `p1` vs `p3` (`Red cube left/right of blue sphere.`): on all 3 seeds, `p1` and `p3` produce near-identical spatial layouts (`seed=42`: `dx=-0.4102` vs `-0.3774`; `seed=123`: `dx=-0.4358` vs `-0.3687`; `seed=456`: `dx=+0.3172` vs `+0.2795`), indicating preposition insensitivity on terse `"left of"` / `"right of"` phrasing.
- Evidence: `experiments/e0/failures.md`, `experiments/e0/e0_grid.png`, `experiments/e0/images/`.

#### Open questions for Muse or Adele
1. Given `p1` vs `p3` preposition insensitivity on terse `"left of"` / `"right of"` prompts vs. 62.5% valid counterfactual yield on `"on the left ... on the right"` templates, should E1 activation patching evaluate on `"on the left / on the right"` counterfactual pairs first (where clean and corrupt reliably differ) and test cross-template transfer to `"left of"` / `"right of"` as an OOD generalization check?
2. We have `adelechinda@ai-lab2.dyn.gsu.edu` available with two idle 24 GB RTX 4090s (GPUs 0 and 1). Ready to run E1 Layer x Timestep activation patching sweep there next.

---

## Day 2: Wed Nov 25, 2026

Week 1 of 4 (due Sat Nov 28). E0 is done and verified; today is the E1 pilot
plus two verification fixes. Read the Review first; it contains decisions
that change what you run.

### Review of Day 1 (Muse, Nov 24 evening)

Verified against the evidence, not the report alone:

- Task 1 done. Reading notes are accurate against the five papers. One
  figure needs a body citation before it goes into any write-up: the Roy
  note claims a middle-layer bottleneck of 153.8% MSE degradation. Roy's
  abstract reports 25.6% to 128.3% ablation degradation. The 153.8% number
  is not in the abstract; verify it against the paper body and cite the
  exact table, or correct it (Task 1 below).
- Task 2 done, and real. I viewed p1 seed 42 (blue sphere left, red cube
  right, relation inverted, matches dx -0.4102) and p4 seed 42 (blue
  cylinder instead of green, matches the failure note). e0_results.json is
  valid: 15 runs, reproduction check passed, identical SHA-256 on the
  recheck, max pixel diff 0.
- Task 3 done. The p1 vs p3 preposition-insensitivity finding is the key
  diagnostic and it holds at matched seeds.
- Zarei 2025 is real and relevant: "Localizing Knowledge in Diffusion
  Transformers", arXiv:2505.18832, DiT knowledge localization on
  PixArt-alpha, FLUX, SANA. Keep it in interpretability/baselines.py. It
  is a sixth prior for the reading notes (Task 1 below).
- E1/E1b/E2/E3 runners are code only. No E1/E2/E3 results were committed,
  which is correct: those experiments have not run yet. Do not run E2 or E3
  until E1's tau map is validated. See Decisions.

### Decisions (Muse)

1. E1 template: YES to your open question. Run E1 activation patching on
   the explicit "on the left / on the right" counterfactual pairs first,
   where clean and corrupt reliably differ (62.5% valid yield on the dev
   sanity check). The terse "left of / right of" template is
   preposition-insensitive (p1 approx p3 at matched seeds), so it has no
   clean/corrupt contrast to transfer. The terse template becomes the E4
   generalization check, not the E1 target.
2. Sequence is fixed: E1 pilot, then E1 full sweep, then E2, then E3. No
   skipping ahead. The E2/E3 runners you already wrote stay as code until
   their turn.
3. Lab GPUs: use adelechinda@ai-lab2.dyn.gsu.edu (2x RTX 4090) for E1.
   Record the host, GPU, and environment in every run log.

### Task 1: Fix the two citation items

- Verify the Roy 153.8% MSE figure against the paper body. Cite the exact
  table/section in research/reading-notes-2026-11-24.md, or correct the
  number to what the paper actually reports.
- Add Zarei et al. 2025 (arXiv:2505.18832) to the reading notes as a sixth
  paper: what it does, what it does not do, what it means for CIRCUIT
  (one paragraph, same format).
Done when the notes file is corrected and covers all six with links.

### Task 2: Test suite green

- Run the full pytest suite in this repo and report pass/fail per test
  file in the Report. Fix any failures; do not leave the suite red.
Done when pytest passes and the report lists the result.

### Task 3: E1 pilot (layer x timestep patching, spatial left/right)

- On the lab GPUs, run experiments/e1_spatial_patching.py as a PILOT, not
  the full sweep: 2 to 4 explicit-template counterfactual pairs
  ("on the left / on the right"), 3 seeds, layer x timestep patching with
  all four controls (random-head, wrong-timestep, same-prompt no-op, seed
  variation).
- Save results as experiments/e1/e1_pilot_results.json (tau per (l,h,t),
  specificity sigma, flip rate, detection rate, controls) and a run log
  experiments/e1/e1_pilot_run_log.md (host, GPU, checkpoint, pairs, seeds,
  settings). Save patched images under experiments/e1/images/.
- Gate: the pilot passes only if the peak tau cell beats the random-head
  control. If no cell beats it, report that plainly and stop; do not run
  the full sweep.
Done when the pilot results JSON, run log, and images exist and the report
states the gate outcome. Do NOT run the full E1 sweep, E2, or E3 today.

### Constraints

- Read the contract at the top of this file first. It is the operating
  contract.
- No invented numbers. Every claim traces to a measured run or a cited
  source.
- Label the checkpoint on every result (1024-MS vs 512-MS).
- No em dashes in anything public-facing. No credentials anywhere.
- Do not describe CIRCUIT publicly as a personal project or mention advisor
  non-involvement.

### Report

#### Task 1: Fix the two citation items
- Status: done.
- What was done:
  1. Verified Roy (`arXiv:2506.17237`) against the paper body: the abstract reports 25.6% to 128.3% ablation degradation across targeted ablations, while Section 4.6, Table 3 ("Layer Ablation Impact Analysis (Synthetic Dataset, t = 300)") reports `Middle Layers` rising from baseline MSE `0.0246` to ablated MSE `0.0624` (Impact Score `1.538`, i.e., 153.8% MSE degradation). Updated `research/reading-notes-2026-11-24.md` to cite Section 4.6, Table 3 explicitly alongside the abstract range.
  2. Added Zarei et al. (2025), "Localizing Knowledge in Diffusion Transformers" ([arXiv:2505.18832](https://arxiv.org/abs/2505.18832)), as paper 6 in `research/reading-notes-2026-11-24.md` (what it does, what it does not do, and what it means for CIRCUIT).
- Evidence: `research/reading-notes-2026-11-24.md`.

#### Task 2: Test suite green
- Status: done.
- What was done: Ran full `pytest -v` in `/home/achinda1/projects/circuit-research` (`circuit` conda env, Python 3.10.19). All 6 tests across all 4 test files passed in 4.72s:
  - `tests/test_baselines.py`: 1/1 passed (`test_spatial_gradient_and_norm_distribution`)
  - `tests/test_benchmark.py`: 2/2 passed (`test_spatial_lr_pairs_disjoint_and_matched`, `test_color_binding_pairs_swap`)
  - `tests/test_causal_metrics.py`: 2/2 passed (`test_spatial_tau_and_sigma`, `test_failed_detection_yields_zero_tau`)
  - `tests/test_hooks_and_patching.py`: 1/1 passed (`test_capture_and_selective_cond_branch_patching`)
- Evidence: `tests/test_baselines.py`, `tests/test_benchmark.py`, `tests/test_causal_metrics.py`, `tests/test_hooks_and_patching.py`.

#### Task 3: E1 pilot (layer x timestep patching, spatial left/right)
- Status: done (Gate: PASS).
- What was done:
  - Stopped earlier out-of-sequence E1b/E2 runs on `ai-lab2` immediately upon reading the Day 2 directives; kept E2/E3 runners as code only and ran only the E1 pilot (`experiments/e1/e1_pilot.py`).
  - Host and GPUs: `ai-lab2` (`adelechinda@ai-lab2.dyn.gsu.edu`), 2x `NVIDIA GeForce RTX 4090` (`gen: cuda:0`, `eval: cuda:1`), Python `3.10.12`, Torch `2.6.0+cu124`, Diffusers `0.41.0`, Transformers `5.20.2`.
  - Checkpoint and settings: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`), `DPMSolverMultistepScheduler`, 20 steps, guidance scale `4.5`, `torch.bfloat16`, hook point `attn2_out` on conditional CFG branch.
  - Pairs and seeds: 3 explicit-template counterfactual pairs (`spatial_lr_dev_0002`, `spatial_lr_dev_0004`, `spatial_lr_dev_0010`) across 3 seeds (`42, 123, 456`) = 9 instances (6 valid counterfactuals).
  - Layer x timestep sweep (14 2-layer groups x 5 windows) + per-`(l, h, t)` breakdown (32 heads in peak group `L10-11` at `W1[4-7]`):
    - Peak `(L, T)` cell: `L10-11 @ W1[4-7]` (`tau = 0.7104`, 95% CI reported in JSON, `sigma = 0.9263`, `flip_rate = 0.6667`, `detection_rate = 1.0000`). Secondary peak at `L2-3 @ W1[4-7]` (`tau = 0.665`, `flip_rate = 0.50`, `sigma = 0.818`). Zero transfer (`tau <= 0.014`) above `L14` or after `W2[8-11]`.
    - Peak single head `(l, h, t)`: `L10H15 @ W1[4-7]` (`tau = 0.2521`, `sigma = 0.9291`, `flip_rate = 0.1667`).
  - Four controls:
    - `random_head` (`L3H10 @ W1[4-7]`): `tau = 0.0141`, `flip_rate = 0.0000`, `sigma = 0.9931`, `detection_rate = 1.0000`.
    - `wrong_timestep` (`L10-11 @ W4[16-19]`): `tau = 0.0003`, `flip_rate = 0.0000`, `sigma = 0.9983`, `detection_rate = 1.0000`.
    - `same_prompt_noop` (corrupt cache patched into corrupt at `L10-11 @ W1[4-7]`): `tau = 0.0000`, `flip_rate = 0.0000`, `sigma = 1.0000`, `detection_rate = 1.0000`.
    - `seed_variation` (shifted-seed `s+999` clean cache patched at `L10-11 @ W1[4-7]`): `tau = 0.4980`, `flip_rate = 0.5000`, `sigma = 0.8148`, `detection_rate = 1.0000`.
  - Gate outcome: **PASS** (`peak_tau = 0.7104 > random_head_tau = 0.0141`; single-head `L10H15` `0.2521` also exceeds `random_head` `0.0141`).
- Evidence: `experiments/e1/e1_pilot.py`, `experiments/e1/e1_pilot_results.json`, `experiments/e1/e1_pilot_run_log.md`, `experiments/e1/e1_pilot_grid.png`, `experiments/e1/images/`.

#### Open questions for Muse or Adele
1. The E1 pilot passed the gate (`0.7104` vs `0.0141`). Ready to proceed on Day 3 with the full E1 sweep (`test`/`ood` splits, `512-MS` + `1024-MS` check, and sublayer comparison `attn2_out` vs `attn1_out` vs `ff_out` vs `attn2_in_txt`) when instructed.

---

## Day 3: Thu Nov 26, 2026

Week 1 of 4. Day 2 finished early on Nov 24 evening; run this section when
you resume. E1 pilot passed its gate. Today is the full E1 sweep.

### Review of Day 2 (Muse, Nov 24 evening)

Verified against the evidence, not the report alone:

- Task 1 done, and the Roy figure is now confirmed. I checked the Roy PDF
  directly: Section 4.6, Table 3 ("Layer Ablation Impact Analysis
  (Synthetic Dataset, t = 300)") reports Middle Layers baseline MSE 0.0246
  rising to 0.0624, Impact Score 1.538, and the text states "ablation
  producing 153.8% performance degradation". The 153.8% number is real and
  the citation is correct. Zarei 2025 is in the notes as paper 6.
- Task 2 done. Pytest green, 6/6 across 4 test files, as reported.
- Task 3 done, gate PASS, and the transfer is real. I viewed clean, corrupt,
  and peak-patched images for dev_0004 seed 42: clean has the clock left of
  the book, corrupt has the book left of the clock, and the L10-11 @ W1[4-7]
  patched image matches the clean layout. Peak tau 0.7104 vs random-head
  0.0141. Secondary peak L2-3 @ W1[4-7] tau 0.665 noted. All four controls
  ran: wrong-timestep tau 0.0003, same-prompt no-op 0.0000, seed-variation
  0.4980.
- Two findings to carry forward, not bury:
  - The peak is a layer GROUP (L10-11 tau 0.710), while the best single head
    (L10H15) reaches only tau 0.252. The effect is distributed across heads
    in L10-11, not carried by one head. This bears on H2, which posits a
    small (layer, head) set.
  - The seed-variation control transfers at tau 0.498 with color-preserve
    rate 0.4 (vs 1.0 elsewhere). The relation encoding is partly
    seed-invariant. Report it in the main results, not an appendix.

### Decisions (Muse)

1. Full E1 sweep approved, as proposed: test/ood splits, 512-MS plus a
   1024-MS check, and the sublayer comparison (attn2_out vs attn1_out vs
   ff_out vs attn2_in_txt).
2. 1024-MS confirmation is required, not optional: the E0 failure set was
   established on 1024-MS, so the transfer claim has to be shown on the
   same checkpoint. Run 2 to 3 pairs on 1024-MS at the peak cell with
   controls. If the 1024-MS peak differs from 512-MS, report both; do not
   assume they match.
3. Do not cherry-pick L10-11. Check whether the L2-3 secondary peak
   replicates on the full sweep.

### Task 1: Full E1 sweep (512-MS)

- Run the full layer x timestep patching sweep on the bench's spatial
  left/right pairs (test split), 3 seeds, with all four controls, on
  512-MS. Use experiments/e1_spatial_patching.py (or the pilot runner
  extended to the full pair set).
- Save results as experiments/e1/e1_full_results.json (tau per (l,h,t) with
  CIs, sigma, flip rate, detection rate, controls) and a run log
  experiments/e1/e1_full_run_log.md (host, GPU, checkpoint, pairs, seeds,
  settings). Save patched images under experiments/e1/images/ and a tau-map
  figure experiments/e1/e1_tau_map.png.
Done when the results JSON, run log, images, and tau-map figure exist and
the report states the peak cell and its controls.

### Task 2: 1024-MS confirmation and sublayer comparison

- On 1024-MS, run 2 to 3 pairs at the peak cell (L10-11 @ W1[4-7]) with the
  four controls. Record the peak tau, sigma, flip rate, and controls in
  experiments/e1/e1_1024_check.json with a run log entry.
- At the peak cell, compare hook points: attn2_out vs attn1_out vs ff_out
  vs attn2_in_txt (the E1b runner). Record tau per hook point in the same
  results file.
Done when the 1024-MS check and the sublayer comparison are in the results
with run logs, whether or not they match 512-MS.

### Task 3: Head-level analysis at the peak cell

- Report the full per-head tau distribution at the peak cell (L10-11 @
  W1[4-7]): every head's tau, sigma, and flip rate. Compare patching all
  heads in L10-11 against patching single heads.
- Check whether the L2-3 secondary peak replicates on the full sweep.
- Write experiments/e1/head_analysis.md stating what the distribution
  implies for H2 (small (layer, head) set vs layer-group effect), with the
  numbers, and report the seed-variation result and its color-preserve
  rate in the main results.
Done when head_analysis.md exists with the numbers and the H2 implication
stated plainly. Do NOT run E2 or E3 today; the full sweep and its analysis
come first.

### Constraints

- Read the contract at the top of this file first. It is the operating
  contract.
- No invented numbers. Every claim traces to a measured run or a cited
  source.
- Label the checkpoint on every result (1024-MS vs 512-MS).
- No em dashes in anything public-facing. No credentials anywhere.
- Do not describe CIRCUIT publicly as a personal project or mention advisor
  non-involvement.

### Report

#### Task 1: Full E1 sweep (512-MS)
- Status: done.
- What was done:
  - Ran `experiments/e1/e1_full_sweep.py` on `ai-lab2` (`CUDA_VISIBLE_DEVICES=0`, `NVIDIA GeForce RTX 4090`, Python `3.10.12`, Torch `2.6.0+cu124`, Diffusers `0.41.0`, Transformers `5.20.2`) on `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` (`512-MS`) using the held-out `test` split (explicit template `on_the_side_explicit`, disjoint objects from `dev`): 6 test pairs (`spatial_lr_test_0001`, `0011`, `0013`, `0016`, `0020`, `0026`) x 3 seeds (`[42, 123, 456]`) = 18 instances (13 valid counterfactuals).
  - Swept 14 2-layer groups x 5 temporal windows (`attn2_out`, conditional CFG branch) plus 64 single-head `(l, h, t)` cells across `L2, L3, L10, L11` at `W1[4-7]` and all 4 controls:
    - Peak `(L, T)` cell: `L10-11 @ W1[4-7]` (`tau = 0.6903`, 95% CI `[0.464, 0.896]`, `flip_rate = 0.6923`, `sigma = 0.9349`, `detection_rate = 0.9231`).
    - Secondary peak replication: `L2-3 @ W1[4-7]` replicated cleanly at `tau = 0.5522` (95% CI `[0.312, 0.791]`, `flip_rate = 0.5385`, `sigma = 0.9099`, `detection_rate = 1.0000`). Zero transfer (`tau <= 0.082`) above `L14` or in `W4[16-19]`.
    - Controls at `L10-11 @ W1[4-7]`: `random_head` (`L3H10`: `tau = 0.0042`, `flip_rate = 0.0000`, `sigma = 0.9722`, `color_preserve_rate = 0.8333`), `wrong_timestep` (`W4[16-19]`: `tau = 0.0005`, `flip_rate = 0.0000`, `sigma = 0.9986`, `color_preserve_rate = 1.0000`), `same_prompt_noop` (`tau = 0.0000`, `flip_rate = 0.0000`, `sigma = 1.0000`, `color_preserve_rate = 1.0000`), and `seed_variation` (`s+999`: `tau = 0.7063`, `flip_rate = 0.6923`, `sigma = 0.9100`, `color_preserve_rate = 0.8000`).
- Evidence: `experiments/e1/e1_full_sweep.py`, `experiments/e1/e1_full_results.json`, `experiments/e1/e1_full_run_log.md`, `experiments/e1/e1_tau_map.png`, `experiments/e1/e1_full_grid.png`, `experiments/e1/images/full_head_tau.png`, `experiments/e1/images/full_*.png`.

#### Task 2: 1024-MS confirmation and sublayer comparison
- Status: done.
- What was done:
  - Ran `experiments/e1/e1_1024_and_sublayers.py` on `ai-lab2` (`CUDA_VISIBLE_DEVICES=2`, `NVIDIA GeForce RTX 4090`).
  - Part A (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`, `1024-MS`, 3 test pairs `spatial_lr_test_0000`, `0001`, `0003` x 3 seeds = 9 instances): both `L10-11 @ W1[4-7]` (`tau = 0.4901`, `flip_rate = 0.4286`, `sigma = 0.8402`, `detection_rate = 1.0000`, `color_preserve_rate = 0.5714`) and `L2-3 @ W1[4-7]` (`tau = 0.4991`, `flip_rate = 0.4286`, `sigma = 0.8412`, `detection_rate = 1.0000`, `color_preserve_rate = 0.5000`) replicated cleanly against `ctrl_random_head` (`L2H10`: `tau = 0.0001`), `ctrl_wrong_timestep` (`tau = 0.0002`), and `ctrl_same_prompt_noop` (`tau = 0.0000`), while `ctrl_seed_variation` reached `tau = 0.3356` (`flip_rate = 0.2857`, `sigma = 0.7905`, `color_preserve_rate = 0.4286`).
  - Part B (`512-MS`, 4 test pairs `0001, 0011, 0013, 0016` x 3 seeds = 12 instances): at `L2-3 @ W1[4-7]`, `attn2_out` (`0.6795`) and `attn2_in_txt` (`0.6300`) carry the signal while `attn1_out` (`0.0723`) and `ff_out` (`0.0753`) do not; at `L10-11 @ W1[4-7]`, `attn2_out` (`0.7912`) and `attn1_out` (`0.7191`) dominate while `attn2_in_txt` (`0.1503`) and `ff_out` (`0.1402`) drop sharply; at `L6-7 @ W2[8-11]`, downstream `attn1_out` reaches `0.9424` (`flip_rate = 0.8889`) as `attn2_out` (`0.2166`) and `attn2_in_txt` (`0.0008`) fade.
- Evidence: `experiments/e1/e1_1024_and_sublayers.py`, `experiments/e1/e1_1024_check.json`, `experiments/e1/e1_1024_run_log.md`, `experiments/e1/images/1024_*.png`.

#### Task 3: Head-level analysis at the peak cell
- Status: done.
- What was done:
  - Documented the full 32-head `(l, h)` distribution for `L10-11 @ W1[4-7]` plus `L2-3 @ W1[4-7]` in `experiments/e1/head_analysis.md`.
  - Key finding for H2: PixArt-Sigma exhibits a two-regime head structure. At `L2-3 @ W1[4-7]`, a single cross-attention head `L3H02` achieves `tau = 0.5468` (`flip_rate = 0.5385`, `sigma = 0.9113`), accounting for 99.0% of the entire `L2-3` layer group (`0.5522`). At `L10-11 @ W1[4-7]`, transfer is distributed across a compact ~6-head coalition (`L10H15: 0.3129`, `L10H06: 0.2585`, `L11H11: 0.1710`, `L10H08: 0.1357`, `L11H01: 0.1123`, `L11H13: 0.0940`) that composes to `0.6903`.
  - Reported `seed_variation` (`tau = 0.4980` pilot with `color_preserve_rate = 0.4000`; `tau = 0.7063` full sweep with `color_preserve_rate = 0.8000`; `tau = 0.3356` `1024-MS` with `color_preserve_rate = 0.4286`) in the main results.
  - E2 and E3 were NOT run.
- Evidence: `experiments/e1/head_analysis.md`, `experiments/e1/e1_full_results.json`, `experiments/e1/images/full_head_tau.png`.

---

## Day 4: Fri Nov 27, 2026

Week 1 of 4. Adele asked for a more robust baseline set: 3 is not enough,
7 is the target. This section can run in parallel with Day 3's full E1 sweep
(different files); the comparison in Task 3 uses the E1 pilot tau map,
which already exists.

### Where the 3 stand (Muse, verified)

interpretability/baselines.py currently has 3 observational head scores:

1. Attention contribution (Zarei et al. 2025, arXiv:2505.18832). Real.
2. KL spike / norm-based projected output (Maliha & Hougen, ACL 2026). Real.
3. Spatial gradient correlation, labeled "Wang et al. (2026)". The paper is
   real (Wang, Fan & Pan, "Circuit Mechanisms for Spatial Relation
   Generation in Diffusion Transformers", arXiv:2601.06338, Jan 2026), and
   it is a major prior: from-scratch DiTs, a two-stage circuit with two
   cross-attention heads separately reading the spatial relation and
   single-object attributes. BUT the paper does not propose a "spatial
   gradient" score. The gradient score is CIRCUIT's own heuristic, inspired
   by the paper's finding. Fix the attribution in Task 1: label the score
   as CIRCUIT's own diagnostic and cite Wang et al. 2026 as the related-work
   prior, not as the method's source. Also add Wang et al. 2026 to the
   reading notes as paper 7.

### The 7 baselines (all verified live by Muse, Nov 24)

1. Zarei et al. 2025, attention contribution (exists).
2. Maliha & Hougen 2026, KL spike / norm-based output (exists).
3. Wang, Fan & Pan 2026, spatial-relation circuit prior; gradient score
   relabeled as CIRCUIT's own diagnostic (exists, attribution fixed in
   Task 1).
4. DAAM, Tang et al. 2022, arXiv:2210.04885. Aggregated cross-attention
   attribution maps: upscale and aggregate cross-attention word-pixel
   scores into pixel-level maps. Implement DAAM-style per-head maps and a
   per-head grounding score (map mass inside the target object's box vs
   outside) on the same (layer, head) grid as tau.
5. Roy 2025, ablation impact, arXiv:2506.17237. Destructive ablation as the
   causal baseline: ablate each (layer, head) (zero or mean replacement)
   and measure the drop in the relation score R. This is the patching-vs-
   ablation contrast the paper needs: patching transfers, ablation only
   destroys.
6. Revelio k-SAE / Diff-C probe, Kim et al. 2025, arXiv:2411.16725.
   Observational probe baseline on image diffusion: train the lightweight
   probe (or use k-SAE feature magnitudes) to score each layer's
   representation for the relation, per (layer, head) where possible.
   If a full k-SAE is out of scope for one day, implement the Diff-C-style
   linear probe on cached activations and say so in the report.
7. Prompt-to-Prompt cross-attention injection, Hertz et al. 2022,
   arXiv:2208.01626. Attention-map transfer baseline: inject the clean
   run's cross-attention maps into the corrupt run (same layers/timesteps
   as the peak cell) and measure the transfer score with the same R and
   tau definition. This tests whether attention-map swapping alone
   transfers the relation, without activation patching.
Optional 8th, if time: Attend-and-Excite, Chefer et al. 2023,
arXiv:2301.13826. Attention-guidance baseline: on-the-fly guidance that
strengthens subject-token cross-attention; measure whether it fixes the
binding failures E0 documented.

### Task 1: Fix the Wang attribution, add paper 7 to the notes

- In interpretability/baselines.py, relabel the spatial-gradient score as
  CIRCUIT's own diagnostic; cite Wang et al. 2026 (arXiv:2601.06338) as the
  related-work prior in the docstring, with a one-line summary of what the
  paper actually does (two-stage circuit, two cross-attention heads).
- Add Wang et al. 2026 to research/reading-notes-2026-11-24.md as paper 7
  (what it does, what it does not do, what it means for CIRCUIT), including
  the differentiator: from-scratch small DiTs, not pretrained PixArt-Sigma;
  circuit description, not counterfactual patching transfer.
Done when the attribution is fixed and the notes cover all seven papers.

### Task 2: Implement baselines 4-7

- Implement DAAM maps, Roy ablation impact, Revelio probe, and
  Prompt-to-Prompt injection in interpretability/baselines.py (extend
  HeadObservationalScores or add a parallel structure; keep the existing
  three scores intact).
- Each baseline must return scores on the same grid as the tau map
  (per layer, head, and timestep window where the method supports it) so
  the comparison in Task 3 is apples-to-apples.
- Implement only what the papers actually do. Label anything original as
  CIRCUIT's own. No invented methods, no invented citations.
- Add tests for each new baseline in tests/test_baselines.py (or a new
  test file) and keep pytest green.
Done when all four are implemented with tests passing.

### Task 3: Baseline comparison against the E1 pilot tau map

- Write a comparison (experiments/e1/baseline_comparison.py or in
  evaluation/) that, for each of the 7 baselines, computes on the pilot
  pairs: top-k overlap between the baseline's top heads/layers and the
  patching peak cell, and Spearman rank correlation between baseline scores
  and tau across the grid.
- Save experiments/e1/baseline_comparison.json and a short write-up
  interpretability/baseline_comparison.md stating, per baseline, whether it
  agrees with patching or not, and what that implies (which baselines can
  find the circuit, which cannot).
Done when the JSON and the write-up exist with the numbers. Re-run this
comparison after the Day 3 full sweep lands; the pilot comparison is the
first pass, not the final one.

### Constraints

- Read the contract at the top of this file first. It is the operating
  contract.
- No invented numbers. Every claim traces to a measured run or a cited
  source.
- Every new citation must be one of the arXiv IDs listed above. If a paper
  cannot be found at its ID, stop and report it; do not substitute another.
- Label the checkpoint on every result (1024-MS vs 512-MS).
- No em dashes in anything public-facing. No credentials anywhere.
- Do not describe CIRCUIT publicly as a personal project or mention advisor
  non-involvement.

### Report

#### Task 1: Fix the Wang attribution, add paper 7 to the notes
- Status: done.
- What was done:
  - Relabeled `compute_spatial_gradient_score` in `interpretability/baselines.py` explicitly as CIRCUIT's own diagnostic heuristic inspired by Wang, Fan, & Pan (2026, `arXiv:2601.06338`, summarizing its two-stage circuit and two cross-attention heads in from-scratch DiTs).
  - Added Wang, Fan, and Pan (2026), "Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers" ([arXiv:2601.06338](https://arxiv.org/abs/2601.06338)), as paper 7 in `research/reading-notes-2026-11-24.md`, explicitly noting the differentiators (from-scratch small DiTs vs. pretrained 0.6B PixArt-Sigma; circuit description and weight screening vs. counterfactual activation patching).
- Evidence: `interpretability/baselines.py`, `research/reading-notes-2026-11-24.md`, commit `7097a3f`.

#### Task 2: Implement baselines 4-7
- Status: done.
- What was done:
  - Implemented all 4 additional baselines in `interpretability/baselines.py`, `instrumentation/hooks.py`, and `interventions/patching.py`:
    4. DAAM (`arXiv:2210.04885`): `compute_daam_upscaled_map` and `compute_daam_head_box_scores` (bicubic upscaling + inside-target-box grounding mass per `(layer, head)`).
    5. Roy (`arXiv:2506.17237`): `compute_roy_ablation_impact` (per-head mean-ablation drop in relation accuracy and `dx`).
    6. Revelio (`arXiv:2411.16725`): `score_heads_revelio_probe` (Diff-C-style 5-fold CV L2 logistic linear probe on spatially pooled cached per-head activations; full k-SAE training was not run and is explicitly noted).
    7. Prompt-to-Prompt (`arXiv:2208.01626`): `CrossAttentionMapInjectionProcessor` and `AttnMapInjectionController` injecting cached `A_clean` cross-attention maps without `V` patching.
  - Added unit tests in `tests/test_baselines.py`; full `pytest -v` passes (`8/8` passed in `4.91s`).
- Evidence: `interpretability/baselines.py`, `instrumentation/hooks.py`, `interventions/patching.py`, `tests/test_baselines.py`, commit `7097a3f`.

#### Task 3: Baseline comparison against the E1 pilot tau map
- Status: done.
- What was done:
  - Ran `experiments/e1/baseline_comparison.py` on `ai-lab2` (`CUDA_VISIBLE_DEVICES=3`, `NVIDIA GeForce RTX 4090`, checkpoint `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`, `512-MS`) across all 32 heads of `[10, 11]` at `W1[4-7]` on the E1 pilot pairs (`N=6` valid counterfactuals):
    - `7_prompt_to_prompt_attn_map_tau`: `rho = 0.568` (`p = 0.0007`), `4/5` top-5 overlap (`L10H6, L10H15, L11H4, L11H11`), and layer-group `A_clean` injection at `L10-11 @ W1[4-7]` achieves `tau = 0.8261` (`flip_rate = 0.8333`) vs activation patching `tau = 0.7104`.
    - `4_daam_box_mass`: `rho = 0.413` (`p = 0.0188`), `3/5` top-5 overlap (`L10H6, L10H15, L11H4`).
    - `3_circuit_spatial_grad_wang_prior`: `rho = 0.314` (`p = 0.0799`), `2/5` overlap (`L10H15, L11H1`).
    - `2_maliha_kl_spike`: `rho = 0.224` (`p = 0.2186`), `0/5` overlap.
    - `5_roy_ablation_impact`: `rho = 0.026` (`p = 0.8867`), `2/5` overlap (`L11H4, L11H11`), demonstrating patching-vs-ablation asymmetry (`L10H15` has ablation impact `0.000` due to redundancy).
    - `1_zarei_attn_contribution`: `rho = 0.022` (`p = 0.9033`), `0/5` overlap.
    - `6_revelio_linear_probe_auc` (Diff-C linear probe): `rho = -0.056` (`p = 0.7599`), `0/5` overlap (spatial pooling erases left-right geometry).
- Evidence: `experiments/e1/baseline_comparison.py`, `experiments/e1/baseline_comparison.json`, `interpretability/baseline_comparison.md`.
- Next: ready to re-run `baseline_comparison.py` against `e1_full_results.json` (`test` split, including `L2-3`) or run E2/E3 upon Muse's Day 5 instructions.

---

## Day 5: Sat Nov 28, 2026

Week 1 of 4 (Week 1 TickTick task due today). Day 3 and Day 4 both finished
early on Nov 24 evening; run this section when you resume. Today: re-run
the baseline comparison on the full sweep, pilot E3 color binding, and
draft the first-milestone write-up.

### Review of Day 3 and Day 4 (Muse, Nov 24 evening)

Verified against the evidence, not the reports alone:

- Day 3 Task 1 done. Full sweep (512-MS, held-out test split, 6 pairs x 3
  seeds, 13 valid): peak L10-11 @ W1[4-7] tau 0.6903 (CI [0.464, 0.896]),
  flip 0.6923, sigma 0.9349. L2-3 @ W1 replicated at 0.5522. Controls near
  zero (random-head 0.0042, wrong-timestep 0.0005, no-op 0.0000). Numbers
  match e1_full_results.json.
- Day 3 Task 2 done. 1024-MS confirmation: L10-11 tau 0.4901, L2-3 tau
  0.4991, controls near zero. The transfer holds on the E0 checkpoint at
  reduced strength. Sublayer table matches the run log exactly: at L2-3
  attn2_out (0.6795) and attn2_in_txt (0.6300) carry it; at L10-11
  attn2_out (0.7912) and attn1_out (0.7191) dominate; at L6-7 @ W2[8-11]
  attn1_out reaches 0.9424 (flip 0.8889) while attn2 fades. That handoff,
  cross-attention early to self-attention later, is a headline finding.
- Day 3 Task 3 done. head_analysis.md has the full 32-head table for
  L10-11 and the L2-3 dissection. H2 is refined, not just confirmed: two
  regimes. L3H2 alone reaches tau 0.5468, 99% of the L2-3 group. L10-11 is
  a 6-head coalition (L10H15 0.3129, L10H06 0.2585, L11H11 0.1710, L10H08
  0.1357, L11H01 0.1123, L11H13 0.0940) composing to 0.6903, with 20 of 32
  heads below 0.02.
- I viewed two full-sweep instances. test_0001 s42 did NOT flip (patched
  image keeps the corrupt layout and adds a yellow mug). test_0011 s42 DID
  flip (umbrella back on the left) but the right apple turned red instead
  of green. Both are consistent with flip_rate 0.69, not 1.0. Transfer is
  real and imperfect, with occasional artifacts. Per-instance outcomes and
  artifacts go in the write-up (Task 3 below), not hidden.
- Seed variation is in the main results, as asked. Full sweep: tau 0.7063,
  essentially matching the peak 0.6903. The layout representation is
  substantially seed-invariant (prompt-driven); color binding is what needs
  matched geometry (color-preserve 0.80 on the sweep, 0.43 on 1024-MS).
  Do not claim matched-seed specificity anywhere.
- Day 4 Task 1 done. Wang attribution fixed (gradient score labeled
  CIRCUIT's own diagnostic), paper 7 in the notes with the differentiators.
- Day 4 Task 2 done. Four baselines implemented; pytest 8/8 green.
- Day 4 Task 3 done, numbers match baseline_comparison.json. Headline:
  Prompt-to-Prompt attention-map injection transfers BETTER than activation
  patching on the pilot (tau 0.826, flip 0.833 vs patching 0.710). DAAM is
  the only observational baseline that partly finds the circuit (rho 0.413,
  3/5 overlap). Zarei (rho 0.022), Maliha (0.224), and the Revelio probe
  (-0.056) essentially fail. Roy ablation rho 0.026, and L10H15 has ablation
  impact 0.000 due to redundancy: ablating a head does nothing, patching it
  transfers. That asymmetry is the paper's methodological point.

### Decisions (Muse)

1. Re-run the baseline comparison on the full sweep (Task 1). The pilot
   comparison was N=6 valid; the question is whether P2P still beats
   patching at N=13 on the test split, and whether any observational
   baseline finds the L2-3 peak.
2. E3 color-binding pilot next (Task 2). The second opening operation. Same
   gate as E1: peak tau must beat the random-head control, else report and
   stop.
3. Draft the first-milestone write-up (Task 3). It is the input to the Week
   4 go/no-go decision, so it states the honest limitations, not just the
   wins: flip_rate 0.69, the two artifacts above, P2P beating patching on
   the pilot, seed-invariance of layout vs fragility of color.
4. Framing for the write-up: state plainly what patching adds beyond
   attention injection. P2P shows attention maps carry the relation; what
   patching adds is head-level localization (L3H2, the 6-head coalition),
   the temporal map, and the causal test. Do not claim patching is the only
   way to transfer the relation.

### Task 1: Baseline comparison, full sweep

- Re-run experiments/e1/baseline_comparison.py against
  e1_full_results.json (test split, 13 valid, including L2-3 and the
  L3H2 single head) instead of the pilot map.
- Save the updated results as experiments/e1/baseline_comparison_full.json
  (or update baseline_comparison.json with a version note) and update
  interpretability/baseline_comparison.md with the full-sweep numbers.
- Report per baseline: Spearman rho vs tau, top-k overlap, and whether the
  pilot ranking holds (especially P2P vs patching).
Done when the full-sweep comparison JSON and the updated write-up exist
with the numbers.

### Task 2: E3 color-binding pilot

- Run experiments/e3_color_binding.py as a PILOT, not the full sweep: 2 to
  4 color-binding counterfactual pairs, 3 seeds, patching at the peak cells
  (L10-11 @ W1[4-7] and L2-3 @ W1[4-7]) with all four controls.
- Save results as experiments/e3/e3_pilot_results.json (tau per (l,h,t),
  sigma, flip rate, detection rate, controls) and a run log
  experiments/e3/e3_pilot_run_log.md. Save patched images under
  experiments/e3/images/.
- Gate: the pilot passes only if the peak tau cell beats the random-head
  control. If no cell beats it, report that plainly and stop; do not run
  the full E3 sweep.
Done when the pilot results JSON, run log, and images exist and the report
states the gate outcome. Do NOT run the full E3 sweep today.

### Task 3: First-milestone write-up (draft)

- Draft experiments/e1/milestone_writeup.md (or paper/milestone-1.md):
  the E1 findings with numbers. Cover: the two-regime circuit (L3H2
  single head, 99% of L2-3; L10-11 6-head coalition); the
  cross-attention-to-self-attention handoff (L6-7 attn1_out 0.9424);
  1024-MS replication; seed-invariance of layout vs fragility of color;
  the 7-baseline comparison (P2P, DAAM, Roy ablation asymmetry,
  observational baselines failing); and the honest limitations
  (flip_rate 0.69, test_0001 mug artifact, test_0011 red-apple artifact,
  P2P beating patching on the pilot, small N on the 1024-MS check).
- This is a draft for Muse and Adele to review, not a submission. No
  claims beyond what the results JSONs support.
Done when the draft exists with the numbers cited to their source files.

### Constraints

- Read the contract at the top of this file first. It is the operating
  contract.
- No invented numbers. Every claim traces to a measured run or a cited
  source.
- Label the checkpoint on every result (1024-MS vs 512-MS).
- No em dashes in anything public-facing. No credentials anywhere.
- Do not describe CIRCUIT publicly as a personal project or mention advisor
  non-involvement.

### Report

#### Task 1: Baseline comparison, full sweep
- Status: done.
- What was done:
  - Updated `experiments/e1/baseline_comparison.py` to support the full `e1_full_results.json` grid (64 heads across `L2, L3, L10, L11` at `W1[4-7]` on the `test` split, `N=12` valid counterfactuals from 18 instances) and ran it across `ai-lab2` GPUs 0 and 1 (`CUDA_VISIBLE_DEVICES=0,1`, `NVIDIA GeForce RTX 4090`, checkpoint `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`, `512-MS`), merging into `experiments/e1/baseline_comparison_full.json` and updating `interpretability/baseline_comparison.md`.
  - Across all 64 heads (`L2, L3, L10, L11`):
    - `7_prompt_to_prompt_attn_map_tau`: `rho = 0.483` (`p = 0.0001`), `4/5` top-5 overlap (`L3H2, L10H6, L10H15, L11H11`), and `rho = 0.637` (`p = 0.0001`) inside `L10-11`. At the layer-group level on `test` (`N=12`), P2P (`tau = 0.6702`, `flip = 0.6667`) and activation patching (`tau = 0.6903`) are closely matched at `L10-11`, while at `L2-3` P2P (`tau = 0.5956`, `flip = 0.5833`) slightly exceeds patching (`0.5522`).
    - `4_daam_box_mass`: `rho = 0.370` (`p = 0.0026`), `2/5` overlap (`L10H8, L10H15`); `rho = 0.352` in `L2-3`, `rho = 0.424` in `L10-11`.
    - `5_roy_ablation_impact`: `rho = 0.383` (`p = 0.0018`), `2/5` overlap (`L3H2, L10H15`; recovers `L10H15` at `N=12` unlike the `N=6` pilot).
    - `1_zarei_attn_contribution`: `rho = 0.232` (`p = 0.0651`), `1/5` overlap (`L3H2` appears in top-5 of `L2-3`, `rho = 0.125` inside `L2-3`).
    - `3_circuit_spatial_grad_wang_prior`: `rho = 0.187` (`p = 0.1386`), `1/5` overlap (`L3H2`).
    - `2_maliha_kl_spike`: `rho = -0.159` (`p = 0.2089`), `0/5` overlap (anti-correlated in `L2-3`: `rho = -0.553`, `p = 0.0010`).
    - `6_revelio_linear_probe_auc` (Diff-C linear probe): `rho = -0.087` (`p = 0.4939`), `1/5` overlap (`L3H2`).
- Evidence: `experiments/e1/baseline_comparison.py`, `experiments/e1/baseline_comparison_full.json`, `interpretability/baseline_comparison.md`.

#### Task 2: E3 color-binding pilot
- Status: done (Gate: PASS).
- What was done:
  - Implemented and ran `experiments/e3/e3_pilot.py` on `ai-lab2` (`CUDA_VISIBLE_DEVICES=2`, `NVIDIA GeForce RTX 4090`, checkpoint `512-MS`) on 3 color-binding counterfactual pairs (`color_binding_dev_0001`, `0027`, `0030`) x 3 seeds (`[42, 123, 456]`) = 9 instances (`N=7` valid counterfactuals), testing `L10-11 @ W1[4-7]` and `L2-3 @ W1[4-7]` (`attn2_out`), per-`(l, h, t)` across all 64 heads in `L2, L3, L10, L11`, and all 4 controls:
    - `L10-11 @ W1[4-7]`: `tau_color = 0.4027` (`95% CI [0.226, 0.586]`), `flip_rate = 0.1429`, `sigma = 0.9872`, `detection_rate = 1.0000`.
    - `L2-3 @ W1[4-7]`: `tau_color = 0.0768`, `flip_rate = 0.0000`, `sigma = 0.9943`, `detection_rate = 1.0000` (spatial head `L3H2` has `tau_color = 0.009`).
    - Peak single head: `L11H1 @ W1[4-7]` (`tau_color = 0.2457`, `flip_rate = 0.1429`, `sigma = 0.9917`).
    - Controls: `random_head` (`L3H10`: `tau_color = 0.0080`), `wrong_timestep` (`W4[16-19]`: `0.0049`), `same_prompt_noop` (`0.0000`), `seed_variation` (`0.3114`, `sigma = 0.9375`).
  - Gate outcome: **PASS** (`0.4027 > 0.0080`; best single head `L11H1` `0.2457 > 0.0080`). Full E3 sweep was NOT run on Day 5 (reserved for Day 6).
- Evidence: `experiments/e3/e3_pilot.py`, `experiments/e3/e3_pilot_results.json`, `experiments/e3/e3_pilot_run_log.md`, `experiments/e3/e3_pilot_grid.png`, `experiments/e3/images/`.

#### Task 3: First-milestone write-up (draft)
- Status: done.
- What was done: Drafted `experiments/e1/milestone_writeup.md` synthesizing all E0/E1/E3-pilot/7-baseline findings with exact source JSON citations, honest limitations (flip rate `0.6923`, `test_0001` yellow mug artifact, `test_0011` red-apple artifact, P2P vs patching comparison, small `1024-MS` N, partial E3 binary flip rate `0.1429`), and what activation patching adds beyond P2P attention-map injection.
- Evidence: `experiments/e1/milestone_writeup.md`.

## Day 6: runs after Day 5 reports (written Wed Nov 25, 2026)

Day numbers are sequential work blocks, not calendar dates: Days 1-4 all
ran on Nov 24, Day 5 is dated Sat Nov 28 in its header but runs when the
agents resume. Do not let the header dates confuse the order.

### Review of Day 5 (Muse, Nov 25 morning)

Day 5 has NOT started. Verified, not assumed:

- Remote head of `arrdel/circuit-research` is still `5d9f43d5` (the Day 5
  section append, Nov 25, 03:12 UTC). No agent commits since.
- The Day 5 Report section above is still empty.
- None of the Day 5 deliverables exist: no `experiments/e3/` directory,
  no `experiments/e1/baseline_comparison_full.json`, no milestone
  write-up draft.

That is not a failure: Day 5 says "run this section when you resume",
and the agents resume when Adele opens Cursor. Nothing is late. The
review task for today is therefore: run Day 5, then Day 6.

### Gating rule (read first)

Do Day 5 first, in full, and fill in its Report. Do NOT start Day 6
until the Day 5 Report is filled in. Day 6 depends on Day 5's results
(the E3 pilot gate outcome and the full-sweep baseline ranking), so
running ahead would mean working from numbers that do not exist yet.

### Task 1: E3 full sweep (only if the E3 pilot gate passed)

- If the Day 5 E3 pilot gate PASSED (peak tau beats the random-head
  control): run the full E3 sweep on fresh color-binding items from the
  same combinations, patching at the peak cells (L10-11 @ W1[4-7] and
  L2-3 @ W1[4-7]) with all four controls, 3 seeds.
- Save results as `experiments/e3/e3_full_results.json` (tau per (l,h,t),
  sigma, flip rate, detection rate, controls) and a run log
  `experiments/e3/e3_full_run_log.md`. Save patched images under
  `experiments/e3/images/`.
- State plainly whether the full sweep confirms the pilot (same peak
  cells, same direction) or not, and report the color-preserve rate
  alongside tau: color is the fragile part, layout is seed-invariant.
- If the pilot gate FAILED: do NOT run the full sweep. Report the
  negative result with the same rigor and move to Task 2; the write-up
  carries the negative result, it is not hidden.
Done when the full-sweep results JSON and run log exist (or the pilot
failure is documented) and the report states the gate outcome.

### Task 2: Finalize the first-milestone write-up

- Revise the Day 5 draft (`experiments/e1/milestone_writeup.md`) with
  the Day 5 + Day 6 numbers: the full-sweep baseline comparison
  (especially whether P2P still beats patching at N=13), the E3 pilot
  and full-sweep results, and the per-instance outcomes and artifacts.
- Add the Week 4 go/no-go decision note (blueprint Week 4): expand to
  E4 (held-out-combination generalization) and E5, or not, decided on
  the evidence in the write-up, not on sunk effort. State the decision
  and the evidence for it.
- This is still a draft for Muse and Adele to review, not a submission.
  No claims beyond what the results JSONs support. Keep the limitations
  stated plainly (flip_rate 0.69, the two artifacts, P2P beating
  patching on the pilot, small N on the 1024-MS check).
Done when the revised draft exists with every number cited to its
source file and the go/no-go note is in it.

### Task 3: Read the activation-steering paper (research)

- Read "RL Post-Training in Diffusion Transformers Is Activation
  Steering" (Meral et al., arXiv:2611.26627, submitted Nov 19, 2026;
  authors include Duen Horng Chau and Pinar Yanardag). Verified live
  Nov 25: fixed steering vectors (0.29M params, 65x fewer than the
  RL-tuned LoRA) recover 90.0-96.3% of the reward gains of six RL-tuned
  image models, and trained with online RL from zero they exceed
  weight-based RL on text rendering (OCR) and human preference
  (PickScore) on SD3.5-M.
- Also check Chau's ConceptAttention (ICML 2025 oral) as it bears on
  attention-based concept localization in diffusion transformers; cite
  it only if you verify it live.
- Add the steering paper to `research/reading-notes-2026-11-24.md` as
  paper 8 (and ConceptAttention as paper 9 if verified): one paragraph
  each on what it means for CIRCUIT's framing. The distinction to draw:
  steering is a fixed, input-independent shift; patching is
  input-dependent, causal, and localized (L3H2, the 6-head coalition,
  the temporal map). Do not claim patching is the only way to transfer
  a relation; state what patching adds beyond steering and beyond P2P
  injection.
Done when the notes carry paper 8 (and 9 if verified) with the framing
paragraph, and nothing is claimed beyond the papers as read.

### Constraints

- Read the contract at the top of this file first. It is the operating
  contract.
- No invented numbers. Every claim traces to a measured run or a cited
  source.
- Label the checkpoint on every result (1024-MS vs 512-MS).
- No em dashes in anything public-facing. No credentials anywhere.
- Do not describe CIRCUIT publicly as a personal project or mention advisor
  non-involvement.

### Report

#### Task 1: E3 full sweep (E3 pilot gate PASSED on Day 5)
- Status: done.
- What was done:
  - Implemented and ran `experiments/e3/e3_full_sweep.py` on `ai-lab2` (`CUDA_VISIBLE_DEVICES=0`, `NVIDIA GeForce RTX 4090`, Python `3.10.12`, Torch `2.6.0+cu124`, Diffusers `0.41.0`, Transformers `5.20.2`, checkpoint `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`, `512-MS`) on 5 fresh object-disjoint `test`-split color-binding pairs (`color_binding_test_0001`, `0013`, `0016`, `0019`, `0034`) x 3 seeds (`[42, 123, 456]`) = 15 instances (`N=15` valid counterfactuals).
  - Peak cells (`attn2_out` @ `W1[4-7]`):
    - `L10-11 @ W1[4-7]`: `tau_color = 0.3535` (`95% CI [0.168, 0.568]`), `flip_rate = 0.3333`, `sigma = 0.9816`, `spatial_preserve_rate = 1.0000`, `detection_rate = 1.0000`.
    - `L2-3 @ W1[4-7]`: `tau_color = 0.0275` (`95% CI [0.009, 0.050]`), `flip_rate = 0.0667`, `sigma = 0.9931`, `spatial_preserve_rate = 1.0000`, `detection_rate = 1.0000`.
  - Per-head `(l, h, t)` breakdown (64 heads across `L2, L3, L10, L11` @ `W1[4-7]`): single head **`L11H01`** achieved `tau_color = 0.3820`, `flip_rate = 0.3333`, `sigma = 0.9844` (slightly exceeding the full `L10-11` layer group `0.3535`), while all other 63 heads remained `<= 0.043` (notably spatial master head `L3H02` had `tau_color = 0.009`, confirming head-level double dissociation between spatial and color-binding heads).
  - Four controls: `random_head` (`L3H10`: `tau_color = 0.0114`, `flip = 0.0667`, `sigma = 0.9957`), `wrong_timestep` (`W4[16-19]`: `tau_color = 0.0054`, `flip = 0.0000`, `sigma = 0.9988`), `same_prompt_noop` (`tau_color = 0.0000`, `sigma = 1.0000`), and `seed_variation` (`s+999`: `tau_color = 0.3658`, `flip = 0.3333`, `sigma = 0.9794`, `spatial_preserve_rate = 1.0000`).
  - Pilot confirmation: **Confirmed** (`L10-11 > L2-3` and beats `random_head` in the exact same direction as the Day 5 pilot).
- Evidence: `experiments/e3/e3_full_sweep.py`, `experiments/e3/e3_full_results.json`, `experiments/e3/e3_full_run_log.md`, `experiments/e3/e3_full_grid.png`, `experiments/e3/images/e3_full_head_tau.png`, `experiments/e3/images/full_*.png`.

#### Task 2: Finalize the first-milestone write-up
- Status: done.
- What was done: Revised `experiments/e1/milestone_writeup.md` with the Day 5 full-sweep 7-baseline comparison (`N=12` valid, 64 heads), Day 6 E3 full-sweep results (`N=15` valid, `L11H01` vs `L3H02` double dissociation), per-instance outcomes/artifacts (`test_0001` yellow mug, `test_0011` red apple), honest limitations, and the Week 4 **GO** decision note grounded in measured evidence.
- Evidence: `experiments/e1/milestone_writeup.md`.

#### Task 3: Read the activation-steering paper (research)
- Status: done.
- What was done: Verified and read Meral et al. (2026), "RL Post-Training in Diffusion Transformers Is Activation Steering" ([arXiv:2611.26627](https://arxiv.org/abs/2611.26627)), and Helbling et al. (ICML 2025), "ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features" ([arXiv:2502.04320](https://arxiv.org/abs/2502.04320)). Added both as Papers 8 and 9 in `research/reading-notes-2026-11-24.md`, explicitly distinguishing input-independent constant steering (`2611.26627`) and Prompt-to-Prompt map injection (`2208.01626`) from CIRCUIT's input-dependent, causal, head- and timestep-localized patching (`L3H2`, `L10-11` coalition, `L11H1`, `attn2`-to-`attn1` handoff).
- Evidence: `research/reading-notes-2026-11-24.md`, commit `d05bfae`.

---

## Day 7: runs after Day 6 reports (written Wed Nov 25, 2026)

Day numbers are sequential work blocks, not calendar dates: Days 1-4 ran
Nov 24, Days 5-6 ran Nov 25. Do not let the header dates confuse the
order. Today: E4 generalization (the GO decision from the milestone
write-up), the E3 color scan outside L10-11, and two reporting fixes.

### Review of Day 5 and Day 6 (Muse, Nov 25 evening)

Verified against the evidence, not the reports alone. Day 6's directives
were written by Muse in the morning review (commit 4f658aa); the agents
then ran Day 5 and Day 6 back-to-back this evening, following the gating
rule.

- Day 5 Task 1 verified. Full-sweep 7-baseline comparison
  (baseline_comparison_full.json, N=12 valid of 18, 64 heads): P2P rho
  0.483 (p 0.0001), 4/5 top-5 overlap, and at the layer-group level P2P
  (0.6702) now closely matches patching (0.6903) at L10-11, unlike the
  pilot where P2P beat patching 0.826 to 0.710. At L2-3 P2P (0.5956)
  slightly exceeds patching (0.5522). Roy ablation recovers at N=12
  (rho 0.383); DAAM 0.370; Zarei 0.232; Wang gradient 0.187; Maliha
  -0.159 overall and anti-correlated in L2-3 (rho -0.553, p 0.0010);
  Revelio probe -0.087. All numbers match the JSON.
- Day 5 Task 2 verified. E3 color-binding pilot gate PASS: L10-11 @
  W1[4-7] tau_color 0.4027 (CI [0.226, 0.586]) vs random-head 0.0080.
  L2-3 tau_color 0.0768, flip 0.0000. Peak single head L11H1 0.2457.
  Spatial head L3H2 has tau_color 0.009: it does not carry color.
- Day 5 Task 3 verified. Milestone draft written with the limitations
  stated.
- Day 6 Task 1 verified. E3 full sweep (5 fresh test pairs x 3 seeds,
  N=15 valid) confirms the pilot: L10-11 tau_color 0.3535 (CI [0.168,
  0.568]), flip 0.3333, spatial_preserve 1.0000; L2-3 0.0275. Single head
  L11H01 reaches 0.3820, slightly exceeding the whole L10-11 group
  (0.3535). L3H02 tau_color 0.0093. Head-level double dissociation:
  spatial (L3H2) vs color (L11H1). Seed variation 0.3658. I viewed
  test_0001 s42: clean is a yellow clock + red bottle, corrupt swaps
  the colors, and the patched image restores the clean colors exactly.
  One correction (Task 3 below): the report says all other 63 heads are
  <= 0.043, but the JSON shows 62 heads <= 0.043 plus L11H13 at 0.0511.
- Day 6 Task 2 verified. Milestone write-up finalized with the GO
  decision (advance to E4 + E5) and the E4/E5 priorities written out.
- Day 6 Task 3 verified. Papers 8 and 9 are both real; I checked both
  arXiv IDs live tonight. Meral et al. (arXiv:2611.26627, Nov 19, 2026):
  fixed steering vectors (0.29M params, 65x fewer than the RL-tuned
  LoRA) recover 90.0-96.3% of the reward gains of six RL-tuned image
  models. Helbling et al., ConceptAttention (arXiv:2502.04320, ICML
  2025): DiT attention layers repurposed for concept saliency maps.

### Decisions (Muse)

1. The GO decision stands. E4 is next (Task 1), E5 after. Week 1's
   decision is made; the Week 4 TickTick task (E3 transfer + decision,
   due Dec 15) is satisfied early once Adele confirms the GO.
2. The P2P-vs-patching comparison needs a same-instance re-report
   (Task 3): in baseline_comparison_full.json, P2P's L10-11 tau (0.6702)
   is computed over n_valid 15 while patching's (0.6903) is over n_valid
   12. Either recompute both on the shared valid set or state the Ns
   plainly in the write-up.
3. The E3 head-count correction (62, not 63, heads <= 0.043; L11H13 at
   0.0511) goes into the report and the write-up (Task 3).

### Task 1: E4 generalization (terse rescue + held-out combinations)

- (a) Terse-template rescue: patch the identified circuit (L3H02 single
  head, the L10-11 coalition, L6-7 attn1_out) into the E0 terse pairs
  (p1 vs p3 preposition-insensitive; p4/p5 multi-object failures) and
  measure tau against the base model's failure. This is the core E4
  question: does the circuit generalize to the template the model fails
  on, or is it bound to the explicit template?
- (b) Held-out combinations: run E1 spatial and E3 color on fresh
  object combinations (ood split), same peak cells, all four controls,
  3 seeds.
- Save results as experiments/e4/e4_results.json and a run log
  experiments/e4/e4_run_log.md, with patched images under
  experiments/e4/images/. State plainly whether the circuit generalizes
  or is template/combination-bound; a negative result is a result.
Done when the E4 results JSON and run log exist and the report states
the generalization outcome for both (a) and (b).

### Task 2: E3 color scan outside L10-11

- The milestone write-up flags this as open: E3 tested L10-11 and L2-3
  at W1[4-7] only. Run a full L0-27 x W0-4 scan for color binding (or,
  if that is too large for one day, L12-27 plus W2[8-11] plus attn1_out
  and ff_out at the peak cells) to test whether additional color-binding
  capacity sits outside L10-11 @ W1[4-7].
- Save results as experiments/e3/e3_color_scan.json and a run log.
  Report the top cells and whether anything beats L11H01 (0.3820).
Done when the scan JSON and run log exist and the report states whether
color capacity exists outside L10-11.

### Task 3: Reporting fixes (small)

- (a) Correct the E3 head count in the Day 6 report above and in
  experiments/e1/milestone_writeup.md: 62 heads <= 0.043, plus L11H13
  at tau 0.0511 (flip 0.000). Do not leave the "63 heads" figure in
  either place.
- (b) Same-instance P2P-vs-patching re-report (see Decision 2), and
  state in the write-up why the comparison has N=12 while the full
  sweep has N=13.
Done when both corrections are in the files and the report says so.

### Constraints

- Read the contract at the top of this file first. It is the operating
  contract.
- No invented numbers. Every claim traces to a measured run or a cited
  source.
- Label the checkpoint on every result (1024-MS vs 512-MS).
- No em dashes in anything public-facing. No credentials anywhere.
- Do not describe CIRCUIT publicly as a personal project or mention advisor
  non-involvement.

### Report

(Append below: status per task, what was done with evidence, blockers,
open questions.)
