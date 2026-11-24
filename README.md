# CIRCUIT

**CIRCUIT: Causal Interpretability and Reasoning Under Controlled Interventions in Diffusion Transformers**

A personal research project. No advisor involvement, by Adele's decision.

**Research question:** Do diffusion transformers develop temporally localized, causally identifiable computational mechanisms for object-attribute binding and spatial relationships?

**Opening operations (decided Nov 24, 2026):** spatial relation left/right, and color attribute binding, on PixArt-Sigma (0.6B DiT, Hugging Face Diffusers).

**First milestone:** one successful experiment showing that activation patching at a particular layer and denoising timestep can selectively transfer a spatial relationship between two otherwise matched prompts. If it works, that is the foundation. If it does not, that is a scientific result about where the proposed mechanism may not exist.

**Blueprint:** `~/workspace/frontier-repo/drafts/circuit-blueprint.md` (approved Nov 24, 2026). Literature verified, novelty matrix, hypotheses with falsifiers, causal measurement definitions (transfer score tau, specificity sigma, necessity by ablation, four controls), experiment protocols E0-E6, first-month plan.

**Status:** Week 1 (Nov 24-28, 2026): research foundation and reproduction. Week tasks live in TickTick (Inbox).

**Structure**

- `benchmark/` dataset generation, counterfactual pairs, evaluation splits
- `models/` model loading, architecture adapters, inference
- `instrumentation/` hooks, activation capture, attention extraction
- `interpretability/` feature analysis, SAEs, circuit discovery
- `interventions/` activation patching, ablation, steering
- `evaluation/` metrics, baselines, statistical analysis
- `visualization/` temporal maps, circuit diagrams, dashboards
- `experiments/` reproducible experiment configurations and results
- `demo/` Gradio application
- `paper/` technical report, figures, supplementary experiments

**Standing rules:** no credentials in any repo. No em dashes in anything public-facing. DIAL and ATAT are under review at ICLR 2027, never presented as published. No invented numbers: every figure traces to a measured run or a cited source. One track at a time until it ships something public.
