# Reading Notes: Six Core Papers (2026-11-24, updated 2026-11-25)

## 1. Roy (2025): Mechanistic Interpretability of Diffusion Models: Circuit-Level Analysis and Causal Validation

Link: [arXiv:2506.17237](https://arxiv.org/abs/2506.17237)

**What it does.** Roy presents a circuit-level mechanistic analysis and ablation-based causal validation framework on a custom 262M-parameter hybrid U-Net + selective transformer-attention diffusion model (applying 8-head attention only at 16x16 and 8x8 bottleneck resolutions) trained on 2,000 procedural synthetic facial images and 2,000 CelebA facial images. Using attention entropy, a composite activation complexity metric (std x range x sparsity), mutual-information flow efficiency, and layer/head ablation impact (measured as relative MSE increase), Roy identifies four temporal denoising phases (noise reduction at t=900-700, structure formation at t=600-400, detail refinement at t=300-100, and final enhancement at t=100-0), specialized low- and high-entropy attention heads (reporting 25.6% to 128.3% ablation degradation in the abstract), and a middle-layer computational bottleneck producing an impact score of 1.538 (153.8% MSE degradation, rising from baseline MSE 0.0246 to ablated MSE 0.0624 at t=300 in Section 4.6, Table 3).

**What it does not do.** Roy does not study pretrained text-to-image Diffusion Transformers (DiTs such as PixArt-Sigma), does not analyze cross-attention conditioning or text-to-image compositional operations (spatial relations or attribute binding), and does not perform counterfactual activation patching to transfer a target causal property from one prompt to another while preserving non-target attributes. All experiments are restricted to unconditional/class-like 64x64 facial image generation on a small custom hybrid U-Net, and validation relies on destructive ablation impact on MSE rather than selective causal transfer and specificity controls.

**What it means for CIRCUIT.** Roy is the closest prior in framing diffusion interpretability around temporal denoising phases, head specialization, and interventional necessity testing, and validates the need for statistical rigor (bootstrap CIs, multi-seed checks, effect sizes) and layer/timestep-resolved interventions. For CIRCUIT, this directly establishes our whitespace: moving from destructive MSE ablation on a 262M custom facial U-Net to selective counterfactual activation patching, necessity ablation, and specificity controls on compositional spatial and attribute-binding circuits in a pretrained 0.6B text-to-image DiT (PixArt-Sigma).

---

## 2. Maliha and Hougen (ACL Findings 2026): Mechanistic Interpretability of Text-to-Image Diffusion Models via Cross-Attention Interventions

Link: [ACL Anthology 2026.findings-acl.1265](https://aclanthology.org/2026.findings-acl.1265/)

**What it does.** Maliha and Hougen analyze cross-attention conditioning in U-Net Stable Diffusion (v1.4, v1.5, and v2.0-base) by recording projected cross-attention contributions `Delta = A * V * W_O` rather than raw attention probabilities alone. They introduce a fixed-seed single-token removal protocol (comparing full-prompt vs. single-word-ablated runs) and a head-resolved KL-divergence spike score `S(h, i)` to attribute prompt tokens to cross-attention heads, aggregate sensitivity by part of speech (showing nouns and adjectives trigger distinct head subsets), validate head importance via top-5 vs. random head removal, and demonstrate value-amplification token steering on multi-object prompts.

**What it does not do.** Maliha and Hougen restrict their scope strictly to U-Net cross-attention layers in Stable Diffusion, explicitly leaving self-attention, feed-forward blocks, and Diffusion Transformer (DiT) architectures out of scope, and they aggregate across timesteps and heads for most diagnostics rather than mapping 2D layer-by-timestep causal localization windows. Moreover, their intervention protocol uses destructive token deletion (which alters CLIP contextual embeddings of remaining tokens) and value scaling, rather than counterfactual activation patching between matched relational or attribute-swapped prompts.

**What it means for CIRCUIT.** Their methodological finding that fixed-seed control doubles causal signal-to-noise (and that norm-based projected output `A * V * W_O` outperforms raw attention probabilities) directly informs CIRCUIT's instrumentation and intervention design. At the same time, their focus on U-Net cross-attention token deletion leaves open how self-attention, cross-attention, and MLP layers in a DiT (PixArt-Sigma with T5-XXL conditioning) causally encode non-commutative spatial relations and color attribute binding across denoising timesteps.

---

## 3. Kim, Thomas, and Ghadiyaram (ICCV 2025): Revelio: Interpreting and leveraging semantic information in diffusion models

Link: [arXiv:2411.16725](https://arxiv.org/abs/2411.16725)

**What it does.** Revelio studies how visual semantic information is organized across layers and denoising timesteps in Stable Diffusion (1.5, 2.1), DeepFloyd-IF, and class-conditional DiT using two tools: Top-K sparse autoencoders (k-SAEs) trained on spatially pooled activations (evaluated via label purity `sigma_label` and maximally activating images), and Diff-C, a lightweight 40M-parameter convolutional probe classifier. They show that U-Net bottleneck layers encode coarse class shapes at intermediate noise (`t=200`) while `up_ft1` encodes fine-grained attributes at low noise (`t=25`), whereas DiT concentrates class-specific semantics in middle layers (`block14`) with weaker local spatial inductive bias in PCA visualizations than U-Nets.

**What it does not do.** Revelio is purely observational and correlational: it probes unconditional or single-class representations via k-SAEs and linear/conv classifiers without performing causal interventions, activation patching, or ablation necessity tests. It also does not analyze text-conditioned multi-object scenes, spatial relations, or attribute-object binding in text-to-image DiTs.

**What it means for CIRCUIT.** Revelio confirms that DiT semantic representations concentrate in middle transformer blocks (`block14` in DiT-XL/2) and behave qualitatively differently from U-Net spatial feature maps due to global self-attention. This provides a concrete empirical prior for where CIRCUIT's activation patching sweeps on PixArt-Sigma may locate semantic and relational bottlenecks, while underscoring why observational probing must be paired with counterfactual patching and specificity controls to establish causality.

---

## 4. Wang et al. (ICML 2026): DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders

Link: [arXiv:2602.05859](https://arxiv.org/abs/2602.05859)

**What it does.** DLM-Scope adapts Top-K sparse autoencoders to discrete masked Diffusion Language Models (Dream-7B and LLaDA-8B), comparing masked-token vs. unmasked-token activation sampling, measuring explained variance and masked-token LM loss delta, and performing multi-step diffusion-time steering. They find that early-layer SAEs can reduce masked LM loss, that multi-step steering in deep DLM layers achieves 2-10x higher steering effectiveness than single-pass LLM steering, that confidence-based decoding orders exhibit structured SAE feature drift after decoding, and that base-model SAEs transfer to instruction-tuned models in shallow/middle layers before diverging at the final layer.

**What it does not do.** DLM-Scope studies discrete text diffusion (masked token prediction in language models) rather than continuous latent image Diffusion Transformers, and focuses on single-feature SAE decomposition and concept steering rather than discovering multi-component compositional circuits via counterfactual activation patching.

**What it means for CIRCUIT.** Per CIRCUIT's guardrails, DLM-Scope occupies the discrete DLM SAE extraction space, which CIRCUIT explicitly avoids duplicating. Instead, DLM-Scope contributes a transferable methodological lesson for continuous DiTs: multi-step diffusion trajectories amplify localized interventions across iterative denoising steps, making it essential in CIRCUIT to test whether causal effects depend on early vs. late denoising windows and to separate genuine circuit mechanisms from cumulative perturbation drift using norm-matched and shuffled controls.

---

## 5. Huang et al. (NeurIPS 2023): T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation

Link: [arXiv:2307.06350](https://arxiv.org/abs/2307.06350)

**What it does.** T2I-CompBench (and its extended version T2I-CompBench++) introduces a standardized benchmark of compositional text-to-image prompts spanning attribute binding (color, shape, texture), object relationships (2D spatial, 3D spatial, non-spatial), generative numeracy, and complex multi-object compositions, alongside task-aligned automated metrics: Disentangled BLIP-VQA for attribute binding and UniDet-based object detection/center-comparison metrics for 2D spatial relations and numeracy. It benchmarks leading T2I models (including SD1.4/v2, Attend-and-Excite, SDXL, PixArt-alpha, SD3, and FLUX.1) and validates metric correlation with human judgments, plus proposes GORS reward-driven fine-tuning.

**What it does not do.** T2I-CompBench is purely a black-box evaluation benchmark and fine-tuning baseline; it does not inspect internal model activations, localize computational components, or provide matched counterfactual prompt pairs designed for causal activation patching and specificity evaluation.

**What it means for CIRCUIT.** Per CIRCUIT's guardrails, CIRCUIT-Bench does not rebuild T2I-CompBench's detector scoring from scratch; instead, it builds a counterfactual-pair layer on top of detector-based spatial and attribute evaluation (measuring bounding-box `dx = cx_b - cx_a` and object-crop color binding). T2I-CompBench's empirical observation that PixArt-family models still fail reproducibly on 2D spatial relations and multi-object attribute binding motivates CIRCUIT's E0 baseline and establishes the diagnostic failure cases needed for E1 activation patching.

---

## 6. Zarei et al. (2025): Localizing Knowledge in Diffusion Transformers

Link: [arXiv:2505.18832](https://arxiv.org/abs/2505.18832)

**What it does.** Zarei, Basu, Rezaei, Lin, Nag, and Feizi propose a model- and knowledge-agnostic method for localizing factual and semantic concepts across transformer blocks in Diffusion Transformers (PixArt-alpha, FLUX, and SANA) using the LOCK benchmark of paired knowledge-specific and knowledge-neutral prompts across six categories (artistic styles, celebrities, landmarks, copyrighted characters, animals, and sensitive concepts). They rank DiT blocks via cross-attention contribution scores (`cont_{i,j}`) averaged over image tokens, seeds, and prompts, validate causal influence by overwriting cross-attention inputs in the top-K blocks with knowledge-neutral prompt activations, and show that restricting DreamBooth personalization or concept-unlearning fine-tuning to the localized blocks outperforms full-model fine-tuning while reducing training time and memory.

**What it does not do.** Zarei et al. localize static entity and style concepts rather than compositional operations such as non-commutative spatial relations (left vs. right) or multi-object attribute binding, and they intervene purely at the block-input level (`encoder_hidden_states` / cross-attention K-V replacement) aggregated across all denoising timesteps rather than mapping 2D layer-by-timestep causal windows, dissecting self-attention vs. cross-attention vs. MLP sublayers, or isolating individual attention heads. They also do not measure continuous counterfactual transfer `tau` alongside non-target attribute specificity `sigma` or compare against matched temporal, wrong-layer, and norm-matched controls.

**What it means for CIRCUIT.** Zarei et al. establishes the primary DiT localization baseline (`attn2_in_txt` input swapping and attention-contribution block/head ranking) against which CIRCUIT should benchmark, demonstrating that knowledge distribution varies strongly across DiT architectures and concept types. For CIRCUIT, this motivates testing whether relational spatial mechanisms in PixArt-Sigma follow the same block-level cross-attention text-conditioning pattern or instead depend on temporally localized cross-attention injection followed by downstream self-attention routing, verified with counterfactual patching and specificity controls.

