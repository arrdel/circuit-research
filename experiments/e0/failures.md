# E0 Reproducible Compositional Failures

All runs used `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`, `DPMSolverMultistepScheduler`, 20 steps, guidance scale 4.5, 1024x1024, `torch.bfloat16`, across seeds `[42, 123, 456]`.

## Stable Compositional Failures (Observed Across All 3 Seeds)

### 1. `p4_red_cube_left_blue_sphere_green_cyl`: Three-object spatial relation + attribute binding

- **Prompt:** `Red cube left of blue sphere, green cylinder in front.`
- **What was required:** Three distinct objects with bound attributes and positions: a red cube to the left of a blue sphere, and a green cylinder positioned in front.
- **Seeds tested:** `42`, `123`, `456` (3/3 failed).
- **What the images show:**
  - `seed=42` (`experiments/e0/images/p4_red_cube_left_blue_sphere_green_cyl_seed42.png`): Spatial relation inverted (`dx = -0.2248`, blue sphere on the left and red cube on the right) and cylinder color misbound (a blue cylinder in front instead of green).
  - `seed=123` (`experiments/e0/images/p4_red_cube_left_blue_sphere_green_cyl_seed123.png`): Color-object binding swapped and third object omitted (green cube on the left, red sphere on the right, no cylinder present).
  - `seed=456` (`experiments/e0/images/p4_red_cube_left_blue_sphere_green_cyl_seed456.png`): Object omission, shape distortion, and color leakage (a red sphere resting on a red rectangular pedestal on the left, a teal-green rectangular block rather than a cylinder on the right, and no blue sphere).

### 2. `p5_two_red_cubes_one_blue_sphere`: Numeracy + attribute binding

- **Prompt:** `Two red cubes and one blue sphere.`
- **What was required:** Exact count and color-shape binding: 2 red cubes and 1 blue sphere (3 objects total).
- **Seeds tested:** `42`, `123`, `456` (3/3 failed).
- **What the images show:**
  - `seed=42` (`experiments/e0/images/p5_two_red_cubes_one_blue_sphere_seed42.png`): Undercount of cubes (1 tall red rectangular prism/cube and 1 blue sphere; missing second red cube).
  - `seed=123` (`experiments/e0/images/p5_two_red_cubes_one_blue_sphere_seed123.png`): Count error and attribute leakage (4 objects: 1 blue cube, 1 red cube, 1 blue sphere, and 1 red sphere).
  - `seed=456` (`experiments/e0/images/p5_two_red_cubes_one_blue_sphere_seed456.png`): Overcount / hallucinated object (2 stacked red cubes and 1 blue sphere, plus an extra hallucinated red sphere in the background; 4 objects total).

## Seed-Sensitive Failures and Counterfactual Relational Insensitivity (Not 3/3 Identical Failure, Key Diagnostic)

### 3. `p1` vs `p3`: Preposition Insensitivity (`left of` vs `right of`)

- **Prompts:**
  - `p1`: `Red cube left of blue sphere.`
  - `p3`: `Red cube right of blue sphere.`
- **Seeds tested:** `42`, `123`, `456`.
- **What the images show:**
  - `p1` fails spatial left/right on seeds `42` (`dx = -0.4102`, blue sphere left of red cube) and `123` (`dx = -0.4358`, blue sphere left of red cube), and on seed `456` (`dx = +0.3172`) splits the red cube into two stacked red cubes (`experiments/e0/images/p1_red_cube_left_blue_sphere_seed456.png`).
  - `p3` places the red cube on the right on seeds `42` (`dx = -0.3774`) and `123` (`dx = -0.3687`), and flips to red cube on the left on seed `456` (`dx = +0.2795`).
  - **Key finding:** Comparing `p1` and `p3` at matched seeds reveals near-identical layouts (`seed=42`: `dx=-0.4102` vs `-0.3774`; `seed=123`: `dx=-0.4358` vs `-0.3687`; `seed=456`: `dx=+0.3172` vs `+0.2795`). On this terse prompt template, PixArt-Sigma's left/right layout is governed by the initial latent seed and color-noun tokens (`p2` `"Blue cube left of red sphere."` flips cube to the left on seeds 42 and 123) rather than the relational preposition `"left of"` vs `"right of"`.

### 4. `p2_blue_cube_left_red_sphere`: Partial failure (1/3 seeds)

- **Prompt:** `Blue cube left of red sphere.`
- **Seeds tested:** `42`, `123`, `456`.
- **What the images show:** Seeds `42` (`dx = +0.3413`) and `123` (`dx = +0.4419`) satisfy both spatial order and color-shape binding. Seed `456` (`experiments/e0/images/p2_blue_cube_left_red_sphere_seed456.png`) fails shape and spatial order: it renders a blue translucent block behind a red cube with no sphere present (`dx = -0.1165`).
