"""Controlled counterfactual prompt-pair generation for CIRCUIT."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


TaskType = Literal["spatial_lr", "color_binding"]
SplitType = Literal["dev", "test", "ood"]


@dataclass(frozen=True)
class CounterfactualPair:
    """Matched prompt pair differing on a single controlled causal factor."""

    pair_id: str
    task: TaskType
    split: SplitType
    clean_prompt: str
    corrupt_prompt: str
    obj_a: str
    obj_b: str
    color_a_clean: str
    color_b_clean: str
    color_a_corrupt: str
    color_b_corrupt: str
    relation_clean: str  # "left" | "right" | "none"
    relation_corrupt: str  # "left" | "right" | "none"
    template_id: str
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CounterfactualPair:
        return cls(**data)


# Visually distinct, COCO/OpenImages-friendly object categories for reliable detection.
IN_DOMAIN_OBJECTS: tuple[str, ...] = (
    "apple",
    "cup",
    "cat",
    "dog",
    "car",
    "bicycle",
    "chair",
    "clock",
    "bottle",
    "umbrella",
    "banana",
    "book",
)

OOD_OBJECTS: tuple[str, ...] = (
    "vase",
    "suitcase",
    "teddy bear",
    "laptop",
    "backpack",
    "cake",
)

COLORS: tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
)

SPATIAL_TEMPLATES: dict[str, tuple[str, str]] = {
    # Each entry: (clean_template with A left of B, corrupt_template with A right of B)
    "to_the_side_of": (
        "A {color_a} {obj_a} to the left of a {color_b} {obj_b} on a plain white background",
        "A {color_a} {obj_a} to the right of a {color_b} {obj_b} on a plain white background",
    ),
    "on_the_side_explicit": (
        "A {color_a} {obj_a} on the left and a {color_b} {obj_b} on the right, clean studio photo",
        "A {color_a} {obj_a} on the right and a {color_b} {obj_b} on the left, clean studio photo",
    ),
    "side_by_side_scene": (
        "A {color_a} {obj_a} positioned on the left side and a {color_b} {obj_b} positioned on the right side of a wooden table",
        "A {color_a} {obj_a} positioned on the right side and a {color_b} {obj_b} positioned on the left side of a wooden table",
    ),
}

COLOR_BINDING_TEMPLATES: dict[str, tuple[str, str]] = {
    "fixed_lr_swap_colors": (
        "A {color_a} {obj_a} on the left and a {color_b} {obj_b} on the right, clean studio photo",
        "A {color_b} {obj_a} on the left and a {color_a} {obj_b} on the right, clean studio photo",
    ),
    "next_to_swap_colors": (
        "A {color_a} {obj_a} next to a {color_b} {obj_b} on a plain white background",
        "A {color_b} {obj_a} next to a {color_a} {obj_b} on a plain white background",
    ),
}


def _distinct_object_pairs(objects: tuple[str, ...]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i != j:
                pairs.append((a, b))
    return pairs


def _distinct_color_pairs(colors: tuple[str, ...]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, c1 in enumerate(colors):
        for j, c2 in enumerate(colors):
            if i != j:
                pairs.append((c1, c2))
    return pairs


def build_spatial_lr_pairs(
    num_dev: int = 32,
    num_test: int = 64,
    num_ood: int = 32,
    templates: tuple[str, ...] | None = None,
    base_seed: int = 1000,
) -> list[CounterfactualPair]:
    """Generate controlled left/right spatial counterfactual pairs."""
    rng = random.Random(base_seed)
    tpl_keys = list(templates) if templates is not None else list(SPATIAL_TEMPLATES.keys())

    in_obj_pairs = _distinct_object_pairs(IN_DOMAIN_OBJECTS)
    ood_obj_pairs = _distinct_object_pairs(OOD_OBJECTS)
    color_pairs = _distinct_color_pairs(COLORS)

    rng.shuffle(in_obj_pairs)
    rng.shuffle(ood_obj_pairs)

    all_pairs: list[CounterfactualPair] = []

    def _sample_split(
        split: SplitType,
        count: int,
        obj_pool: list[tuple[str, str]],
        seed_offset: int,
    ) -> None:
        for idx in range(count):
            obj_a, obj_b = obj_pool[idx % len(obj_pool)]
            color_a, color_b = color_pairs[(idx * 7 + seed_offset) % len(color_pairs)]
            tpl_id = tpl_keys[idx % len(tpl_keys)]
            clean_tpl, corrupt_tpl = SPATIAL_TEMPLATES[tpl_id]
            clean_prompt = clean_tpl.format(
                color_a=color_a, obj_a=obj_a, color_b=color_b, obj_b=obj_b
            )
            corrupt_prompt = corrupt_tpl.format(
                color_a=color_a, obj_a=obj_a, color_b=color_b, obj_b=obj_b
            )
            all_pairs.append(
                CounterfactualPair(
                    pair_id=f"spatial_lr_{split}_{idx:04d}",
                    task="spatial_lr",
                    split=split,
                    clean_prompt=clean_prompt,
                    corrupt_prompt=corrupt_prompt,
                    obj_a=obj_a,
                    obj_b=obj_b,
                    color_a_clean=color_a,
                    color_b_clean=color_b,
                    color_a_corrupt=color_a,
                    color_b_corrupt=color_b,
                    relation_clean="left",
                    relation_corrupt="right",
                    template_id=tpl_id,
                    seed=base_seed + seed_offset + idx,
                )
            )

    dev_pool = in_obj_pairs[: max(1, len(in_obj_pairs) // 3)]
    test_pool = in_obj_pairs[max(1, len(in_obj_pairs) // 3) :]
    _sample_split("dev", num_dev, dev_pool, seed_offset=0)
    _sample_split("test", num_test, test_pool, seed_offset=10000)
    _sample_split("ood", num_ood, ood_obj_pairs, seed_offset=20000)
    return all_pairs


def build_color_binding_pairs(
    num_dev: int = 32,
    num_test: int = 64,
    num_ood: int = 32,
    templates: tuple[str, ...] | None = None,
    base_seed: int = 5000,
) -> list[CounterfactualPair]:
    """Generate controlled color-attribute binding counterfactual pairs."""
    rng = random.Random(base_seed)
    tpl_keys = list(templates) if templates is not None else list(COLOR_BINDING_TEMPLATES.keys())

    in_obj_pairs = _distinct_object_pairs(IN_DOMAIN_OBJECTS)
    ood_obj_pairs = _distinct_object_pairs(OOD_OBJECTS)
    color_pairs = _distinct_color_pairs(COLORS)

    rng.shuffle(in_obj_pairs)
    rng.shuffle(ood_obj_pairs)

    all_pairs: list[CounterfactualPair] = []

    def _sample_split(
        split: SplitType,
        count: int,
        obj_pool: list[tuple[str, str]],
        seed_offset: int,
    ) -> None:
        for idx in range(count):
            obj_a, obj_b = obj_pool[idx % len(obj_pool)]
            color_a, color_b = color_pairs[(idx * 5 + seed_offset) % len(color_pairs)]
            tpl_id = tpl_keys[idx % len(tpl_keys)]
            clean_tpl, corrupt_tpl = COLOR_BINDING_TEMPLATES[tpl_id]
            clean_prompt = clean_tpl.format(
                color_a=color_a, obj_a=obj_a, color_b=color_b, obj_b=obj_b
            )
            corrupt_prompt = corrupt_tpl.format(
                color_a=color_a, obj_a=obj_a, color_b=color_b, obj_b=obj_b
            )
            rel = "left" if tpl_id == "fixed_lr_swap_colors" else "none"
            all_pairs.append(
                CounterfactualPair(
                    pair_id=f"color_binding_{split}_{idx:04d}",
                    task="color_binding",
                    split=split,
                    clean_prompt=clean_prompt,
                    corrupt_prompt=corrupt_prompt,
                    obj_a=obj_a,
                    obj_b=obj_b,
                    color_a_clean=color_a,
                    color_b_clean=color_b,
                    color_a_corrupt=color_b,
                    color_b_corrupt=color_a,
                    relation_clean=rel,
                    relation_corrupt=rel,
                    template_id=tpl_id,
                    seed=base_seed + seed_offset + idx,
                )
            )

    dev_pool = in_obj_pairs[: max(1, len(in_obj_pairs) // 3)]
    test_pool = in_obj_pairs[max(1, len(in_obj_pairs) // 3) :]
    _sample_split("dev", num_dev, dev_pool, seed_offset=0)
    _sample_split("test", num_test, test_pool, seed_offset=10000)
    _sample_split("ood", num_ood, ood_obj_pairs, seed_offset=20000)
    return all_pairs


def save_pairs_jsonl(pairs: list[CounterfactualPair], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict()) + "\n")
    return out_path


def load_pairs_jsonl(path: str | Path) -> list[CounterfactualPair]:
    in_path = Path(path)
    pairs: list[CounterfactualPair] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(CounterfactualPair.from_dict(json.loads(line)))
    return pairs
