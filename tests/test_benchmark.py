"""Unit tests for counterfactual benchmark generation."""

from pathlib import Path

from benchmark.counterfactual_pairs import (
    build_color_binding_pairs,
    build_spatial_lr_pairs,
    load_pairs_jsonl,
    save_pairs_jsonl,
)


def test_spatial_lr_pairs_disjoint_and_matched(tmp_path: Path) -> None:
    pairs = build_spatial_lr_pairs(num_dev=12, num_test=24, num_ood=12)
    assert len(pairs) == 48

    dev_objs = {(p.obj_a, p.obj_b) for p in pairs if p.split == "dev"}
    test_objs = {(p.obj_a, p.obj_b) for p in pairs if p.split == "test"}
    ood_objs = {(p.obj_a, p.obj_b) for p in pairs if p.split == "ood"}

    assert dev_objs.isdisjoint(test_objs)
    assert dev_objs.isdisjoint(ood_objs)
    assert test_objs.isdisjoint(ood_objs)

    for p in pairs:
        assert p.clean_prompt != p.corrupt_prompt
        assert p.relation_clean == "left"
        assert p.relation_corrupt == "right"
        assert p.obj_a != p.obj_b
        assert p.color_a_clean != p.color_b_clean
        assert p.color_a_clean == p.color_a_corrupt

    out_file = save_pairs_jsonl(pairs, tmp_path / "spatial.jsonl")
    loaded = load_pairs_jsonl(out_file)
    assert loaded == pairs


def test_color_binding_pairs_swap() -> None:
    pairs = build_color_binding_pairs(num_dev=8, num_test=8, num_ood=8)
    assert len(pairs) == 24
    for p in pairs:
        assert p.color_a_clean == p.color_b_corrupt
        assert p.color_b_clean == p.color_a_corrupt
        assert p.clean_prompt != p.corrupt_prompt
