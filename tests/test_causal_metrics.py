"""Unit tests for causal metrics (tau, sigma, bootstrap CI)."""

import numpy as np

from evaluation.causal_metrics import (
    compute_pair_causal_outcome,
    summarize_causal_outcomes,
)
from evaluation.spatial_eval import DetectionBox, ImageEvalResult


def _make_eval(
    dx: float,
    rel: str,
    color_ok: bool,
    emb_vec: list[float],
) -> ImageEvalResult:
    box_a = DetectionBox("apple", 0.9, 0.1, 0.2, 0.3, 0.5)
    box_b = DetectionBox("cup", 0.9, 0.1 + dx, 0.2, 0.3 + dx, 0.5)
    vec = np.asarray(emb_vec, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return ImageEvalResult(
        box_a=box_a,
        box_b=box_b,
        both_detected=True,
        dx=dx,
        relation_pred=rel,
        relation_correct=(rel == "left"),
        color_a_margin=0.1 if color_ok else -0.1,
        color_b_margin=0.1 if color_ok else -0.1,
        color_binding_correct=color_ok,
        siglip_embedding=vec,
    )


def test_spatial_tau_and_sigma() -> None:
    clean = _make_eval(0.5, "left", True, [1.0, 0.0])
    corrupt = _make_eval(-0.5, "right", True, [0.0, 1.0])
    # Full selective flip: spatial matches clean, colors & embedding stay close to corrupt
    patched = _make_eval(0.5, "left", True, [0.1, 0.99])

    outcome = compute_pair_causal_outcome(
        pair_id="p0",
        task="spatial_lr",
        clean=clean,
        corrupt=corrupt,
        patched=patched,
    )
    assert outcome.valid_counterfactual is True
    assert abs(outcome.tau - 1.0) < 1e-6
    assert outcome.flipped_to_clean is True
    assert outcome.color_preserved is True
    assert outcome.sigma_non_target is not None and outcome.sigma_non_target > 0.9

    summary = summarize_causal_outcomes([outcome, outcome], valid_only=True)
    assert summary.n_valid == 2
    assert summary.tau_mean == 1.0
    assert summary.flip_rate == 1.0
