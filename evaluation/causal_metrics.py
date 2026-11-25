"""Formal causal metrics: transfer score (tau), specificity (sigma), necessity, and CIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from evaluation.spatial_eval import ImageEvalResult


@dataclass
class PairCausalOutcome:
    pair_id: str
    task: str
    valid_counterfactual: bool
    tau: float | None  # [0, 1] continuous transfer toward clean
    flipped_to_clean: bool | None
    sigma_non_target: float | None  # non-target property retention [0, 1]
    both_detected_patched: bool
    color_preserved: bool | None
    sim_to_corrupt: float | None
    sim_to_clean: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CausalSummary:
    n_total: int
    n_valid: int
    tau_mean: float | None
    tau_ci_low: float | None
    tau_ci_high: float | None
    flip_rate: float | None
    sigma_mean: float | None
    sigma_ci_low: float | None
    sigma_ci_high: float | None
    both_detected_rate: float
    color_preserve_rate: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return None
    return float(np.dot(a, b) / denom)


def compute_pair_causal_outcome(
    pair_id: str,
    task: str,
    clean: ImageEvalResult,
    corrupt: ImageEvalResult,
    patched: ImageEvalResult,
    clean_relation: str = "left",
    corrupt_relation: str = "right",
) -> PairCausalOutcome:
    """Compute transfer score tau and specificity sigma for a single counterfactual pair."""
    sim_corrupt = _cosine(patched.siglip_embedding, corrupt.siglip_embedding)
    sim_clean = _cosine(patched.siglip_embedding, clean.siglip_embedding)

    if task == "spatial_lr":
        valid = (
            clean.dx is not None
            and corrupt.dx is not None
            and (clean.dx - corrupt.dx) > 0.04
            and clean.relation_pred == clean_relation
            and corrupt.relation_pred == corrupt_relation
        )
        tau: float | None = None
        flipped: bool | None = None
        if clean.dx is not None and corrupt.dx is not None and patched.dx is not None:
            denom = clean.dx - corrupt.dx
            if abs(denom) > 0.04:
                raw_tau = (patched.dx - corrupt.dx) / denom
                tau = float(np.clip(raw_tau, 0.0, 1.0))
            flipped = patched.relation_pred == clean_relation

        # Non-target preservation: both objects detected AND color binding preserved
        color_ok = patched.color_binding_correct
        if corrupt.color_binding_correct is False and patched.color_binding_correct is False:
            # Neither bound cleanly at baseline; use detection retention
            color_preserved: bool | None = None
        else:
            color_preserved = color_ok

        sigma_terms: list[float] = [1.0 if patched.both_detected else 0.0]
        if color_preserved is not None:
            sigma_terms.append(1.0 if color_preserved else 0.0)
        if sim_corrupt is not None:
            # Clamp normalized SigLIP similarity to [0, 1]
            sigma_terms.append(float(np.clip(sim_corrupt, 0.0, 1.0)))
        sigma = float(np.mean(sigma_terms))

    elif task == "color_binding":
        m_clean = (
            (clean.color_a_margin + clean.color_b_margin)
            if (clean.color_a_margin is not None and clean.color_b_margin is not None)
            else None
        )
        m_corrupt = (
            (corrupt.color_a_margin + corrupt.color_b_margin)
            if (corrupt.color_a_margin is not None and corrupt.color_b_margin is not None)
            else None
        )
        m_patched = (
            (patched.color_a_margin + patched.color_b_margin)
            if (patched.color_a_margin is not None and patched.color_b_margin is not None)
            else None
        )
        valid = (
            m_clean is not None
            and m_corrupt is not None
            and (m_clean - m_corrupt) > 0.02
            and bool(clean.color_binding_correct)
        )
        tau = None
        flipped = None
        if m_clean is not None and m_corrupt is not None and m_patched is not None:
            denom = m_clean - m_corrupt
            if abs(denom) > 0.02:
                tau = float(np.clip((m_patched - m_corrupt) / denom, 0.0, 1.0))
            flipped = bool(patched.color_binding_correct)

        color_preserved = None
        sigma_terms = [1.0 if patched.both_detected else 0.0]
        if clean.relation_pred in ("left", "right"):
            sigma_terms.append(1.0 if patched.relation_pred == corrupt.relation_pred else 0.0)
        if sim_corrupt is not None:
            sigma_terms.append(float(np.clip(sim_corrupt, 0.0, 1.0)))
        sigma = float(np.mean(sigma_terms))
    else:
        raise ValueError(f"Unsupported task: {task}")

    return PairCausalOutcome(
        pair_id=pair_id,
        task=task,
        valid_counterfactual=valid,
        tau=tau,
        flipped_to_clean=flipped,
        sigma_non_target=sigma,
        both_detected_patched=patched.both_detected,
        color_preserved=color_preserved,
        sim_to_corrupt=sim_corrupt,
        sim_to_clean=sim_clean,
    )


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via nonparametric percentile bootstrap."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute bootstrap CI on empty values.")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return mean, lo, hi


def summarize_causal_outcomes(
    outcomes: list[PairCausalOutcome],
    valid_only: bool = True,
) -> CausalSummary:
    """Aggregate causal outcomes with bootstrap confidence intervals."""
    subset = [o for o in outcomes if o.valid_counterfactual] if valid_only else outcomes
    taus = [o.tau for o in subset if o.tau is not None]
    flips = [1.0 if o.flipped_to_clean else 0.0 for o in subset if o.flipped_to_clean is not None]
    sigmas = [o.sigma_non_target for o in subset if o.sigma_non_target is not None]
    colors = [1.0 if o.color_preserved else 0.0 for o in subset if o.color_preserved is not None]

    tau_m, tau_lo, tau_hi = bootstrap_ci(taus) if taus else (None, None, None)
    sig_m, sig_lo, sig_hi = bootstrap_ci(sigmas) if sigmas else (None, None, None)

    return CausalSummary(
        n_total=len(outcomes),
        n_valid=len([o for o in outcomes if o.valid_counterfactual]),
        tau_mean=tau_m,
        tau_ci_low=tau_lo,
        tau_ci_high=tau_hi,
        flip_rate=float(np.mean(flips)) if flips else None,
        sigma_mean=sig_m,
        sigma_ci_low=sig_lo,
        sigma_ci_high=sig_hi,
        both_detected_rate=float(np.mean([1.0 if o.both_detected_patched else 0.0 for o in subset]))
        if subset
        else 0.0,
        color_preserve_rate=float(np.mean(colors)) if colors else None,
    )
