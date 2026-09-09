from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV


def normalized_hypervolume(
    candidate_objectives: np.ndarray,
    oracle_objectives: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """Manuscript Eq. (25): HV(P;u) / HV(P*;u)."""
    reference = np.asarray(reference_point, dtype=float)
    candidate = np.asarray(candidate_objectives, dtype=float).reshape(-1, len(reference))
    oracle = np.asarray(oracle_objectives, dtype=float).reshape(-1, len(reference))
    oracle_hv = float(HV(ref_point=reference)(oracle))
    if oracle_hv <= 0.0:
        raise ValueError("Hidden-oracle hypervolume must be positive.")
    candidate_hv = 0.0 if len(candidate) == 0 else float(HV(ref_point=reference)(candidate))
    return candidate_hv / oracle_hv


def igd_plus(candidate_objectives: np.ndarray, reference_objectives: np.ndarray) -> float:
    """Manuscript Eq. (26), implemented directly for auditability."""
    candidate = np.asarray(candidate_objectives, dtype=float)
    reference = np.asarray(reference_objectives, dtype=float)
    if candidate.ndim != 2 or reference.ndim != 2 or candidate.shape[1] != reference.shape[1]:
        raise ValueError("Candidate and reference objectives must be compatible matrices.")
    if len(candidate) == 0 or len(reference) == 0:
        raise ValueError("IGD+ requires non-empty candidate and reference sets.")
    distances = []
    for target in reference:
        dominance_displacement = np.maximum(candidate - target, 0.0)
        distances.append(np.min(np.linalg.norm(dominance_displacement, axis=1)))
    return float(np.mean(distances))


def vargha_delaney_a12(sample_x: np.ndarray, sample_y: np.ndarray) -> float:
    """Manuscript Eq. (27): P(X>Y) + 0.5 P(X=Y)."""
    x = np.ravel(np.asarray(sample_x, dtype=float))
    y = np.ravel(np.asarray(sample_y, dtype=float))
    if len(x) == 0 or len(y) == 0:
        raise ValueError("A12 requires two non-empty samples.")
    comparisons = x[:, None] - y[None, :]
    return float((np.sum(comparisons > 0.0) + 0.5 * np.sum(comparisons == 0.0)) / comparisons.size)


def force_retention(force_after_cycles: float, initial_force: float) -> float:
    """Manuscript Eq. (28)."""
    if initial_force == 0.0:
        raise ValueError("Initial force must be non-zero.")
    return 100.0 * float(force_after_cycles) / float(initial_force)


def actuation_strain(length_at_temperature: float, initial_length: float) -> float:
    """Manuscript Eq. (29)."""
    if initial_length == 0.0:
        raise ValueError("Initial length must be non-zero.")
    return 100.0 * (float(initial_length) - float(length_at_temperature)) / float(initial_length)


def leading_order_iteration_work(
    epochs: int,
    paired_cases: int,
    trainable_parameters: int,
    candidate_count: int,
    evaluation_counts: np.ndarray,
    fidelity_costs: np.ndarray,
) -> float:
    """Numeric proxy for the three terms in manuscript Eq. (24)."""
    counts = np.asarray(evaluation_counts, dtype=float)
    costs = np.asarray(fidelity_costs, dtype=float)
    if counts.shape != (4,) or costs.shape != (4,):
        raise ValueError("F0-F3 counts and costs must each contain four values.")
    return float(
        epochs * paired_cases * trainable_parameters
        + candidate_count * trainable_parameters
        + np.dot(counts, costs)
    )
