from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.indicators.hv import HV

from .records import immutable_float_array


@dataclass(frozen=True)
class AcquisitionPrediction:
    design: np.ndarray
    objective_samples: np.ndarray
    convergence_probability: float
    predicted_hf_cost: float

    def __post_init__(self) -> None:
        design = np.asarray(self.design, dtype=float)
        samples = np.asarray(self.objective_samples, dtype=float)
        if design.shape != (14,):
            raise ValueError("Acquisition design must contain 14 components.")
        if samples.ndim != 2 or samples.shape[1:] != (3,) or len(samples) == 0:
            raise ValueError("Acquisition objective samples must be a non-empty matrix of three objectives.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Acquisition objective samples must be finite.")
        if not np.isfinite(self.convergence_probability) or not 0.0 <= self.convergence_probability <= 1.0:
            raise ValueError("Convergence probability must lie in [0, 1].")
        if not np.isfinite(self.predicted_hf_cost) or self.predicted_hf_cost <= 0.0:
            raise ValueError("Predicted HF cost must be finite and positive.")
        for name, value in (("design", design), ("objective_samples", samples)):
            object.__setattr__(self, name, immutable_float_array(value))


def _hypervolume(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    values = np.asarray(objectives, dtype=float)
    if values.size == 0:
        return 0.0
    return float(HV(ref_point=np.asarray(reference_point, dtype=float))(values))


def hypervolume_improvement_samples(
    objective_samples: np.ndarray,
    archive_objectives: np.ndarray,
    reference_point: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(objective_samples, dtype=float)
    archive = np.asarray(archive_objectives, dtype=float).reshape(-1, samples.shape[-1])
    baseline = _hypervolume(archive, reference_point)
    improvement = []
    for sample in samples:
        combined = np.vstack((archive, sample)) if len(archive) else sample[None, :]
        improvement.append(max(0.0, _hypervolume(combined, reference_point) - baseline))
    return np.asarray(improvement, dtype=float)


def normalized_frontier_distance(
    predicted_objective: np.ndarray,
    archive_objectives: np.ndarray,
    objective_scales: np.ndarray,
) -> float:
    """Operational completion of the manuscript's unnamed d_F calculation."""
    archive = np.asarray(archive_objectives, dtype=float)
    if archive.size == 0:
        return 1.0
    scales = np.asarray(objective_scales, dtype=float)
    return float(np.min(np.linalg.norm((archive - predicted_objective) / scales, axis=1)))


def acquisition_score(
    prediction: AcquisitionPrediction,
    archive_objectives: np.ndarray,
    reference_point: np.ndarray,
    objective_scales: np.ndarray,
    eta: float,
    epsilon_0: float,
) -> float:
    """Manuscript Eq. (22): lower-quartile HV gain times reliability over HF cost."""
    if epsilon_0 <= 0.0:
        raise ValueError("epsilon_0 must be positive.")
    improvements = hypervolume_improvement_samples(
        prediction.objective_samples,
        archive_objectives,
        reference_point,
    )
    lower_quartile = float(np.quantile(improvements, 0.25))
    mean_objective = np.mean(prediction.objective_samples, axis=0)
    frontier = normalized_frontier_distance(mean_objective, archive_objectives, objective_scales)
    return float(
        lower_quartile
        * prediction.convergence_probability
        * (1.0 + eta * frontier)
        / (prediction.predicted_hf_cost + epsilon_0)
    )


def select_dual_infill(
    predictions: list[AcquisitionPrediction],
    archive_objectives: np.ndarray,
    reference_point: np.ndarray,
    objective_scales: np.ndarray,
    eta: float,
    epsilon_0: float,
    slots: int = 2,
) -> list[AcquisitionPrediction]:
    """Manuscript Eq. (23): reliable improvement, then frontier completion."""
    if slots <= 0 or not predictions:
        return []
    scores = np.asarray(
        [
            acquisition_score(
                item,
                archive_objectives,
                reference_point,
                objective_scales,
                eta,
                epsilon_0,
            )
            for item in predictions
        ]
    )
    first_index = int(np.argmax(scores))
    selected = [predictions[first_index]]
    if slots == 1 or len(predictions) == 1:
        return selected
    first_objective = np.mean(predictions[first_index].objective_samples, axis=0)
    comparison = np.vstack((archive_objectives, first_objective)) if len(archive_objectives) else first_objective[None, :]
    best_index = None
    best_distance = -np.inf
    for index, item in enumerate(predictions):
        if index == first_index:
            continue
        mean_objective = np.mean(item.objective_samples, axis=0)
        distance = normalized_frontier_distance(mean_objective, comparison, objective_scales)
        if distance > best_distance:
            best_index = index
            best_distance = distance
    # len(predictions) > 1 here, so one non-first candidate always exists.
    selected.append(predictions[int(best_index)])
    return selected
