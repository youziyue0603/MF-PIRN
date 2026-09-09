from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectiveInputs:
    predicted_shape: np.ndarray
    target_shape: np.ndarray
    shape_scales: np.ndarray
    blocking_force: float
    mass: float
    maximum_stress: float
    allowable_stress: float


def compute_objectives(inputs: ObjectiveInputs) -> np.ndarray:
    """Manuscript Eqs. (3)-(4), returned in the common minimization form."""
    predicted = np.asarray(inputs.predicted_shape, dtype=float)
    target = np.asarray(inputs.target_shape, dtype=float)
    scales = np.asarray(inputs.shape_scales, dtype=float)
    if predicted.shape != target.shape or predicted.shape != scales.shape:
        raise ValueError("Shape, target and scale arrays must have equal shape.")
    if np.any(scales <= 0.0) or inputs.mass <= 0.0 or inputs.allowable_stress <= 0.0:
        raise ValueError("Shape scales, mass and allowable stress must be positive.")
    shape_error = float(np.sqrt(np.mean(((predicted - target) / scales) ** 2)))
    specific_force = float(inputs.blocking_force / inputs.mass)
    stress_ratio = float(inputs.maximum_stress / inputs.allowable_stress)
    return np.asarray([shape_error, -specific_force, stress_ratio], dtype=float)

