from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

import numpy as np

from .records import EvaluationRecord, immutable_float_array
from .schema import DesignDomain, GateThresholds


@dataclass(frozen=True)
class GateEvidence:
    convergence_probability: float
    mechanical_residual: float
    director_residual: float
    incompressibility_residual: float
    director_norm_residual: float
    calibrated_uncertainty: float
    peak_stress_ratio: float
    curvature: float
    pitch: float
    turns: float
    shape_error: float
    predicted_hf_cost: float


class ReliabilityEstimator(Protocol):
    provenance_label: str

    def predict(
        self,
        lower_records: tuple[EvaluationRecord, EvaluationRecord, EvaluationRecord],
        predicted_response: np.ndarray,
        standard_deviation: np.ndarray,
        calibrated_uncertainty: float,
    ) -> GateEvidence:
        ...


class MarginalUncertaintyCalibrator:
    """Explicit operational completion for the manuscript's unspecified calibration map."""

    _FROZEN_FIELDS = frozenset(
        {
            "_nominal_coverage",
            "_epsilon",
            "_provenance_label",
            "_multiplier",
            "_data_sha256",
            "_sample_count",
            "_is_fitted",
        }
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_is_fitted", False) and name in self._FROZEN_FIELDS:
            raise AttributeError("A fitted uncertainty calibrator is immutable.")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        nominal_coverage: float = 0.90,
        epsilon: float = 1.0e-12,
        provenance_label: str = "UNSPECIFIED",
    ):
        if not np.isfinite(nominal_coverage) or not 0.0 < nominal_coverage < 1.0:
            raise ValueError("nominal_coverage must lie between zero and one.")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        self._nominal_coverage = float(nominal_coverage)
        self._epsilon = float(epsilon)
        self._provenance_label = str(provenance_label)
        self._multiplier: np.ndarray | None = None
        self._data_sha256: str | None = None
        self._sample_count: int = 0
        self._is_fitted = False

    @property
    def provenance_label(self) -> str:
        return self._provenance_label

    @property
    def nominal_coverage(self) -> float:
        return self._nominal_coverage

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def multiplier(self) -> np.ndarray | None:
        return self._multiplier

    @property
    def data_sha256(self) -> str | None:
        return self._data_sha256

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def fit(self, prediction: np.ndarray, standard_deviation: np.ndarray, observed: np.ndarray) -> None:
        if self.multiplier is not None:
            raise RuntimeError("A fitted calibrator is frozen and cannot be refitted.")
        prediction = np.asarray(prediction, dtype=float)
        standard_deviation = np.asarray(standard_deviation, dtype=float)
        observed = np.asarray(observed, dtype=float)
        if prediction.shape != standard_deviation.shape or prediction.shape != observed.shape:
            raise ValueError("Prediction, standard deviation and observation arrays must have equal shape.")
        if prediction.ndim != 2 or prediction.shape[1:] != (8,) or len(prediction) == 0:
            raise ValueError("Calibration arrays must be non-empty matrices with eight response components.")
        if not all(np.all(np.isfinite(values)) for values in (prediction, standard_deviation, observed)):
            raise ValueError("Calibration arrays must contain finite values.")
        if np.any(standard_deviation < 0.0):
            raise ValueError("Calibration standard deviations must be nonnegative.")
        standardized = np.abs(observed - prediction) / (standard_deviation + self.epsilon)
        multiplier = np.quantile(standardized, self.nominal_coverage, axis=0, method="higher")
        self._multiplier = immutable_float_array(multiplier)
        digest = hashlib.sha256()
        digest.update(b"MarginalUncertaintyCalibrator/v1\0")
        digest.update(np.asarray([self.nominal_coverage, self.epsilon], dtype="<f8").tobytes())
        for values in (prediction, standard_deviation, observed):
            digest.update(np.asarray(values, dtype="<f8").tobytes())
        digest.update(np.asarray(self._multiplier, dtype="<f8").tobytes())
        self._data_sha256 = digest.hexdigest().upper()
        self._sample_count = len(prediction)
        self._is_fitted = True

    def freeze(self) -> FrozenMarginalCalibration:
        if self.multiplier is None or self.data_sha256 is None:
            raise RuntimeError("Uncertainty calibrator has not been fitted.")
        return FrozenMarginalCalibration(
            multipliers=tuple(float(value) for value in self.multiplier),
            provenance_label=self.provenance_label,
            data_sha256=self.data_sha256,
            sample_count=self.sample_count,
        )

    def interval_half_width(self, standard_deviation: np.ndarray) -> np.ndarray:
        if self.multiplier is None:
            raise RuntimeError("Uncertainty calibrator has not been fitted.")
        values = np.asarray(standard_deviation, dtype=float)
        if values.shape != (8,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Prediction standard deviations must contain eight finite nonnegative values.")
        return values * self.multiplier

    def scalar_uncertainty(self, standard_deviation: np.ndarray, response_scales: np.ndarray) -> float:
        half_width = self.interval_half_width(standard_deviation)
        scales = np.asarray(response_scales, dtype=float)
        if scales.shape != (8,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("Response scales must contain eight finite positive values.")
        return float(np.max(half_width / scales))

    def calibrate_members(self, mean: np.ndarray, members: np.ndarray) -> np.ndarray:
        """Apply the frozen marginal scale factors to the acquisition ensemble."""
        if self.multiplier is None:
            raise RuntimeError("Uncertainty calibrator has not been fitted.")
        center = np.asarray(mean, dtype=float)
        values = np.asarray(members, dtype=float)
        if center.shape != (8,) or values.ndim != 2 or values.shape[1:] != (8,) or len(values) == 0:
            raise ValueError("Mean and members must follow the eight-component response contract.")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(values)):
            raise ValueError("Mean and members must contain finite values.")
        return center[None, :] + (values - center[None, :]) * self.multiplier[None, :]


@dataclass(frozen=True)
class FrozenMarginalCalibration:
    """Optimizer-owned immutable snapshot of a fitted marginal calibration."""

    multipliers: tuple[float, ...]
    provenance_label: str
    data_sha256: str
    sample_count: int

    def __post_init__(self) -> None:
        values = np.asarray(self.multipliers, dtype=float)
        if values.shape != (8,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("A frozen calibration requires eight finite nonnegative scale factors.")
        object.__setattr__(self, "multipliers", tuple(float(value) for value in values))

    @property
    def multiplier(self) -> np.ndarray:
        return immutable_float_array(np.asarray(self.multipliers, dtype=float))

    def interval_half_width(self, standard_deviation: np.ndarray) -> np.ndarray:
        values = np.asarray(standard_deviation, dtype=float)
        if values.shape != (8,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Prediction standard deviations must contain eight finite nonnegative values.")
        return values * np.asarray(self.multipliers, dtype=float)

    def scalar_uncertainty(self, standard_deviation: np.ndarray, response_scales: np.ndarray) -> float:
        half_width = self.interval_half_width(standard_deviation)
        scales = np.asarray(response_scales, dtype=float)
        if scales.shape != (8,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("Response scales must contain eight finite positive values.")
        return float(np.max(half_width / scales))

    def calibrate_members(self, mean: np.ndarray, members: np.ndarray) -> np.ndarray:
        center = np.asarray(mean, dtype=float)
        values = np.asarray(members, dtype=float)
        if center.shape != (8,) or values.ndim != 2 or values.shape[1:] != (8,) or len(values) == 0:
            raise ValueError("Mean and members must follow the eight-component response contract.")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(values)):
            raise ValueError("Mean and members must contain finite values.")
        multiplier = np.asarray(self.multipliers, dtype=float)
        return center[None, :] + (values - center[None, :]) * multiplier[None, :]


def passes_gate(
    design: np.ndarray,
    evidence: GateEvidence,
    domain: DesignDomain,
    thresholds: GateThresholds,
) -> bool:
    """Manuscript Eq. (21) and its printed morphology limits."""
    values = np.asarray(
        [
            evidence.convergence_probability,
            evidence.mechanical_residual,
            evidence.director_residual,
            evidence.incompressibility_residual,
            evidence.director_norm_residual,
            evidence.calibrated_uncertainty,
            evidence.peak_stress_ratio,
            evidence.curvature,
            evidence.pitch,
            evidence.turns,
            evidence.shape_error,
            evidence.predicted_hf_cost,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        return False
    if not 0.0 <= evidence.convergence_probability <= 1.0:
        return False
    if min(
        evidence.mechanical_residual,
        evidence.director_residual,
        evidence.incompressibility_residual,
        evidence.director_norm_residual,
        evidence.calibrated_uncertainty,
        evidence.peak_stress_ratio,
        evidence.shape_error,
    ) < 0.0 or evidence.predicted_hf_cost <= 0.0:
        return False
    return bool(
        domain.feasible(design)
        and evidence.convergence_probability >= thresholds.p_min
        and evidence.mechanical_residual <= thresholds.tau_mech
        and evidence.director_residual <= thresholds.tau_dir
        and evidence.incompressibility_residual <= thresholds.tau_J
        and evidence.director_norm_residual <= thresholds.tau_n
        and evidence.peak_stress_ratio <= 1.0
        and evidence.calibrated_uncertainty <= thresholds.uncertainty_max
        and thresholds.curvature_min <= evidence.curvature <= thresholds.curvature_max
        and thresholds.pitch_min <= evidence.pitch <= thresholds.pitch_max
        and thresholds.turns_min <= evidence.turns <= thresholds.turns_max
        and evidence.shape_error <= thresholds.shape_error_max
    )
