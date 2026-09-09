from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Iterable

import numpy as np


RESPONSE_NAMES = (
    "pitch",
    "turn_count",
    "curvature",
    "mean_radius",
    "tip_displacement",
    "maximum_stress",
    "WT_energy",
    "Frank_energy",
)


def immutable_float_array(values: np.ndarray) -> np.ndarray:
    """Return a float64 array backed by immutable bytes."""
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)


@dataclass(frozen=True)
class SolverDiagnostics:
    mechanical_residual: float
    director_residual: float
    incompressibility_residual: float
    director_norm_residual: float
    converged: bool
    wall_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.converged, (bool, np.bool_)):
            raise ValueError("Solver convergence status must be Boolean.")
        values = np.asarray(
            [
                self.mechanical_residual,
                self.director_residual,
                self.incompressibility_residual,
                self.director_norm_residual,
                self.wall_time,
            ],
            dtype=float,
        )
        if np.any(np.isnan(values)) or np.any(values < 0.0):
            raise ValueError("Solver residuals and wall time must be nonnegative and not NaN.")
        if self.converged and not np.all(np.isfinite(values)):
            raise ValueError("Converged solver diagnostics must be finite.")
        object.__setattr__(self, "converged", bool(self.converged))

    def fingerprint_tail(self) -> np.ndarray:
        return np.asarray(
            [
                self.mechanical_residual,
                self.director_residual,
                self.incompressibility_residual,
                self.director_norm_residual,
                float(self.converged),
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class EvaluationRecord:
    design: np.ndarray
    fidelity: int
    response: np.ndarray
    objectives: np.ndarray
    constraint_values: np.ndarray
    diagnostics: SolverDiagnostics
    material_package: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    loading_state: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    error: str | None = None

    def __post_init__(self) -> None:
        design = np.asarray(self.design, dtype=float)
        response = np.asarray(self.response, dtype=float)
        objectives = np.asarray(self.objectives, dtype=float)
        if design.shape != (14,):
            raise ValueError("EvaluationRecord.design must contain the 14 manuscript variables.")
        if response.shape != (8,):
            raise ValueError("EvaluationRecord.response must follow manuscript Eq. (12) and contain 8 values.")
        if objectives.shape != (3,):
            raise ValueError("EvaluationRecord.objectives must follow manuscript Eq. (3) and contain 3 values.")
        if self.fidelity not in {0, 1, 2, 3}:
            raise ValueError("Fidelity must be F0, F1, F2 or F3.")
        constraint_values = np.atleast_1d(np.asarray(self.constraint_values, dtype=float))
        material_package = np.atleast_1d(np.asarray(self.material_package, dtype=float))
        loading_state = np.atleast_1d(np.asarray(self.loading_state, dtype=float))
        diagnostics = np.asarray(
            [
                self.diagnostics.mechanical_residual,
                self.diagnostics.director_residual,
                self.diagnostics.incompressibility_residual,
                self.diagnostics.director_norm_residual,
                self.diagnostics.wall_time,
            ],
            dtype=float,
        )
        if self.diagnostics.converged and not all(
            np.all(np.isfinite(value))
            for value in (
                design,
                response,
                objectives,
                constraint_values,
                material_package,
                loading_state,
                diagnostics,
            )
        ):
            raise ValueError("A converged evaluation record cannot contain nonfinite values.")
        for name, value in (
            ("design", design),
            ("response", response),
            ("objectives", objectives),
            ("constraint_values", constraint_values),
            ("material_package", material_package),
            ("loading_state", loading_state),
        ):
            object.__setattr__(self, name, immutable_float_array(value))

    @property
    def feasible(self) -> bool:
        diagnostics = np.asarray(
            [
                self.diagnostics.mechanical_residual,
                self.diagnostics.director_residual,
                self.diagnostics.incompressibility_residual,
                self.diagnostics.director_norm_residual,
                self.diagnostics.wall_time,
            ],
            dtype=float,
        )
        return bool(
            self.diagnostics.converged
            and self.error is None
            and np.all(np.isfinite(self.response))
            and np.all(np.isfinite(self.objectives))
            and np.all(np.isfinite(self.constraint_values))
            and np.all(np.isfinite(diagnostics))
            and np.all(self.constraint_values <= 0.0)
            and self.objectives[2] <= 1.0
        )

    @property
    def fingerprint(self) -> np.ndarray:
        """Manuscript Eq. (14): response plus four residuals and solver status."""
        return np.concatenate((self.response, self.diagnostics.fingerprint_tail()))

    @property
    def usable_for_residual_learning(self) -> bool:
        diagnostics = np.asarray(
            [
                self.diagnostics.mechanical_residual,
                self.diagnostics.director_residual,
                self.diagnostics.incompressibility_residual,
                self.diagnostics.director_norm_residual,
            ]
        )
        return bool(
            self.diagnostics.converged
            and self.error is None
            and np.all(np.isfinite(self.response))
            and np.all(np.isfinite(self.material_package))
            and np.all(np.isfinite(self.loading_state))
            and np.all(np.isfinite(diagnostics))
        )


def _design_key(design: np.ndarray) -> str:
    canonical = np.asarray(design, dtype="<f8").tobytes()
    return hashlib.sha256(canonical).hexdigest()


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("Objectives must be a two-dimensional array.")
    mask = np.ones(len(values), dtype=bool)
    for index, value in enumerate(values):
        dominated = np.any(np.all(values <= value, axis=1) & np.any(values < value, axis=1))
        mask[index] = not dominated
    return mask


class FidelityRepository:
    """Complete repository R and HF-only feasible archive P from Algorithm 1."""

    def __init__(self) -> None:
        self._records: list[EvaluationRecord] = []
        self._latest: dict[tuple[str, int], EvaluationRecord] = {}

    def add(self, record: EvaluationRecord) -> None:
        key = (_design_key(record.design), record.fidelity)
        self._records.append(record)
        self._latest[key] = record

    def extend(self, records: Iterable[EvaluationRecord]) -> None:
        for record in records:
            self.add(record)

    def get(self, design: np.ndarray, fidelity: int) -> EvaluationRecord | None:
        return self._latest.get((_design_key(design), int(fidelity)))

    def at_fidelity(self, fidelity: int) -> list[EvaluationRecord]:
        return [record for record in self._records if record.fidelity == fidelity]

    def records(self) -> tuple[EvaluationRecord, ...]:
        return tuple(self._records)

    def paired(self, lower: int, upper: int) -> list[tuple[EvaluationRecord, EvaluationRecord]]:
        if upper != lower + 1:
            raise ValueError("The manuscript residual hierarchy uses adjacent fidelity pairs only.")
        pairs: list[tuple[EvaluationRecord, EvaluationRecord]] = []
        for record in self.at_fidelity(upper):
            lower_record = self.get(record.design, lower)
            if lower_record is not None:
                pairs.append((lower_record, record))
        return pairs

    @property
    def high_fidelity_count(self) -> int:
        return len(self.at_fidelity(3))

    @property
    def archive(self) -> list[EvaluationRecord]:
        """Only F3-verified feasible designs may enter the reported archive."""
        latest: dict[str, EvaluationRecord] = {}
        for record in self.at_fidelity(3):
            latest[_design_key(record.design)] = record
        feasible = [record for record in latest.values() if record.feasible]
        if not feasible:
            return []
        mask = nondominated_mask(np.vstack([record.objectives for record in feasible]))
        return [record for record, keep in zip(feasible, mask, strict=True) if keep]

    def validate_nested_indices(self, index_sets: dict[int, set[int]]) -> None:
        required = {0, 1, 2, 3}
        if set(index_sets) != required:
            raise ValueError("Nested initialization requires index sets I0, I1, I2 and I3.")
        if not index_sets[3] <= index_sets[2] <= index_sets[1] <= index_sets[0]:
            raise ValueError("Algorithm 1 requires I3 subset I2 subset I1 subset I0.")
