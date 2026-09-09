"""Deterministic test fixture only; it is not the manuscript's missing F0-F3 solver."""

from __future__ import annotations

import numpy as np

from .objectives import ObjectiveInputs, compute_objectives
from .records import EvaluationRecord, SolverDiagnostics
from .reliability import GateEvidence
from .reliability import MarginalUncertaintyCalibrator
from .schema import DesignDomain


def _fixture_response(design: np.ndarray, fidelity: int) -> np.ndarray:
    unit = DesignDomain().normalize(design)
    bias = np.asarray([0.08, 0.18, 0.04, 0.08, 0.12, 0.50, 0.10, 0.06]) * (3 - fidelity) / 3.0
    response = np.asarray(
        [
            2.62 + 0.16 * unit[0],
            5.3 + 1.2 * unit[3],
            0.78 + 0.10 * unit[2],
            0.65 + 0.25 * unit[0],
            0.15 + 0.20 * unit[13],
            7.0 + 4.0 * unit[10],
            0.8 + 0.7 * unit[10],
            0.12 + 0.35 * unit[2],
        ],
        dtype=float,
    )
    return response + bias


def _fixture_objectives(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    blocking_force = 20.0 + 18.0 * (1.0 - DesignDomain().normalize(design)[11])
    mass = 0.9 + 0.3 * DesignDomain().normalize(design)[6]
    return compute_objectives(
        ObjectiveInputs(
            predicted_shape=response[:4],
            target_shape=np.asarray([2.72, 6.0, 0.83, 0.75]),
            shape_scales=np.asarray([1.50, 10.0, 0.80, 1.50]),
            blocking_force=blocking_force,
            mass=mass,
            maximum_stress=response[5],
            allowable_stress=15.0,
        )
    )


class DeterministicFixtureEvaluator:
    provenance_label = "fixture_only_deterministic_evaluator"
    structural_fixture = True

    def __init__(self, domain: DesignDomain):
        self.domain = domain

    def evaluate(self, design, fidelity, material_package, loading_state) -> EvaluationRecord:
        design = np.asarray(design, dtype=float)
        response = _fixture_response(design, fidelity)
        factor = 10.0 ** (-(fidelity + 4))
        diagnostics = SolverDiagnostics(
            mechanical_residual=0.7 * factor,
            director_residual=0.6 * factor,
            incompressibility_residual=0.4 * factor * 1.0e-2,
            director_norm_residual=0.3 * factor * 1.0e-6,
            converged=True,
            wall_time=float((fidelity + 1) ** 3),
        )
        return EvaluationRecord(
            design=design,
            fidelity=int(fidelity),
            response=response,
            objectives=_fixture_objectives(design, response),
            constraint_values=self.domain.constraint_values(design),
            diagnostics=diagnostics,
            material_package=np.asarray(material_package, dtype=float),
            loading_state=np.asarray(loading_state, dtype=float),
        )


class FixtureReliabilityEstimator:
    provenance_label = "fixture_only_reliability_estimator"
    structural_fixture = True

    def predict(self, lower_records, predicted_response, standard_deviation, calibrated_uncertainty):
        record = lower_records[2]
        response = np.asarray(predicted_response, dtype=float)
        objective = _fixture_objectives(record.design, response)
        diagnostics = record.diagnostics
        return GateEvidence(
            convergence_probability=0.99,
            mechanical_residual=diagnostics.mechanical_residual,
            director_residual=diagnostics.director_residual,
            incompressibility_residual=diagnostics.incompressibility_residual,
            director_norm_residual=diagnostics.director_norm_residual,
            calibrated_uncertainty=float(calibrated_uncertainty),
            peak_stress_ratio=float(min(objective[2], 0.95)),
            curvature=float(response[2]),
            pitch=float(response[0]),
            turns=float(response[1]),
            shape_error=float(objective[0]),
            predicted_hf_cost=64.0,
        )


class FixtureObjectiveProjector:
    provenance_label = "fixture_only_no_F3_or_oracle_access"
    structural_fixture = True

    def project(self, design: np.ndarray, response_samples: np.ndarray) -> np.ndarray:
        return np.vstack([_fixture_objectives(design, response) for response in response_samples])


def fitted_fixture_calibrator() -> MarginalUncertaintyCalibrator:
    calibrator = MarginalUncertaintyCalibrator(0.90, provenance_label="fixture_only_frozen_calibration")
    calibrator.fit(np.zeros((2, 8)), np.ones((2, 8)), np.full((2, 8), 0.1))
    return calibrator
