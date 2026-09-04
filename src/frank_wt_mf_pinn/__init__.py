"""Strict structural reproduction of the final Version 1.5 algorithm."""

from .acquisition import AcquisitionPrediction, select_dual_infill
from .engine import AlgorithmConfig, FrankWTMFPINNOptimizer
from .mechanics import (
    frank_energy_density,
    normalized_solver_residuals,
    step_length_tensor,
    thermal_fraction,
    temperature_modulus,
    warner_terentjev_energy_density,
)
from .records import EvaluationRecord, FidelityRepository, SolverDiagnostics
from .schema import DESIGN_VARIABLES, DesignDomain, GateThresholds

__all__ = [
    "AcquisitionPrediction",
    "AlgorithmConfig",
    "DESIGN_VARIABLES",
    "DesignDomain",
    "EvaluationRecord",
    "FidelityRepository",
    "FrankWTMFPINNOptimizer",
    "GateThresholds",
    "SolverDiagnostics",
    "frank_energy_density",
    "normalized_solver_residuals",
    "select_dual_infill",
    "step_length_tensor",
    "temperature_modulus",
    "thermal_fraction",
    "warner_terentjev_energy_density",
]

