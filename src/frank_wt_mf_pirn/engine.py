from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .acquisition import AcquisitionPrediction, select_dual_infill
from .records import EvaluationRecord, FidelityRepository, SolverDiagnostics, immutable_float_array
from .reliability import (
    GateEvidence,
    MarginalUncertaintyCalibrator,
    ReliabilityEstimator,
    passes_gate,
)
from .residual import AdjacentResidualPIRN, TriCriterionReplay
from .schema import DesignDomain, GateThresholds


class FidelityEvaluator(Protocol):
    provenance_label: str

    def evaluate(
        self,
        design: np.ndarray,
        fidelity: int,
        material_package: np.ndarray,
        loading_state: np.ndarray,
    ) -> EvaluationRecord:
        ...


class ObjectiveProjector(Protocol):
    provenance_label: str

    def project(self, design: np.ndarray, response_samples: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class AlgorithmConfig:
    """Paper-locked controls plus explicitly marked operational completions."""

    seed: int
    initial_counts: tuple[int, int, int, int]
    candidate_pool_size: int
    response_scales: np.ndarray
    objective_scales: np.ndarray
    reference_point: np.ndarray
    energy_scale: float
    high_fidelity_budget: int = 200
    eta: float = 1.0
    epsilon_0: float = 1.0e-12
    replay_capacity: int = 96
    max_empty_pools: int = 100
    structural_fixture: bool = False

    def __post_init__(self) -> None:
        counts = tuple(self.initial_counts)
        if len(counts) != 4 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in counts
        ):
            raise ValueError("initial_counts must define four integer I0, I1, I2 and I3 sizes.")
        object.__setattr__(self, "initial_counts", tuple(int(value) for value in counts))
        if self.high_fidelity_budget != 200:
            raise ValueError("The final 1.5 manuscript locks the HF budget at 200.")
        if not (
            self.initial_counts[3]
            <= self.initial_counts[2]
            <= self.initial_counts[1]
            <= self.initial_counts[0]
        ):
            raise ValueError("Initial counts must satisfy |I3| <= |I2| <= |I1| <= |I0|.")
        if self.initial_counts[3] <= 0 or self.initial_counts[3] >= self.high_fidelity_budget:
            raise ValueError("Initial F3 count must be positive and smaller than the HF budget.")
        if self.candidate_pool_size < 2:
            raise ValueError("Dual infill requires a candidate pool with at least two designs.")
        if np.asarray(self.response_scales).shape != (8,):
            raise ValueError("response_scales must contain eight entries.")
        if np.asarray(self.objective_scales).shape != (3,):
            raise ValueError("objective_scales must contain three entries.")
        if np.asarray(self.reference_point).shape != (3,):
            raise ValueError("reference_point must contain three entries.")
        if not np.isfinite(self.energy_scale) or self.energy_scale <= 0.0:
            raise ValueError("energy_scale must be finite and positive.")
        if not np.isfinite(self.eta) or self.eta < 0.0:
            raise ValueError("eta must be finite and nonnegative.")
        if not np.isfinite(self.epsilon_0) or self.epsilon_0 <= 0.0:
            raise ValueError("epsilon_0 must be finite and positive.")
        if self.replay_capacity <= 0 or self.max_empty_pools <= 0:
            raise ValueError("replay_capacity and max_empty_pools must be positive.")
        for name in ("response_scales", "objective_scales", "reference_point"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values.")
            if name != "reference_point" and np.any(value <= 0.0):
                raise ValueError(f"{name} must contain positive values.")
            object.__setattr__(self, name, immutable_float_array(value))


@dataclass(frozen=True)
class CandidateAudit:
    design: np.ndarray
    passed_gate: bool
    evidence: GateEvidence
    acquisition_score_available: bool


@dataclass(frozen=True)
class IterationAudit:
    iteration: int
    pool_index: int
    high_count_before: int
    high_count_after: int
    candidate_audit: tuple[CandidateAudit, ...]
    selected_designs: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class OptimizationRun:
    archive: tuple[EvaluationRecord, ...]
    repository: FidelityRepository
    iterations: tuple[IterationAudit, ...]


class FrankWTMFPIRNOptimizer:
    """Executable transcription of Algorithm 1 in the final Version 1.5 paper."""

    def __init__(
        self,
        config: AlgorithmConfig,
        domain: DesignDomain,
        evaluator: FidelityEvaluator,
        residual_model: AdjacentResidualPIRN,
        reliability_estimator: ReliabilityEstimator,
        objective_projector: ObjectiveProjector,
        *,
        thresholds: GateThresholds | None = None,
        calibrator: MarginalUncertaintyCalibrator,
    ) -> None:
        self.config = config
        self.domain = domain
        if domain.extra_constraints is None and not config.structural_fixture:
            raise ValueError(
                "A strict production run requires the manuscript's unprinted geometry-dependent L and remaining g_j constraints."
            )
        self.evaluator = evaluator
        self.residual_model = residual_model
        self.reliability_estimator = reliability_estimator
        self.objective_projector = objective_projector
        if not hasattr(residual_model, "response_scales") or not np.array_equal(
            np.asarray(residual_model.response_scales, dtype=float),
            np.asarray(config.response_scales, dtype=float),
        ):
            raise ValueError("Algorithm and residual model must use identical response scales.")
        if not hasattr(residual_model, "energy_scale") or not np.isclose(
            float(residual_model.energy_scale), float(config.energy_scale)
        ):
            raise ValueError("Algorithm and residual model must use the same energy scale.")
        adapters = {
            "fidelity evaluator": evaluator,
            "reliability estimator": reliability_estimator,
            "objective projector": objective_projector,
        }
        for role, adapter in adapters.items():
            provenance = str(getattr(adapter, "provenance_label", "UNSPECIFIED")).strip()
            if not provenance or provenance == "UNSPECIFIED":
                raise ValueError(f"The {role} must carry audited provenance.")
            if not config.structural_fixture and getattr(adapter, "structural_fixture", False):
                raise ValueError(f"A bundled structural fixture cannot be registered as a production {role}.")
        locked_thresholds = GateThresholds()
        if thresholds is not None and thresholds != locked_thresholds:
            raise ValueError("Strict final-1.5 runs cannot override the paper-locked gate thresholds.")
        self.thresholds = locked_thresholds
        if calibrator.multiplier is None:
            raise ValueError("Algorithm 1 requires a calibrator fitted on frozen calibration data before optimization.")
        multiplier = np.asarray(calibrator.multiplier, dtype=float)
        if multiplier.shape != (8,) or not np.all(np.isfinite(multiplier)) or np.any(multiplier < 0.0):
            raise ValueError("The frozen calibrator must contain eight finite nonnegative scale factors.")
        if not calibrator.provenance_label.strip() or calibrator.provenance_label == "UNSPECIFIED" or calibrator.data_sha256 is None:
            raise ValueError("Calibration provenance must identify a frozen calibration dataset.")
        if not config.structural_fixture and calibrator.provenance_label.lower().startswith("fixture"):
            raise ValueError("A fixture calibration dataset cannot be registered for a production run.")
        self.calibrator = calibrator.freeze()
        self.replay = TriCriterionReplay(config.replay_capacity)
        self.repository = FidelityRepository()

    @staticmethod
    def _failed_record(
        design: np.ndarray,
        fidelity: int,
        material: np.ndarray,
        loading: np.ndarray,
        error: str,
    ) -> EvaluationRecord:
        return EvaluationRecord(
            design=np.asarray(design, dtype=float),
            fidelity=fidelity,
            response=np.full(8, np.nan),
            objectives=np.full(3, np.inf),
            constraint_values=np.asarray([np.inf]),
            diagnostics=SolverDiagnostics(np.inf, np.inf, np.inf, np.inf, False, np.inf),
            material_package=material,
            loading_state=loading,
            error=error,
        )

    def _evaluate(self, design: np.ndarray, fidelity: int, material: np.ndarray, loading: np.ndarray) -> EvaluationRecord:
        try:
            record = self.evaluator.evaluate(design, fidelity, material, loading)
        except Exception as error:
            record = self._failed_record(
                design,
                fidelity,
                material,
                loading,
                f"{type(error).__name__}: {error}",
            )
        if not isinstance(record, EvaluationRecord):
            failed = self._failed_record(
                design,
                fidelity,
                material,
                loading,
                f"EvaluatorContractError: returned {type(record).__name__}",
            )
            self.repository.add(failed)
            raise TypeError("Evaluator must return an EvaluationRecord.")
        if record.fidelity != fidelity or not np.array_equal(record.design, design):
            failed = self._failed_record(
                design,
                fidelity,
                material,
                loading,
                "EvaluatorContractError: wrong design or fidelity",
            )
            self.repository.add(failed)
            raise ValueError("Evaluator returned a record for the wrong design or fidelity.")
        self.repository.add(record)
        return record

    def _initialize(self, material: np.ndarray, loading: np.ndarray) -> None:
        counts = self.config.initial_counts
        designs = self.domain.sample(counts[0], self.config.seed)
        index_sets = {level: set(range(counts[level])) for level in range(4)}
        self.repository.validate_nested_indices(index_sets)
        for index in sorted(index_sets[0]):
            self._evaluate(designs[index], 0, material, loading)
        for fidelity in (1, 2, 3):
            for index in sorted(index_sets[fidelity]):
                if self.repository.get(designs[index], fidelity - 1) is None:
                    raise RuntimeError("Adjacent response is missing during nested initialization.")
                self._evaluate(designs[index], fidelity, material, loading)

    def _candidate_pool(self, pool_index: int) -> np.ndarray:
        # The manuscript fixes a seeded shared pool but not its generator.
        return self.domain.sample(self.config.candidate_pool_size, self.config.seed + 100000 + pool_index)

    def run(self, material_package: np.ndarray, loading_state: np.ndarray) -> OptimizationRun:
        material = np.atleast_1d(np.asarray(material_package, dtype=float))
        loading = np.atleast_1d(np.asarray(loading_state, dtype=float))
        if not np.all(np.isfinite(material)) or not np.all(np.isfinite(loading)):
            raise ValueError("Material package and loading state must contain finite values.")
        self._initialize(material, loading)
        iterations: list[IterationAudit] = []
        pool_index = 0
        empty_pools = 0
        iteration = 0
        while self.repository.high_fidelity_count < self.config.high_fidelity_budget:
            iteration += 1
            replay_records = self.replay.select(self.repository)
            self.residual_model.fit(self.repository, replay_records)
            candidate_forecasts: list[AcquisitionPrediction] = []
            candidate_audits: list[CandidateAudit] = []
            pool = self._candidate_pool(pool_index)
            pool_index += 1
            archive_objectives = (
                np.vstack([record.objectives for record in self.repository.archive])
                if self.repository.archive
                else np.empty((0, 3), dtype=float)
            )
            for design in pool:
                if self.repository.get(design, 3) is not None:
                    continue
                records = []
                for fidelity in (0, 1, 2):
                    record = self.repository.get(design, fidelity)
                    records.append(record or self._evaluate(design, fidelity, material, loading))
                predicted_response, standard_deviation, members = self.residual_model.predict_next(records[2], 3)
                calibrated_uncertainty = self.calibrator.scalar_uncertainty(
                    standard_deviation,
                    np.asarray(self.config.response_scales, dtype=float),
                )
                calibrated_members = self.calibrator.calibrate_members(predicted_response, members)
                evidence = self.reliability_estimator.predict(
                    tuple(records),
                    predicted_response,
                    standard_deviation,
                    calibrated_uncertainty,
                )
                accepted = passes_gate(design, evidence, self.domain, self.thresholds)
                candidate_audits.append(CandidateAudit(design.copy(), accepted, evidence, accepted))
                if not accepted:
                    continue
                objective_samples = self.objective_projector.project(design, calibrated_members)
                candidate_forecasts.append(
                    AcquisitionPrediction(
                        design=design.copy(),
                        objective_samples=objective_samples,
                        convergence_probability=evidence.convergence_probability,
                        predicted_hf_cost=evidence.predicted_hf_cost,
                    )
                )
            if not candidate_forecasts:
                empty_pools += 1
                if empty_pools >= self.config.max_empty_pools:
                    raise RuntimeError("No candidate passed the physical gate in the registered pool limit.")
                continue
            empty_pools = 0
            remaining = self.config.high_fidelity_budget - self.repository.high_fidelity_count
            selected = select_dual_infill(
                candidate_forecasts,
                archive_objectives,
                np.asarray(self.config.reference_point, dtype=float),
                np.asarray(self.config.objective_scales, dtype=float),
                self.config.eta,
                self.config.epsilon_0,
                slots=min(2, remaining),
            )
            high_before = self.repository.high_fidelity_count
            for item in selected:
                self._evaluate(item.design, 3, material, loading)
            high_after = self.repository.high_fidelity_count
            if high_after - high_before != len(selected):
                raise RuntimeError("Every selected F3 attempt must consume one HF budget call.")
            iterations.append(
                IterationAudit(
                    iteration=iteration,
                    pool_index=pool_index - 1,
                    high_count_before=high_before,
                    high_count_after=high_after,
                    candidate_audit=tuple(candidate_audits),
                    selected_designs=tuple(item.design.copy() for item in selected),
                )
            )
        if self.repository.high_fidelity_count != self.config.high_fidelity_budget:
            raise RuntimeError("The optimizer did not terminate exactly at the locked HF budget.")
        return OptimizationRun(tuple(self.repository.archive), self.repository, tuple(iterations))
