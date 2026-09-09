from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .records import EvaluationRecord, FidelityRepository, immutable_float_array, nondominated_mask


@dataclass(frozen=True)
class LossWeights:
    potential: float = 1.0
    incompressibility: float = 1.0
    director_norm: float = 1.0
    energy: float = 1.0
    consolidation: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.potential,
            self.incompressibility,
            self.director_norm,
            self.energy,
            self.consolidation,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("All manuscript PIRN and EWC loss weights must be finite and nonnegative.")


@dataclass(frozen=True)
class TrainingReport:
    total: float
    data: float
    potential: float
    incompressibility: float
    director_norm: float
    energy: float
    ewc: float
    paired_cases: int


class _ResidualNet(nn.Module):
    def __init__(self, input_dimension: int, hidden: tuple[int, ...], output_dimension: int = 8):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dimension
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        layers.append(nn.Linear(previous, output_dimension))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


@dataclass
class _MemberState:
    model: _ResidualNet
    old_parameters: dict[str, torch.Tensor]
    importance: dict[str, torch.Tensor]


def model_input(record: EvaluationRecord) -> np.ndarray:
    """z plus geometry/material descriptors and the lower-level fingerprint."""
    return np.concatenate(
        (
            record.design,
            record.material_package,
            record.loading_state,
            record.fingerprint,
        )
    )


class AdjacentResidualPIRN:
    """Descriptor-level implementation of manuscript Eqs. (13)-(20).

    One ensemble is fitted for each adjacent correction F0->F1, F1->F2 and
    F2->F3. Network width, optimizer and ensemble size are operational
    completions because the final manuscript does not print them.
    """

    def __init__(
        self,
        response_scales: np.ndarray,
        energy_scale: float,
        *,
        hidden: tuple[int, ...] = (64, 64),
        ensemble_size: int = 5,
        epochs: int = 100,
        learning_rate: float = 1.0e-3,
        seed: int = 20260827,
        weights: LossWeights | None = None,
    ) -> None:
        scales = np.asarray(response_scales, dtype=float)
        if scales.shape != (8,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("response_scales must contain eight finite positive values.")
        if not np.isfinite(energy_scale) or energy_scale <= 0.0:
            raise ValueError("energy_scale must be finite and positive.")
        self.response_scales = immutable_float_array(scales)
        self.energy_scale = float(energy_scale)
        self.hidden = tuple(int(value) for value in hidden)
        self.ensemble_size = int(ensemble_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.weights = weights or LossWeights()
        self.members: dict[int, list[_MemberState]] = {}
        self._input_mean: dict[int, np.ndarray] = {}
        self._input_scale: dict[int, np.ndarray] = {}

    @property
    def input_mean(self) -> Mapping[int, np.ndarray]:
        return MappingProxyType(self._input_mean)

    @property
    def input_scale(self) -> Mapping[int, np.ndarray]:
        return MappingProxyType(self._input_scale)

    @staticmethod
    def _physics_terms(
        prediction: torch.Tensor,
        upper_response: torch.Tensor,
        diagnostics: torch.Tensor,
        energy_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Manuscript Eq. (18). Residual terms come from solver outputs, not
        # derivatives of the network. Columns are mech, dir, J and n.
        potential = torch.mean(diagnostics[:, 0] ** 2 + diagnostics[:, 1] ** 2)
        incompressibility = torch.mean(diagnostics[:, 2] ** 2)
        director_norm = torch.mean(diagnostics[:, 3] ** 2)
        predicted_energy = prediction[:, 6] + prediction[:, 7]
        observed_energy = upper_response[:, 6] + upper_response[:, 7]
        energy = torch.mean(((predicted_energy - observed_energy) / energy_scale) ** 2)
        return potential, incompressibility, director_norm, energy

    def _paired_arrays(
        self,
        repository: FidelityRepository,
        upper_fidelity: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[bytes, int]]]:
        pairs = repository.paired(upper_fidelity - 1, upper_fidelity)
        pairs = [
            (lower, upper)
            for lower, upper in pairs
            if lower.usable_for_residual_learning and upper.usable_for_residual_learning
        ]
        if not pairs:
            raise ValueError(f"No adjacent F{upper_fidelity - 1}->F{upper_fidelity} pairs are available.")
        inputs = np.vstack([model_input(lower) for lower, _ in pairs])
        lower_response = np.vstack([lower.response for lower, _ in pairs])
        upper_response = np.vstack([upper.response for _, upper in pairs])
        target = upper_response - lower_response
        diagnostics = np.vstack(
            [
                [
                    upper.diagnostics.mechanical_residual,
                    upper.diagnostics.director_residual,
                    upper.diagnostics.incompressibility_residual,
                    upper.diagnostics.director_norm_residual,
                ]
                for _, upper in pairs
            ]
        )
        upper_keys = [
            (np.asarray(upper.design, dtype="<f8").tobytes(), upper.fidelity)
            for _, upper in pairs
        ]
        return inputs, lower_response, upper_response, target, diagnostics, upper_keys

    def fit(
        self,
        repository: FidelityRepository,
        replay_records: Iterable[EvaluationRecord] | None = None,
    ) -> dict[int, TrainingReport]:
        replay_keys = None
        if replay_records is not None:
            replay_keys = {
                (np.asarray(record.design, dtype="<f8").tobytes(), record.fidelity)
                for record in replay_records
            }
        reports: dict[int, TrainingReport] = {}
        for upper_fidelity in (1, 2, 3):
            arrays = self._paired_arrays(repository, upper_fidelity)
            inputs, lower_response, upper_response, target, diagnostics, upper_keys = arrays
            if upper_fidelity not in self._input_mean:
                mean = inputs.mean(axis=0)
                scale = inputs.std(axis=0)
                scale[scale < 1.0e-12] = 1.0
                self._input_mean[upper_fidelity] = immutable_float_array(mean)
                self._input_scale[upper_fidelity] = immutable_float_array(scale)
            else:
                # Keep the original coordinate system so retained weights and
                # the EWC parameter anchor remain mathematically comparable.
                mean = self._input_mean[upper_fidelity]
                scale = self._input_scale[upper_fidelity]
            x = torch.as_tensor((inputs - mean) / scale, dtype=torch.float32)
            lower_tensor = torch.as_tensor(lower_response, dtype=torch.float32)
            upper_tensor = torch.as_tensor(upper_response, dtype=torch.float32)
            target_tensor = torch.as_tensor(target, dtype=torch.float32)
            diagnostics_tensor = torch.as_tensor(diagnostics, dtype=torch.float32)
            response_scales = torch.as_tensor(np.array(self.response_scales, copy=True), dtype=torch.float32)
            replay_rows = [
                index for index, key in enumerate(upper_keys)
                if replay_keys is None or key in replay_keys
            ]
            states = self.members.get(upper_fidelity, [])
            trained: list[_MemberState] = []
            member_terms: list[list[float]] = []
            for member_index in range(self.ensemble_size):
                torch.manual_seed(self.seed + 1009 * upper_fidelity + member_index)
                if member_index < len(states) and states[member_index].model.network[0].in_features == x.shape[1]:
                    state = states[member_index]
                    model = state.model
                    old_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
                    # Manuscript Eq. (20): estimate importance on the current
                    # replay set before the update, then hold it fixed.
                    if replay_rows:
                        replay_index = torch.as_tensor(replay_rows, dtype=torch.long)
                        importance = self._estimate_importance(
                            model,
                            x[replay_index],
                            target_tensor[replay_index],
                            response_scales,
                        )
                    else:
                        importance = {name: torch.zeros_like(value) for name, value in model.named_parameters()}
                else:
                    model = _ResidualNet(x.shape[1], self.hidden)
                    old_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
                    importance = {name: torch.zeros_like(value) for name, value in model.named_parameters()}
                generator = np.random.default_rng(self.seed + 7919 * upper_fidelity + member_index)
                bootstrap = generator.integers(0, len(x), size=len(x))
                batch = torch.as_tensor(bootstrap, dtype=torch.long)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                final_terms: list[torch.Tensor] = []
                model.train()
                for _ in range(self.epochs):
                    optimizer.zero_grad()
                    delta = model(x[batch])
                    corrected = lower_tensor[batch] + delta
                    # Manuscript Eq. (17): diagonal inverse response scales.
                    data = torch.mean(torch.sum(((target_tensor[batch] - delta) / response_scales) ** 2, dim=1))
                    potential, incompressibility, director_norm, energy = self._physics_terms(
                        corrected,
                        upper_tensor[batch],
                        diagnostics_tensor[batch],
                        self.energy_scale,
                    )
                    ewc = corrected.new_tensor(0.0)
                    for name, parameter in model.named_parameters():
                        ewc = ewc + torch.sum(importance[name] * (parameter - old_parameters[name]) ** 2)
                    total = (
                        data
                        + self.weights.potential * potential
                        + self.weights.incompressibility * incompressibility
                        + self.weights.director_norm * director_norm
                        + self.weights.energy * energy
                        + 0.5 * self.weights.consolidation * ewc
                    )
                    total.backward()
                    optimizer.step()
                    final_terms = [total, data, potential, incompressibility, director_norm, energy, ewc]
                model.eval()
                trained.append(
                    _MemberState(
                        model=model,
                        old_parameters={name: value.detach().clone() for name, value in model.named_parameters()},
                        importance=importance,
                    )
                )
                member_terms.append([float(term.detach()) for term in final_terms])
            self.members[upper_fidelity] = trained
            average = np.mean(np.asarray(member_terms, dtype=float), axis=0)
            reports[upper_fidelity] = TrainingReport(
                total=float(average[0]),
                data=float(average[1]),
                potential=float(average[2]),
                incompressibility=float(average[3]),
                director_norm=float(average[4]),
                energy=float(average[5]),
                ewc=float(average[6]),
                paired_cases=len(inputs),
            )
        return reports

    @staticmethod
    def _estimate_importance(
        model: _ResidualNet,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        response_scales: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Manuscript Eq. (20): mean squared per-sample data-loss gradients."""
        importance = {name: torch.zeros_like(value) for name, value in model.named_parameters()}
        model.train()
        for row in range(len(inputs)):
            model.zero_grad(set_to_none=True)
            prediction = model(inputs[row : row + 1])
            loss = torch.sum(((targets[row : row + 1] - prediction) / response_scales) ** 2)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    importance[name] += parameter.grad.detach() ** 2 / len(inputs)
        model.eval()
        return importance

    def predict_next(
        self,
        lower_record: EvaluationRecord,
        upper_fidelity: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if upper_fidelity not in self.members:
            raise RuntimeError(f"F{upper_fidelity - 1}->F{upper_fidelity} residual ensemble is not fitted.")
        inputs = model_input(lower_record)
        normalized = (inputs - self._input_mean[upper_fidelity]) / self._input_scale[upper_fidelity]
        tensor = torch.as_tensor(normalized[None, :], dtype=torch.float32)
        with torch.no_grad():
            corrections = np.stack(
                [state.model(tensor).numpy()[0] for state in self.members[upper_fidelity]],
                axis=0,
            )
        members = lower_record.response[None, :] + corrections
        return members.mean(axis=0), members.std(axis=0, ddof=0), members


class TriCriterionReplay:
    """Deterministic high-residual/near-front/sparse-regime replay selector.

    The three categories are manuscript-specified. Capacity and tie-breaking
    are operational completions because the paper does not print them.
    """

    def __init__(self, capacity: int = 96):
        if capacity < 3:
            raise ValueError("Replay capacity must be at least three.")
        self.capacity = int(capacity)

    def select(self, repository: FidelityRepository) -> list[EvaluationRecord]:
        by_design: dict[bytes, dict[int, EvaluationRecord]] = {}
        for record in repository.records():
            if record.fidelity == 0 or not record.usable_for_residual_learning:
                continue
            key = np.asarray(record.design, dtype="<f8").tobytes()
            by_design.setdefault(key, {})[record.fidelity] = record
        candidates = [levels[max(levels)] for levels in by_design.values()]
        all_latest = [
            levels[level]
            for levels in by_design.values()
            for level in (1, 2, 3)
            if level in levels
        ]
        if len(all_latest) <= self.capacity:
            return all_latest
        group = max(1, self.capacity // 3)
        residual_score = np.asarray(
            [
                record.diagnostics.mechanical_residual
                + record.diagnostics.director_residual
                + record.diagnostics.incompressibility_residual
                + record.diagnostics.director_norm_residual
                for record in candidates
            ]
        )
        high_residual = np.argsort(-residual_score, kind="stable")[:group]
        f3_indices = [
            index for index, record in enumerate(candidates)
            if record.fidelity == 3 and record.feasible
        ]
        near_front: np.ndarray
        if f3_indices:
            objective = np.vstack([candidates[index].objectives for index in f3_indices])
            span = np.ptp(objective, axis=0)
            span[span < 1.0e-12] = 1.0
            front = objective[nondominated_mask(objective)]
            distances = np.asarray(
                [
                    np.min(np.linalg.norm((front - value) / span, axis=1))
                    for value in objective
                ]
            )
            local = np.argsort(distances, kind="stable")[:group]
            near_front = np.asarray([f3_indices[index] for index in local], dtype=int)
        else:
            near_front = np.empty(0, dtype=int)
        descriptors = np.vstack(
            [
                np.concatenate((record.material_package, record.design))
                if len(record.material_package)
                else record.design
                for record in candidates
            ]
        )
        descriptor_span = np.ptp(descriptors, axis=0)
        descriptor_span[descriptor_span < 1.0e-12] = 1.0
        normalized = (descriptors - descriptors.min(axis=0)) / descriptor_span
        centroid = normalized.mean(axis=0)
        sparse = np.argsort(-np.linalg.norm(normalized - centroid, axis=1), kind="stable")[:group]
        ordered = list(dict.fromkeys(np.concatenate((high_residual, near_front, sparse)).tolist()))
        ordered.extend(index for index in range(len(candidates)) if index not in set(ordered))
        selected: list[EvaluationRecord] = []
        for index in ordered:
            key = np.asarray(candidates[index].design, dtype="<f8").tobytes()
            levels = by_design[key]
            for fidelity in (3, 2, 1):
                if fidelity in levels:
                    selected.append(levels[fidelity])
                    if len(selected) == self.capacity:
                        return selected
        return selected
