from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class VariableSpec:
    symbol: str
    lower: float
    upper: float
    unit: str


# Manuscript Eq. (1) and Table 1, in the exact printed order.
DESIGN_VARIABLES: tuple[VariableSpec, ...] = (
    VariableSpec("R_o", 520.0, 1400.0, "um"),
    VariableSpec("R_i", 180.0, 1360.0, "um"),
    VariableSpec("H_g", 80.0, 450.0, "um"),
    VariableSpec("L", 5.0, 90.0, "mm"),
    VariableSpec("alpha_o", -9.0e-4, 9.0e-4, "K^-1"),
    VariableSpec("k_o", 0.6, 1.8, "W m^-1 K^-1"),
    VariableSpec("rho_o", 980.0, 1420.0, "kg m^-3"),
    VariableSpec("alpha_i", -9.0e-4, 9.0e-4, "K^-1"),
    VariableSpec("k_i", 0.6, 1.8, "W m^-1 K^-1"),
    VariableSpec("rho_i", 980.0, 1420.0, "kg m^-3"),
    VariableSpec("E_1", 150.0, 1600.0, "MPa"),
    VariableSpec("E_2", 0.2, 120.0, "MPa"),
    VariableSpec("T_c", 45.0, 92.0, "degC"),
    VariableSpec("Delta_T", 8.0, 55.0, "degC"),
)


@dataclass(frozen=True)
class GateThresholds:
    """Locked constants printed below manuscript Eq. (21)."""

    p_min: float = 0.5
    tau_mech: float = 1.0e-4
    tau_dir: float = 1.0e-4
    tau_J: float = 1.0e-6
    tau_n: float = 1.0e-10
    uncertainty_max: float = 0.25
    curvature_min: float = 0.75
    curvature_max: float = 0.90
    pitch_min: float = 2.4
    pitch_max: float = 3.0
    turns_min: float = 5.0
    turns_max: float = 7.2
    shape_error_max: float = 0.10


ConstraintFunction = Callable[[np.ndarray], np.ndarray]


class DesignDomain:
    """The printed 14-D box plus explicit geometry-dependent constraints.

    The manuscript prints dynamic upper bounds for R_i and H_g but does not
    print the geometry-dependent expression for L or the remaining g_j.
    Additional constraints are therefore injected, never silently invented.
    """

    def __init__(self, extra_constraints: ConstraintFunction | None = None):
        self.specs = DESIGN_VARIABLES
        self.lower = np.asarray([item.lower for item in self.specs], dtype=float)
        self.upper = np.asarray([item.upper for item in self.specs], dtype=float)
        self.extra_constraints = extra_constraints

    @property
    def dimension(self) -> int:
        return len(self.specs)

    def validate_shape(self, design: np.ndarray) -> np.ndarray:
        value = np.asarray(design, dtype=float)
        if value.shape != (self.dimension,):
            raise ValueError(f"Expected one {self.dimension}-component design, got {value.shape}.")
        if not np.all(np.isfinite(value)):
            raise ValueError("Design components must be finite.")
        return value

    def normalize(self, design: np.ndarray) -> np.ndarray:
        value = self.validate_shape(design)
        return (value - self.lower) / (self.upper - self.lower)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        value = np.asarray(normalized, dtype=float)
        if value.shape != (self.dimension,):
            raise ValueError(f"Expected one {self.dimension}-component design, got {value.shape}.")
        return self.lower + value * (self.upper - self.lower)

    def constraint_values(self, design: np.ndarray) -> np.ndarray:
        """Return g_j(d) <= 0 values used by manuscript Eqs. (3) and (21)."""
        value = self.validate_shape(design)
        box = np.concatenate((self.lower - value, value - self.upper))
        r_o, r_i, h_g = value[:3]
        dynamic = np.asarray(
            [r_i - (r_o - 40.0), h_g - min(450.0, r_o)],
            dtype=float,
        )
        if self.extra_constraints is None:
            return np.concatenate((box, dynamic))
        extra = np.atleast_1d(np.asarray(self.extra_constraints(value), dtype=float))
        return np.concatenate((box, dynamic, extra))

    def feasible(self, design: np.ndarray, tolerance: float = 0.0) -> bool:
        return bool(np.all(self.constraint_values(design) <= tolerance))

    def sample(self, count: int, seed: int) -> np.ndarray:
        """Deterministic candidate source; sampling law is an operational completion."""
        from scipy.stats import qmc

        if count <= 0:
            return np.empty((0, self.dimension), dtype=float)
        sampler = qmc.LatinHypercube(d=self.dimension, seed=int(seed))
        accepted: list[np.ndarray] = []
        attempts = 0
        while len(accepted) < count:
            unit = sampler.random(n=max(int(count), 8))
            designs = self.lower + unit * (self.upper - self.lower)
            # Sample conditional intervals directly instead of clipping a
            # Latin-hypercube draw onto a constraint boundary.
            designs[:, 1] = self.lower[1] + unit[:, 1] * (
                designs[:, 0] - 40.0 - self.lower[1]
            )
            designs[:, 2] = self.lower[2] + unit[:, 2] * (
                np.minimum(450.0, designs[:, 0]) - self.lower[2]
            )
            for design in designs:
                if self.feasible(design):
                    accepted.append(design.copy())
                    if len(accepted) == count:
                        break
            attempts += len(designs)
            if attempts > max(10000, 1000 * count):
                raise RuntimeError("Unable to sample the complete registered design domain.")
        return np.vstack(accepted)

    def symbols(self) -> Iterable[str]:
        return (item.symbol for item in self.specs)
