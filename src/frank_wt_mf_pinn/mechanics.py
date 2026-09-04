from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


def thermal_fraction(temperature: float | np.ndarray, transition: float, width: float) -> np.ndarray:
    """Manuscript Eq. (2): retained low-temperature fraction vartheta(T)."""
    if width <= 0.0:
        raise ValueError("Transition width must be positive.")
    temperature_value = np.asarray(temperature, dtype=float)
    return expit(-4.0 * (temperature_value - transition) / width)


def temperature_modulus(
    temperature: float | np.ndarray,
    modulus_1: float,
    modulus_2: float,
    transition: float,
    width: float,
) -> np.ndarray:
    """Manuscript Eq. (2): E(T)."""
    theta = thermal_fraction(temperature, transition, width)
    return modulus_2 + (modulus_1 - modulus_2) * theta


def step_length_tensor(
    director: np.ndarray,
    ell_parallel: float,
    ell_perpendicular: float,
) -> np.ndarray:
    """Manuscript Eq. (6), first line."""
    n = np.asarray(director, dtype=float)
    if n.ndim != 1:
        raise ValueError("Director must be a vector.")
    norm = np.linalg.norm(n)
    if not np.isclose(norm, 1.0, atol=1.0e-10):
        raise ValueError("Director must have unit norm.")
    identity = np.eye(len(n), dtype=float)
    return ell_perpendicular * identity + (ell_parallel - ell_perpendicular) * np.outer(n, n)


def anisotropy_ratio(temperature: float | np.ndarray, zeta_c: float, transition: float, width: float) -> np.ndarray:
    """Manuscript Eq. (6), second line."""
    return 1.0 + (zeta_c - 1.0) * thermal_fraction(temperature, transition, width)


def warner_terentjev_energy_density(
    deformation_gradient: np.ndarray,
    reference_step_length: np.ndarray,
    current_step_length: np.ndarray,
    shear_modulus: float,
) -> float:
    """Manuscript Eq. (7), with the tensor ordering printed in the paper."""
    deformation = np.asarray(deformation_gradient, dtype=float)
    ell_0 = np.asarray(reference_step_length, dtype=float)
    ell = np.asarray(current_step_length, dtype=float)
    if deformation.shape[0] != deformation.shape[1] or ell.shape != deformation.shape or ell_0.shape != deformation.shape:
        raise ValueError("F, ell_0 and ell must be square tensors of equal shape.")
    value = ell_0 @ deformation.T @ np.linalg.inv(ell) @ deformation
    return float(0.5 * shear_modulus * np.trace(value))


def frank_energy_density(
    director: np.ndarray,
    divergence: float,
    curl: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
) -> float:
    """Manuscript Eq. (8): separate splay, twist and bend constants."""
    n = np.asarray(director, dtype=float)
    curl_value = np.asarray(curl, dtype=float)
    if n.shape != curl_value.shape:
        raise ValueError("Director and curl must have equal shape.")
    splay = 0.5 * k1 * float(divergence) ** 2
    twist = 0.5 * k2 * float(np.dot(n, curl_value)) ** 2
    bend = 0.5 * k3 * float(np.linalg.norm(np.cross(n, curl_value))) ** 2
    return float(splay + twist + bend)


def constrained_potential(
    wt_energy: np.ndarray,
    frank_energy: np.ndarray,
    volume_weights: np.ndarray,
    jacobian: np.ndarray,
    director_norm_squared: np.ndarray,
    pressure: np.ndarray,
    director_multiplier: np.ndarray,
    external_work: float,
) -> float:
    """Discrete quadrature of manuscript Eq. (9)."""
    wt = np.asarray(wt_energy, dtype=float)
    frank = np.asarray(frank_energy, dtype=float)
    weights = np.asarray(volume_weights, dtype=float)
    integrand = (
        wt
        + frank
        + np.asarray(pressure, dtype=float) * (np.asarray(jacobian, dtype=float) - 1.0)
        + 0.5
        * np.asarray(director_multiplier, dtype=float)
        * (np.asarray(director_norm_squared, dtype=float) - 1.0)
    )
    if integrand.shape != weights.shape:
        raise ValueError("All quadrature arrays must have equal shape.")
    return float(np.sum(integrand * weights) - external_work)


@dataclass(frozen=True)
class NormalizedResiduals:
    mechanical: float
    director: float
    incompressibility: float
    director_norm: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.mechanical, self.director, self.incompressibility, self.director_norm],
            dtype=float,
        )


def normalized_solver_residuals(
    mechanical_stationarity: np.ndarray,
    external_force: np.ndarray,
    director_stationarity: np.ndarray,
    frank_director_stationarity: np.ndarray,
    jacobian_minus_one: np.ndarray,
    director_norm_minus_one: np.ndarray,
    volume_weights: np.ndarray,
    domain_volume: float,
    epsilon_0: float,
) -> NormalizedResiduals:
    """Manuscript Eq. (15), evaluated from solver outputs."""
    if epsilon_0 <= 0.0 or domain_volume <= 0.0:
        raise ValueError("epsilon_0 and domain_volume must be positive.")
    weights = np.asarray(volume_weights, dtype=float)
    j_error = np.asarray(jacobian_minus_one, dtype=float)
    n_error = np.asarray(director_norm_minus_one, dtype=float)
    if weights.shape != j_error.shape or weights.shape != n_error.shape:
        raise ValueError("Residual fields and volume weights must have equal shape.")
    mechanical = np.linalg.norm(mechanical_stationarity) / (np.linalg.norm(external_force) + epsilon_0)
    director = np.linalg.norm(director_stationarity) / (np.linalg.norm(frank_director_stationarity) + epsilon_0)
    incompressibility = np.sqrt(np.sum(weights * j_error**2) / domain_volume)
    director_norm = np.sqrt(np.sum(weights * n_error**2) / domain_volume)
    return NormalizedResiduals(float(mechanical), float(director), float(incompressibility), float(director_norm))
