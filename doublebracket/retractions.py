"""
Retraction layer: the Cayley map and the exponential map on SO(3) (or SO(3)^N, factor by factor), with the dual of the inverse of their right trivialized tangents.

For a retraction tau: g -> G, the right trivialized tangent dtau_x: g -> g is defined by T_x tau(eta) = dtau_x(eta) tau(x), and the DDB method needs the linear map mu -> (dtau_x^{-1})^* mu on g*. On SO(3), with hat(x) y = x × y and the dot product as pairing, the dual of a linear map is its transpose; in particular hat(x)^* = -hat(x).

As in so3.py, every function acts on the last axis, so arrays of shape (N, 3) are handled factor by factor.
"""

from dataclasses import dataclass
from functools import cache
from math import factorial

import numpy as np
from scipy.special import bernoulli

from doublebracket.array_types import Covec, GroupElement, Vec
from doublebracket.so3 import cross, hat


def cayley(x: Vec) -> GroupElement:
    """
    The Cayley map cay(x) = (I - hat(x)/2)^{-1} (I + hat(x)/2), computed without inverses as
        cay(x) = I + (4 hat(x) + 2 hat(x)^2) / (4 + |x|^2).
    """
    X = hat(x)
    denominator = (4 + np.sum(x**2, axis=-1))[..., None, None]
    return np.eye(3) + (4 * X + 2 * X @ X) / denominator


def rodrigues(x: Vec) -> GroupElement:
    """
    The exponential map exp(hat(x)), by Rodrigues' formula
        exp(hat(x)) = I + (sin θ / θ) hat(x) + ((1 - cos θ) / θ^2) hat(x)^2,    θ = |x|.
    The coefficients are evaluated with np.sinc(s) = sin(π s) / (π s), which is accurate also for small θ:
        sin θ / θ = sinc(θ / π),    (1 - cos θ) / θ^2 = 2 sin^2(θ/2) / θ^2 = sinc(θ / 2π)^2 / 2.
    """
    X = hat(x)
    theta = np.linalg.norm(x, axis=-1)[..., None, None]
    return np.eye(3) + np.sinc(theta / np.pi) * X + np.sinc(theta / (2 * np.pi)) ** 2 / 2 * X @ X


@dataclass(frozen=True)
class Cayley:
    """The Cayley retraction tau = cay on SO(3)."""

    name: str = "cay"

    def tau(self, x: Vec) -> GroupElement:
        return cayley(x)

    def dtau_inv_star(self, x: Vec, mu: Covec) -> Covec:
        """
        (dcay_x^{-1})^* mu = (I + hat(x)/2 + x ⊗ x / 4) mu = mu + (x × mu)/2 + x (x · mu)/4.

        This is the transpose of dcay_x^{-1} = I - hat(x)/2 + x ⊗ x / 4.
        """
        return mu + cross(x, mu) / 2 + x * np.sum(x * mu, axis=-1, keepdims=True) / 4


@dataclass(frozen=True)
class Exp:
    """
    The exponential retraction tau = exp on SO(3).

    (dexp_x^{-1})^* is the series sum_{j >= 0} (-1)^j B_j hat(x)^j / j!, with the Bernoulli numbers B_j (B_1 = -1/2). With order=None we use its exact closed form; with an integer order, the series truncated after the term j = order, which gives a different method. The paper's figures (rigid body and lattice) use order=2.
    """

    order: int | None = None
    name: str = "exp"

    def tau(self, x: Vec) -> GroupElement:
        return rodrigues(x)

    def dtau_inv_star(self, x: Vec, mu: Covec) -> Covec:
        """
        Exact (order=None):
            (dexp_x^{-1})^* mu = mu + (x × mu)/2 + b(|x|) x × (x × mu),    b(θ) = (1 - (θ/2) cot(θ/2)) / θ^2,
        valid for |x| < 2π. Truncated: the series up to j = order, applying hat(x) j times.
        """
        if self.order is None:
            theta = np.linalg.norm(x, axis=-1, keepdims=True)
            return mu + cross(x, mu) / 2 + b_exp(theta) * cross(x, cross(x, mu))

        result = mu
        term = mu
        for coefficient in _dexp_inv_star_coefficients(self.order)[1:]:
            term = cross(x, term)  # hat(x)^j mu, for j = 1, 2, ...
            result = result + coefficient * term
        return result


@cache
def _dexp_inv_star_coefficients(order: int) -> np.ndarray:
    """The coefficients (-1)^j B_j / j! of hat(x)^j in (dexp_x^{-1})^*, for j = 0, ..., order. SciPy uses the convention B_1 = -1/2."""
    B = bernoulli(order)
    return np.array([(-1) ** j * B[j] / factorial(j) for j in range(order + 1)])


# Taylor coefficients of b(θ) = sum_{n >= 1} c_n θ^(2n - 2), with c_n = -(-1)^n B_{2n} / (2n)!:
# c_1 = 1/12, c_2 = 1/720, c_3 = 1/30240, ... The series converges for θ < 2π.
_N_SERIES_TERMS = 12
_B = bernoulli(2 * _N_SERIES_TERMS)
_B_EXP_SERIES = np.array([-((-1) ** n) * _B[2 * n] / factorial(2 * n) for n in range(1, _N_SERIES_TERMS + 1)])


def b_exp(theta: np.ndarray) -> np.ndarray:
    """
    b(θ) = (1 - (θ/2) cot(θ/2)) / θ^2, the coefficient of hat(x)^2 in the exact (dexp_x^{-1})^*.

    The closed form loses accuracy for small θ (it subtracts numbers close to 1), so for θ < 1 we sum the Taylor series 1/12 + θ^2/720 + θ^4/30240 + ... instead (12 terms; the truncation error is below 1e-18 there).
    """
    small = theta < 1
    series = np.polynomial.polynomial.polyval(theta**2, _B_EXP_SERIES)
    t = np.where(small, 1.0, theta)  # avoids 0/0 in the branch that is not used
    closed_form = (1 - (t / 2) / np.tan(t / 2)) / t**2
    return np.where(small, series, closed_form)
