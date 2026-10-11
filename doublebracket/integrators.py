"""
General-purpose integrators, used for comparison: RK4, 3-stage Lobatto IIIC and SciPy's DOP853 (the reference solution).

They integrate the continuous equations dmu/dt = ad*_{xi + phi(xi)} mu (models.lie_poisson_rhs) for the momentum mu, flattened to a vector of R^d. None of them preserves the coadjoint orbits.
"""

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from doublebracket.array_types import Covec, CovecArray, FlatState, FlatStateArray
from doublebracket.interfaces import Model
from doublebracket.models import lie_poisson_rhs

type VectorField = Callable[[FlatState], FlatState]  # the right-hand side f of an autonomous ODE dy/dt = f(y)


def rk4(f: VectorField, y0: FlatState, h: float, n_steps: int) -> FlatStateArray:
    """The classical 4th-order Runge-Kutta method."""
    y = np.empty((n_steps + 1, len(y0)))
    y[0] = y0
    for k in range(n_steps):
        k1 = f(y[k])
        k2 = f(y[k] + h / 2 * k1)
        k3 = f(y[k] + h / 2 * k2)
        k4 = f(y[k] + h * k3)
        y[k + 1] = y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


# Butcher tableau of the 3-stage Lobatto IIIC method (order 4), c = (0, 1/2, 1)
LOBATTO_IIIC_A = np.array(
    [
        [1 / 6, -1 / 3, 1 / 6],
        [1 / 6, 5 / 12, -1 / 12],
        [1 / 6, 2 / 3, 1 / 6],
    ]
)
LOBATTO_IIIC_B = np.array([1 / 6, 2 / 3, 1 / 6])


def lobatto_iiic(f: VectorField, y0: FlatState, h: float, n_steps: int) -> FlatStateArray:
    """
    The 3-stage Lobatto IIIC implicit Runge-Kutta method (order 4). The stage slopes K_1, K_2, K_3 solve
        K_i = f(y_k + h sum_j a_ij K_j),
    which we solve with fsolve, starting from explicit Euler-type predictions. Then y_{k+1} = y_k + h sum_i b_i K_i.
    """
    y = np.empty((n_steps + 1, len(y0)))
    y[0] = y0
    for k in range(n_steps):
        K1 = f(y[k])
        K2 = f(y[k] + h / 2 * K1)
        K3 = f(y[k] + h * K2)
        K = fsolve(_lobatto_iiic_stage_residual, np.concatenate([K1, K2, K3]), args=(f, y[k], h))
        y[k + 1] = y[k] + h * LOBATTO_IIIC_B @ K.reshape(3, -1)
    return y


def _lobatto_iiic_stage_residual(K_flat: np.ndarray, f: VectorField, y: FlatState, h: float) -> np.ndarray:
    """K_i - f(Y_i), i = 1, 2, 3, with the stage values Y_i = y + h sum_j a_ij K_j."""
    K = K_flat.reshape(3, -1)
    Y = y + h * LOBATTO_IIIC_A @ K
    return (K - np.array([f(Y_i) for Y_i in Y])).ravel()


def dop853(f: VectorField, y0: FlatState, h: float, n_steps: int, rtol: float, atol: float) -> FlatStateArray:
    """SciPy's DOP853 (explicit Runge-Kutta of order 8 with adaptive steps), with output on the grid t_k = k h."""
    t_eval = h * np.arange(n_steps + 1)
    solution = solve_ivp(lambda t, y: f(y), (0.0, t_eval[-1]), y0, method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(f"DOP853 failed: {solution.message}")
    return solution.y.T


def integrate(
    integrator: Callable[..., FlatStateArray],
    model: Model,
    M0: Covec,
    h: float,
    n_steps: int,
    **options: float,
) -> CovecArray:
    """
    Apply one of the integrators above to the continuous equations of the model, from M0. The momentum is flattened for the integrator and reshaped back, so the output has shape (n_steps + 1, *M0.shape). The options are passed to the integrator (rtol and atol for dop853).
    """

    def f(mu_flat: FlatState) -> FlatState:
        return lie_poisson_rhs(model, mu_flat.reshape(M0.shape)).ravel()

    M_flat = integrator(f, M0.ravel(), h, n_steps, **options)
    return M_flat.reshape(n_steps + 1, *M0.shape)
