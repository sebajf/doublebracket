"""
Solver layer: the discrete double-bracket (DDB) method, Algorithm 1 of the paper.

Given M_0, a step h and a number of steps n, the method computes the discrete momenta M_0, ..., M_n, with M_k ≈ mu(t_0 + k h), and the stage velocities xi_0, ..., xi_{n-1}. It uses only the protocols of interfaces.py, so it works for any group, retraction and model.
"""

import numpy as np
from scipy.optimize import fsolve

from doublebracket.array_types import Covec, CovecArray, GroupElement, GroupElementArray, Vec, VecArray
from doublebracket.interfaces import Model, Retraction


class StageSolverError(RuntimeError):
    """Raised when the stage equations (line 3 of Algorithm 1) cannot be solved, typically because h is too large."""


def solve_stage_equations(model: Model, retraction: Retraction, M: Covec, h: float, residual_tol: float = 1e-8) -> Vec:
    """
    Lines 2 and 3 of Algorithm 1: find xi such that
        (dtau_{h xi}^{-1})^* dl/dxi(xi) = M,
    starting from the inverse Legendre transform of M, which is O(h)-close to the solution. The unknown is a vector in R^n (n = dim g) with no constraints, so we use SciPy's fsolve (MINPACK's hybrid Powell method), with its default tolerance.

    fsolve stops when its steps become small, so we check the residual of the solution: if it is larger than residual_tol (relative to max(1, |M|)), the method has failed, and we raise StageSolverError.
    """

    def residual(xi_flat: np.ndarray) -> np.ndarray:
        xi = xi_flat.reshape(M.shape)
        return (retraction.dtau_inv_star(h * xi, model.legendre(xi)) - M).ravel()

    xi_guess = model.inverse_legendre(M)
    xi_flat = fsolve(residual, xi_guess.ravel(), full_output=True)[0]  # full_output: no warning, we check the residual
    max_residual = np.abs(residual(xi_flat)).max()
    if max_residual > residual_tol * max(1.0, np.abs(M).max()):
        raise StageSolverError(f"DDB ({retraction.name}) stage equations, h = {h}: residual {max_residual:.2e}")
    return xi_flat.reshape(M.shape)


def ddb_step(model: Model, retraction: Retraction, M: Covec, h: float) -> tuple[Covec, Vec]:
    """
    One step of the DDB method, M_k -> M_{k+1} (lines 2 to 6 of Algorithm 1):
        solve (dtau_{h xi_k}^{-1})^* dl/dxi(xi_k) = M_k for xi_k,
        w_k = tau(h xi_k),
        v_k = tau(h phi(xi_k))   (this is phi_d(w_k)),
        M_{k+1} = Ad*_{v_k} Ad*_{w_k} M_k = Ad*_{w_k v_k} M_k.
    Returns M_{k+1} and the stage velocity xi_k.
    """
    G = model.group
    xi = solve_stage_equations(model, retraction, M, h)
    w = retraction.tau(h * xi)
    v = retraction.tau(h * model.phi(xi))
    M_next = G.coadjoint_action(v, G.coadjoint_action(w, M))
    return M_next, xi


def ddb_solve(model: Model, retraction: Retraction, M0: Covec, h: float, n_steps: int) -> tuple[CovecArray, VecArray]:
    """
    The DDB method (Algorithm 1): n_steps steps from M0.

    Returns the discrete momenta M_0, ..., M_n (shape (n + 1, *M0.shape)) and the stage velocities xi_0, ..., xi_{n-1} (shape (n, *M0.shape)). The velocities define the group increments w_k = tau(h xi_k), from which the configurations can be reconstructed (see reconstruct).
    """
    M = np.empty((n_steps + 1, *M0.shape))
    xi = np.empty((n_steps, *M0.shape))
    M[0] = M0
    for k in range(n_steps):
        M[k + 1], xi[k] = ddb_step(model, retraction, M[k], h)
    return M, xi


def reconstruct(retraction: Retraction, g0: GroupElement, xi: VecArray, h: float) -> GroupElementArray:
    """
    The configurations g_0, ..., g_n from g_0 and the velocities xi_0, ..., xi_{n-1}, by
        g_{k+1} = g_k w_k,    w_k = tau(h xi_k).
    For the DDB methods, xi_k are the stage velocities returned by ddb_solve, and the dissipative factor v_k does not enter. For other methods, one can use the exponential retraction and xi_k = dH/dmu(M_k).
    """
    g = np.empty((len(xi) + 1, *np.shape(g0)))
    g[0] = g0
    for k in range(len(xi)):
        g[k + 1] = g[k] @ retraction.tau(h * xi[k])
    return g
