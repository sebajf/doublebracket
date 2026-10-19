"""Tests for the solver layer (ddb.py): properties of the DDB method stated in the paper."""

from dataclasses import replace

import numpy as np
import pytest

from doublebracket import examples
from doublebracket.ddb import StageSolverError, ddb_solve, ddb_step, reconstruct, solve_stage_equations
from doublebracket.interfaces import Retraction
from doublebracket.methods import METHODS
from doublebracket.retractions import Cayley, Exp
from doublebracket.so3 import spatial_momenta

RETRACTIONS: list[Retraction] = [Cayley(), Exp()]
EXAMPLES = [examples.Scenario_A, examples.Lattice_A]


@pytest.mark.parametrize("retraction", RETRACTIONS, ids=lambda r: r.name)
@pytest.mark.parametrize("ex", EXAMPLES, ids=lambda ex: ex.name)
def test_one_step(ex: examples.Example, retraction: Retraction) -> None:
    """The stage equations are solved, the Casimirs are preserved, and the energy decreases (alpha > 0)."""
    model, M0, h = ex.model, ex.M0, ex.h
    xi = solve_stage_equations(model, retraction, M0, h)
    residual = retraction.dtau_inv_star(h * xi, model.legendre(xi)) - M0
    assert np.abs(residual).max() < 1e-12

    M1, xi_step = ddb_step(model, retraction, M0, h)
    assert np.array_equal(xi_step, xi)
    assert np.allclose(model.casimirs(M1), model.casimirs(M0), rtol=0, atol=1e-14)
    assert model.hamiltonian(M1) < model.hamiltonian(M0)


def test_stage_solver_failure_raises() -> None:
    """With h = 6, some |h xi_i| would have to be close to 2 pi, where dexp is singular."""
    ex = examples.Lattice_A
    with pytest.raises(StageSolverError):
        solve_stage_equations(ex.model, Exp(), ex.M0, 6.0)


@pytest.mark.parametrize("retraction", RETRACTIONS, ids=lambda r: r.name)
def test_casimirs_preserved_along_a_run(retraction: Retraction) -> None:
    ex = replace(examples.Lattice_A, alpha=0.0)  # no dissipation: the energy is not monotone, but the Casimirs are still preserved
    M, _ = ddb_solve(ex.model, retraction, ex.M0, ex.h, 20)
    assert np.allclose(ex.model.casimirs(M), 0.5, rtol=0, atol=1e-14)


@pytest.mark.parametrize(("method_key", "expected_error"), [("DDB cay", 8.769e-02), ("DDB exp", 4.797e-02)])
def test_lattice_reference_numbers(method_key: str, expected_error: float) -> None:
    """Final error (max norm) at T = 10 against DOP853, with h = 0.1, as in the prototype of the lattice note; the energy is nonincreasing."""
    ex = replace(examples.Lattice_A, h=0.1)
    M_ref = METHODS["DOP853"].run(replace(ex, h=10.0)).M[-1]
    assert ex.model.hamiltonian(M_ref) == pytest.approx(5.5175912159, abs=1e-10)

    M = METHODS[method_key].run(ex).M
    assert np.abs(M[-1] - M_ref).max() == pytest.approx(expected_error, rel=1e-3)
    assert np.all(np.diff([ex.model.hamiltonian(Mk) for Mk in M]) <= 1e-15)


@pytest.mark.parametrize("retraction", RETRACTIONS, ids=lambda r: r.name)
def test_spatial_momenta_conserved_without_dissipation(retraction: Retraction) -> None:
    """
    For alpha = 0, each spatial momentum pi_i = R_i M_i is conserved exactly by the DDB method, with R reconstructed from the stage velocities: M_{k+1} = tau(h xi_k)^T M_k gives pi_{k+1} = R_k tau(h xi_k) M_{k+1} = pi_k. With dissipation, the spatial momenta change, but not their norms.
    """
    for ex, conserved in ((replace(examples.Lattice_A, alpha=0.0), True), (examples.Lattice_A, False)):
        M, xi = ddb_solve(ex.model, retraction, ex.M0, ex.h, 20)
        R = reconstruct(retraction, np.broadcast_to(np.eye(3), (16, 3, 3)), xi, ex.h)
        pi = spatial_momenta(R, M)
        if conserved:
            assert np.allclose(pi, pi[0], rtol=0, atol=1e-13)
        else:
            assert np.abs(pi - pi[0]).max() > 1e-3
            assert np.allclose(np.linalg.norm(pi, axis=-1), np.linalg.norm(M, axis=-1), rtol=0, atol=1e-13)


def test_stage_velocities_differ_from_inverse_legendre() -> None:
    """The stage velocity xi_k is O(h)-close to, but different from, dH/dmu(M_k)."""
    ex = examples.Scenario_A
    M, xi = ddb_solve(ex.model, Cayley(), ex.M0, ex.h, 5)
    velocities = np.array([ex.model.inverse_legendre(Mk) for Mk in M[:-1]])
    assert not np.allclose(xi, velocities, rtol=0, atol=1e-6)
    assert np.allclose(xi, velocities, rtol=0, atol=ex.h)
