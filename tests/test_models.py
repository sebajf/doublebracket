"""Tests for the model layer (models.py) and the examples."""

import numpy as np
import pytest

from doublebracket import examples
from doublebracket.models import RigidBodySystem, energy_rate, lie_poisson_rhs, torus_adjacency, torus_lattice

rng = np.random.default_rng(1)


def random_inertia() -> np.ndarray:
    A = rng.normal(size=(3, 3))
    return A @ A.T + np.eye(3)


def test_legendre_lagrangian_hamiltonian_energy() -> None:
    """H(mu) = E_l(xi) = l(xi) for mu = dl/dxi(xi), and the inverse Legendre transform inverts it."""
    for _ in range(10):
        model = RigidBodySystem(inertia=random_inertia(), alpha=0.3)
        Omega = rng.normal(size=3)
        M = model.legendre(Omega)
        assert np.allclose(model.inverse_legendre(M), Omega)
        assert model.hamiltonian(M) == pytest.approx(model.lagrangian(Omega))
        assert model.energy(Omega) == pytest.approx(model.lagrangian(Omega))


def test_rigid_body_equations() -> None:
    """The continuous equations of the rigid body: I dOmega/dt = I Omega × Omega + alpha I Omega × ((I Omega) × Omega)."""
    ex = examples.Scenario_A
    I, alpha = ex.Itensor, ex.alpha
    Omega = rng.normal(size=3)
    expected = np.cross(I @ Omega, Omega) + alpha * np.cross(I @ Omega, np.cross(I @ Omega, Omega))
    assert np.allclose(lie_poisson_rhs(ex.model, I @ Omega), expected)


def test_rejects_non_regular_lagrangian() -> None:
    """The parameters I = Id, J = diag(1, 2, 3) of Arnaudon-Takao give an indefinite LL."""
    with pytest.raises(ValueError, match="positive definite"):
        torus_lattice(4, np.array([np.eye(3)] * 16), np.diag([1.0, 2.0, 3.0]), 0.2)


def test_examples_check_their_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        examples.RigidBodyExample(Omega0=np.zeros(2))
    with pytest.raises(ValueError, match="positive"):
        examples.RigidBodyExample(h=-0.1)


# Lattice


def test_adjacency_is_4_regular_torus() -> None:
    A = torus_adjacency(4)
    assert np.array_equal(A, A.T)
    assert np.all(A.sum(axis=1) == 4)
    assert np.all(np.diag(A) == 0)
    assert A[4 * 1 + 2, 4 * 2 + 2] == 1 and A[4 * 0 + 3, 4 * 0 + 0] == 1  # (1,2)~(2,2), (0,3)~(0,0)


def test_lattice_hamiltonian_matrix() -> None:
    """II^{-1} = LL = blockdiag(I_i^{-1}) - 1/4 A ⊗ J, with the spectrum stated in the lattice note."""
    ex = examples.Lattice_A
    LL = -0.25 * np.kron(torus_adjacency(4), ex.J)
    for i, Itensor in enumerate(ex.I_list):
        LL[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += np.linalg.inv(Itensor)
    assert np.allclose(ex.model.inverse_inertia, LL, rtol=0, atol=1e-14)

    eigenvalues = np.linalg.eigvalsh(LL)
    assert eigenvalues.min() > 4 / 15  # site-wise regularity criterion
    assert eigenvalues.min() == pytest.approx(0.329731, abs=1e-6)
    assert eigenvalues.max() == pytest.approx(2.199055, abs=1e-6)


def test_lattice_initial_condition() -> None:
    ex = examples.Lattice_A
    assert np.allclose(ex.model.casimirs(ex.M0), 0.5, rtol=0, atol=1e-15)
    assert ex.model.hamiltonian(ex.M0) == pytest.approx(8.5418210263, abs=1e-10)


def test_lattice_velocity_sitewise_formula() -> None:
    """xi_i = I_i^{-1} mu_i - 1/4 J sum_{j ~ i} mu_j."""
    ex = examples.Lattice_A
    mu = rng.normal(size=(16, 3))
    A = torus_adjacency(4)
    expected = np.array([np.linalg.solve(ex.I_list[i], mu[i]) - 0.25 * ex.J @ (A[i] @ mu) for i in range(16)])
    assert np.allclose(ex.model.inverse_legendre(mu), expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("ex", [examples.Scenario_A, examples.Lattice_A], ids=lambda ex: ex.name)
def test_casimirs_and_energy_identity(ex: examples.Example) -> None:
    """Along the continuous flow, dC_i/dt = 0 and dH/dt = -alpha sum_i |mu_i × xi_i|^2."""
    model = ex.model
    for _ in range(10):
        mu = rng.normal(size=ex.M0.shape)
        dmu = lie_poisson_rhs(model, mu)
        assert np.allclose(np.sum(mu * dmu, axis=-1), 0, rtol=0, atol=1e-13)  # dC_i/dt
        xi = model.inverse_legendre(mu)
        assert energy_rate(model, mu) == pytest.approx(-model.alpha * np.sum(np.cross(mu, xi) ** 2), rel=1e-12)
