"""
Model layer: systems of N rigid bodies on G = SO(3)^N, with a quadratic Lagrangian and double-bracket dissipation.

The free rigid body is the case N = 1 (states of shape (3,)). The lattice example is a network of N = side^2 bodies on a periodic square lattice, coupled through their momenta as in Arnaudon and Takao, "Networks of coadjoint orbits: from geometric to statistical mechanics", J. Geom. Mech. 11 (2019).

Besides the Model protocol (interfaces.py), this module has two functions that hold for any model: the continuous equations (lie_poisson_rhs) and the rate of change of the Hamiltonian along them (energy_rate).
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from doublebracket.array_types import Covec, Mat3x3, Matrix, SiteScalars, Vec
from doublebracket.interfaces import Model
from doublebracket.so3 import SO3, cross


@dataclass(frozen=True)
class RigidBodySystem:
    """
    N rigid bodies on SO(3)^N with the quadratic reduced Lagrangian
        l(xi) = 1/2 <xi, II xi>,
    where II is a symmetric positive definite 3N x 3N matrix (the inertia tensor I for a single body), and the double-bracket dissipation map
        phi(xi)_i = alpha mu_i × xi_i,    mu = II xi,
    with alpha >= 0. Arrays of shape (N, 3) are flattened to shape (3N,) before multiplying by II.

    Since II is positive definite, l is hyperregular: the Legendre transform mu = II xi is a linear isomorphism, and the Hamiltonian is H(mu) = 1/2 <mu, II^{-1} mu>. The Casimirs are C_i(mu) = |mu_i|^2 / 2, one per body, and along the continuous equations
        dH/dt = -alpha sum_i |mu_i × xi_i|^2 <= 0.
    """

    inertia: Matrix  # II
    alpha: float
    group: SO3 = field(default_factory=SO3)

    def __post_init__(self) -> None:
        """The only check on the input: II must be symmetric positive definite, so that l is (hyper)regular."""
        II = self.inertia
        if not (np.allclose(II, II.T, rtol=0, atol=1e-14 * np.abs(II).max()) and np.linalg.eigvalsh(II).min() > 0):
            raise ValueError("The inertia must be symmetric positive definite, otherwise the Lagrangian is not regular")

    @cached_property
    def inverse_inertia(self) -> Matrix:
        """II^{-1}."""
        return symmetric_inverse(self.inertia)

    def lagrangian(self, xi: Vec) -> float:
        """l(xi) = 1/2 <xi, II xi>, the kinetic energy."""
        return 0.5 * np.sum(xi * self.legendre(xi))

    def legendre(self, xi: Vec) -> Covec:
        """The Legendre transform mu = dl/dxi(xi) = II xi (M = I Omega for a single rigid body)."""
        return (self.inertia @ xi.ravel()).reshape(xi.shape)

    def inverse_legendre(self, mu: Covec) -> Vec:
        """xi = dH/dmu(mu) = II^{-1} mu (Omega = I^{-1} M for a single rigid body)."""
        return (self.inverse_inertia @ mu.ravel()).reshape(mu.shape)

    def energy(self, xi: Vec) -> float:
        """The Lagrangian energy E_l(xi) = <dl/dxi(xi), xi> - l(xi). It equals l(xi) because l is quadratic, and H(mu) = E_l(xi) for mu = dl/dxi(xi)."""
        return np.sum(self.legendre(xi) * xi) - self.lagrangian(xi)

    def hamiltonian(self, mu: Covec) -> float:
        """H(mu) = 1/2 <mu, II^{-1} mu>."""
        return 0.5 * np.sum(mu * self.inverse_legendre(mu))

    def phi(self, xi: Vec) -> Vec:
        """phi(xi)_i = alpha mu_i × xi_i, with mu = II xi; for a single body, phi(Omega) = alpha (I Omega) × Omega."""
        return self.alpha * cross(self.legendre(xi), xi)

    def casimirs(self, mu: Covec) -> float | SiteScalars:
        """C_i(mu) = |mu_i|^2 / 2, one per body (a single number for one body)."""
        return 0.5 * np.sum(mu**2, axis=-1)


def lie_poisson_rhs(model: Model, mu: Covec) -> Covec:
    """
    The forced Lie-Poisson equations with double-bracket dissipation,
        dmu/dt = ad*_{xi + phi(xi)} mu,    xi = dH/dmu(mu).
    For the rigid body this is dM/dt = M × (Omega + phi(Omega)).
    """
    xi = model.inverse_legendre(mu)
    return model.group.ad_star(xi + model.phi(xi), mu)


def energy_rate(model: Model, mu: Covec) -> float:
    """
    dH/dt at mu along the continuous equations: <dmu/dt, xi> = <ad*_{phi(xi)} mu, xi>, with xi = dH/dmu(mu).
    For RigidBodySystem this is -alpha sum_i |mu_i × xi_i|^2.
    """
    return np.sum(lie_poisson_rhs(model, mu) * model.inverse_legendre(mu))


# Constructors for the examples


def rigid_body(Itensor: Mat3x3, alpha: float) -> RigidBodySystem:
    """The free rigid body with inertia tensor I: l(Omega) = 1/2 <Omega, I Omega>, phi(Omega) = alpha (I Omega) × Omega."""
    return RigidBodySystem(inertia=Itensor, alpha=alpha)


def torus_adjacency(side: int) -> Float[ndarray, "N N"]:
    """
    Adjacency matrix of the side x side discrete torus Z_side x Z_side. Site (a, b) has index i = side*a + b, and its neighbors are (a ± 1, b) and (a, b ± 1), mod side. For side >= 3 the four neighbors are distinct.
    """
    n_sites = side * side
    A = np.zeros((n_sites, n_sites))
    for a in range(side):
        for b in range(side):
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                A[side * a + b, side * ((a + da) % side) + (b + db) % side] = 1.0
    return A


def extended_laplacian(I_list: Float[ndarray, "N 3 3"], J: Mat3x3, A: Float[ndarray, "N N"]) -> Matrix:
    """
    The 3N x 3N matrix
        LL = blockdiag(I_1^{-1}, ..., I_N^{-1}) - (D^{-1/2} A D^{-1/2}) ⊗ J,    D = diag(degrees),
    which defines the Hamiltonian H(mu) = 1/2 <mu, LL mu> of Arnaudon-Takao, eq. (4.10), with the normalization 1/(2 sqrt(d_i d_j)) of the coupling. For the 4-regular torus the coupling weight is 1/4, so
        xi_i = I_i^{-1} mu_i - (1/4) J sum_{j ~ i} mu_j.
    """
    inv_sqrt_degrees = A.sum(axis=1) ** -0.5
    LL = -np.kron(inv_sqrt_degrees[:, None] * A * inv_sqrt_degrees[None, :], J)
    for i, Itensor in enumerate(I_list):
        LL[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += np.linalg.inv(Itensor)
    return LL


def torus_lattice(side: int, I_list: Float[ndarray, "N 3 3"], J: Mat3x3, alpha: float) -> RigidBodySystem:
    """
    The side x side periodic lattice of rigid bodies, with inertia tensor I_list[side*a + b] at site (a, b) and coupling matrix J.

    The model is given by its Hamiltonian H(mu) = 1/2 <mu, LL mu> (see extended_laplacian), so II = LL^{-1}. The Lagrangian is (hyper)regular if and only if LL is positive definite.
    """
    LL = extended_laplacian(I_list, J, torus_adjacency(side))
    if not np.linalg.eigvalsh(LL).min() > 0:
        raise ValueError("LL must be positive definite, otherwise the Lagrangian is not regular")
    return RigidBodySystem(inertia=symmetric_inverse(LL), alpha=alpha)


def symmetric_inverse(S: Matrix) -> Matrix:
    """The inverse of a symmetric matrix S, symmetrized to remove rounding asymmetry."""
    inverse = np.linalg.inv(S)
    return (inverse + inverse.T) / 2
