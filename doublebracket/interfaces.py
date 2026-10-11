"""
The ingredients of the DDB method (Section 5 of the paper), as Python protocols.

The code is organized in four layers:
- group (LieGroup, implemented in so3.py),
- retraction (Retraction, implemented in retractions.py),
- model (Model, implemented in models.py),
- solver (ddb.py), which uses only the methods listed here.

To add a new example, implement these three protocols. Elements of g and g* are NumPy arrays whose shape is fixed by the group (for SO(3)^N: (N, 3), or (3,) for one body), and the pairing <mu, xi> is the sum of the componentwise products. Group elements are matrices, or stacks of matrices for a product group, so that the product in G is the matrix product (the @ operator).
"""

from typing import Protocol

from doublebracket.array_types import Covec, GroupElement, SiteScalars, Vec


class LieGroup(Protocol):
    def coadjoint_action(self, g: GroupElement, mu: Covec) -> Covec:
        """Ad*_g mu, where Ad*_g: g* -> g* is the dual of Ad_g. The coadjoint action of g on g* is Ad*_{g^{-1}}, and Ad*_{gh} = Ad*_h Ad*_g."""
        ...

    def ad_star(self, xi: Vec, mu: Covec) -> Covec:
        """ad*_xi mu, where ad*_xi is the dual of ad_xi = [xi, .]. Used only by the continuous equations."""
        ...


class Retraction(Protocol):
    name: str  # used in labels and file names

    def tau(self, x: Vec) -> GroupElement:
        """The retraction map tau: g -> G."""
        ...

    def dtau_inv_star(self, x: Vec, mu: Covec) -> Covec:
        """(dtau_x^{-1})^* mu, the dual of the inverse of the right trivialized tangent of tau at x, as defined in Section 4 of the paper."""
        ...


class Model(Protocol):
    """
    A forced Euler-Poincare system with double-bracket dissipation: a reduced Lagrangian l on g, assumed hyperregular, and a dissipation map phi: g -> g.

    The DDB method uses only legendre, inverse_legendre (for the starting guess of the stage equations) and phi. The continuous equations and the diagnostics use the rest.
    """

    group: LieGroup

    def lagrangian(self, xi: Vec) -> float:
        """The reduced Lagrangian l(xi)."""
        ...

    def legendre(self, xi: Vec) -> Covec:
        """The Legendre transform xi -> dl/dxi(xi)."""
        ...

    def inverse_legendre(self, mu: Covec) -> Vec:
        """The inverse of the Legendre transform, mu -> dH/dmu(mu)."""
        ...

    def hamiltonian(self, mu: Covec) -> float:
        """The Hamiltonian H(mu) = <mu, xi> - l(xi), with xi = dH/dmu(mu)."""
        ...

    def phi(self, xi: Vec) -> Vec:
        """The dissipation map phi: g -> g."""
        ...

    def casimirs(self, mu: Covec) -> float | SiteScalars:
        """The Casimir functions at mu (a single number, or one per factor of a product group)."""
        ...
