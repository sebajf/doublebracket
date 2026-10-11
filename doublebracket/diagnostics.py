"""
Quantities computed along a trajectory M_0, ..., M_n, used to compare the methods.
"""

import numpy as np

from doublebracket.array_types import CovecArray, Mat3x3, ScalarArray
from doublebracket.models import RigidBodySystem, energy_rate


def energies(model: RigidBodySystem, M: CovecArray) -> ScalarArray:
    """The Hamiltonian H(M_k) at each time."""
    return np.array([model.hamiltonian(Mk) for Mk in M])


def casimir_drift(model: RigidBodySystem, M: CovecArray) -> ScalarArray:
    """max_i |C_i(M_k) - C_i(M_0)| at each time: the distance to the initial coadjoint orbit, measured by the Casimirs."""
    C = model.casimirs(M)  # shape (n,) or (n, N)
    return np.abs(C - C[0]).reshape(len(M), -1).max(axis=1)


def energy_dissipation_residual(model: RigidBodySystem, M: CovecArray, h: float) -> ScalarArray:
    """
    r_k = H(M_{k+1}) - H(M_k) - h dH/dt(M_k), where dH/dt is the rate of change of the Hamiltonian along the continuous equations. It is O(h^2) for a consistent method.
    """
    rates = np.array([energy_rate(model, Mk) for Mk in M[:-1]])
    return np.diff(energies(model, M)) - h * rates


def momentum_error(M: CovecArray, M_ref: CovecArray) -> ScalarArray:
    """||M_k - M_ref,k|| at each time (the Euclidean norm over all components)."""
    return np.linalg.norm((M - M_ref).reshape(len(M), -1), axis=1)


# Asymptotic behavior of the dissipative rigid body (Section 6 of the paper)


def largest_inertia_axes(Itensor: Mat3x3) -> np.ndarray:
    """The axes of the largest principal moment of inertia, for a diagonal inertia tensor. The trajectories tend to the set of minimum energy on the Casimir sphere, which is a pair of points if there is one such axis, and a great circle if there are two."""
    moments = np.diag(Itensor)
    return np.flatnonzero(moments == moments.max())


def distance_to_great_circle(M: CovecArray, radius: float, normal_axis: int) -> ScalarArray:
    """Distance from each M_k to the circle of the given radius in the coordinate plane orthogonal to normal_axis."""
    in_plane = np.delete(M, normal_axis, axis=-1)
    radial_offset = np.linalg.norm(in_plane, axis=-1) - radius
    return np.sqrt(M[..., normal_axis] ** 2 + radial_offset**2)
