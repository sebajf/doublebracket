"""
The examples: initial conditions, parameters and simulation settings.

To define a new example, create one from scratch or derive it from an existing one:
    Small_step = RigidBodyExample(Itensor=np.diag([1.0, 2.0, 3.0]), alpha=0.1, Omega0=np.array([1.0, 0.0, 0.1]), h=0.01)
    Variant = replace(Scenario_A, t_span=(0.0, 250.0))
The variable names (Small_step, Variant, ...) are used automatically as names of the output folders, plots/<name>/.

The input is checked once, when an example is created (__post_init__); the rest of the code assumes valid input.
"""

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from doublebracket.array_types import Covec, Mat3x3, TimeArray, Vec
from doublebracket.models import RigidBodySystem, rigid_body, torus_lattice


@dataclass(frozen=True, kw_only=True)
class Example:
    """Settings shared by all examples. Subclasses define the model (a RigidBodySystem) and the initial momentum M0."""

    name: str = ""  # set automatically to the variable name, see the end of this file
    t_span: tuple[float, float] = (0.0, 10.0)
    h: float = 0.1
    # DDB exp: None for the exact (dexp^{-1})^*, or the order at which its series is truncated (see retractions.Exp)
    truncation_order: int | None = None
    # Tolerances of the DOP853 reference solution
    ref_rtol: float = 1e-12
    ref_atol: float = 1e-14
    # h-sweep figures: reference lines factor * h^order, given as (order, factor) pairs, in the plot of the final
    # momentum error of all methods, and one line per method (by method key) in the per-method plots
    hsweep_order_lines: tuple[tuple[int, float], ...] = ()
    hsweep_method_order_lines: dict[str, tuple[int, float]] = field(
        default_factory=lambda: {"RK4": (4, 1e-3), "Lobatto IIIC": (4, 1e-3)}
    )

    def __post_init__(self) -> None:
        if not self.t_span[1] > self.t_span[0]:
            raise ValueError(f"t_span must be increasing, got {self.t_span}")
        if not self.h > 0:
            raise ValueError(f"The step h must be positive, got {self.h}")

    @property
    def n_steps(self) -> int:
        """The number of steps, round(T / h) with T = t_span[1] - t_span[0]."""
        return round((self.t_span[1] - self.t_span[0]) / self.h)

    @property
    def times(self) -> TimeArray:
        """The time grid t_k = t_0 + k h, k = 0, ..., n_steps."""
        return self.t_span[0] + self.h * np.arange(self.n_steps + 1)


@dataclass(frozen=True, kw_only=True)
class RigidBodyExample(Example):
    """The free rigid body with inertia tensor Itensor, dissipation coefficient alpha and initial body angular velocity Omega0."""

    Itensor: Mat3x3 = field(default_factory=lambda: np.eye(3))
    alpha: float = 0.0
    Omega0: Vec = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self) -> None:
        super().__post_init__()
        if np.shape(self.Itensor) != (3, 3) or np.shape(self.Omega0) != (3,):
            raise ValueError("Itensor must have shape (3, 3) and Omega0 shape (3,)")
        _ = self.model  # building the model checks that the inertia tensor is symmetric positive definite

    @cached_property
    def model(self) -> RigidBodySystem:
        return rigid_body(self.Itensor, self.alpha)

    @property
    def M0(self) -> Covec:
        """The initial body angular momentum M0 = I Omega0."""
        return self.model.legendre(self.Omega0)


def lattice_inertia_tensors(side: int = 4) -> Float[ndarray, "N 3 3"]:
    """I_(a,b) = diag(1 + 0.1 a, 1.2 + 0.1 b, 0.6 + 0.05 (a + b)) at site (a, b), with index i = side*a + b."""
    return np.array(
        [np.diag([1 + 0.1 * a, 1.2 + 0.1 * b, 0.6 + 0.05 * (a + b)]) for a in range(side) for b in range(side)]
    )


def random_unit_momenta(n_sites: int, seed: int) -> Covec:
    """A reproducible initial condition: one Gaussian vector per site, normalized, so that every Casimir equals 1/2."""
    M0 = np.random.default_rng(seed).normal(size=(n_sites, 3))
    return M0 / np.linalg.norm(M0, axis=1)[:, None]


@dataclass(frozen=True, kw_only=True)
class LatticeExample(Example):
    """
    A side x side periodic lattice of rigid bodies with momentum coupling J (see models.torus_lattice). The state is the momentum M of shape (side^2, 3); row i = side*a + b is the body momentum at site (a, b).
    """

    side: int = 4
    I_list: Float[ndarray, "N 3 3"] = field(default_factory=lattice_inertia_tensors)
    J: Mat3x3 = field(default_factory=lambda: np.diag([0.5, 0.4, 0.8]))
    alpha: float = 0.0
    M0: Covec = field(default_factory=lambda: random_unit_momenta(16, 2026))

    def __post_init__(self) -> None:
        super().__post_init__()
        N = self.side**2
        if self.side < 3:
            raise ValueError("side must be at least 3, so that each site has four distinct neighbors")
        if np.shape(self.I_list) != (N, 3, 3) or np.shape(self.J) != (3, 3) or np.shape(self.M0) != (N, 3):
            raise ValueError(f"I_list, J and M0 must have shapes ({N}, 3, 3), (3, 3) and ({N}, 3)")
        _ = self.model  # building the model checks that the Lagrangian is regular

    @cached_property
    def model(self) -> RigidBodySystem:
        return torus_lattice(self.side, self.I_list, self.J, self.alpha)


###########################################################
# The examples


# Rigid body (Section 6 of the paper)
Scenario_A = RigidBodyExample(
    Itensor=np.diag([1.0, 1.3, 0.5]),
    alpha=0.5,
    Omega0=np.array([0.0, 0.05, 1.0]),
    t_span=(0.0, 30.0),
    h=0.1,
    truncation_order=2,  # as in the paper
    hsweep_order_lines=((1, 1e-1), (4, 1e-2)),
    hsweep_method_order_lines={"RK4": (4, 1e-2), "Lobatto IIIC": (4, 1e-3), "DDB exp": (1, 1e-5), "DDB cay": (1, 1e-5)},
)
Scenario_B = replace(Scenario_A, t_span=(0.0, 250.0), hsweep_order_lines=())
Scenario_B_long = replace(Scenario_B, t_span=(0.0, 1000.0))  # tip: disable Lobatto IIIC, it takes too long
Scenario_C = replace(Scenario_B, Itensor=np.diag([1.0, 1.0, 0.5]))  # two equal largest moments of inertia

# Longer runs, for videos
For_Video_A = replace(Scenario_A, t_span=(0.0, 150.0), Omega0=np.array([0.0, 0.001, 1.0]))
For_Video_C = replace(Scenario_C, t_span=(0.0, 150.0), Omega0=np.array([0.0, 0.001, 1.0]))
For_Video_C1 = replace(For_Video_C, Omega0=np.array([0.001, 0.0, 1.0]))
For_Video_C2 = replace(For_Video_C, Omega0=np.array([-0.0007, 0.0007, 1.0]))
For_omega = replace(
    Scenario_A, Itensor=np.diag([1.0, 1.49, 0.5]), t_span=(0.0, 1000.0), Omega0=np.array([-0.0000001, 0.0, 1.0])
)
FlipSide = RigidBodyExample(Omega0=np.array([0.0, 0.08, 1.0]), h=1.0)

# 4x4 periodic lattice of rigid bodies, I_list and J as above
Lattice_A = LatticeExample(
    alpha=0.2,
    t_span=(0.0, 10.0),
    h=0.1,
    ref_rtol=1e-13,
    ref_atol=1e-13,
    truncation_order=2,  # as for the rigid body
    hsweep_order_lines=((1, 5.0), (4, 1e-2)),
)
Lattice_B = replace(Lattice_A, t_span=(0.0, 200.0))
Lattice_B_T400 = replace(Lattice_B, t_span=(0.0, 400.0))


##############################################################
# Don't define examples below this line
for _name, _value in list(globals().items()):
    if isinstance(_value, Example):
        object.__setattr__(_value, "name", _name)
