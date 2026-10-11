"""
The numerical methods compared in the paper, with their names and plot styles.

Each method takes an example and returns a Trajectory on the example's time grid.
"""

from collections.abc import Callable
from dataclasses import dataclass

from doublebracket.array_types import CovecArray, TimeArray, VecArray
from doublebracket.ddb import ddb_solve
from doublebracket.examples import Example
from doublebracket.integrators import dop853, integrate, lobatto_iiic, rk4
from doublebracket.interfaces import Retraction
from doublebracket.retractions import Cayley, Exp


@dataclass(frozen=True)
class Trajectory:
    """
    The output of a method: the times t_k and momenta M_k. For the DDB methods, also the stage velocities xi_0, ..., xi_{n-1} (one fewer than the times) and the retraction, which together define the group increments w_k = tau(h xi_k) (see ddb.reconstruct).
    """

    t: TimeArray
    M: CovecArray
    xi: VecArray | None = None
    retraction: Retraction | None = None


@dataclass(frozen=True)
class Method:
    full_name: str
    label: str  # for plot legends
    short_name: str  # for file names
    linespec: str  # matplotlib format string
    run: Callable[[Example], Trajectory]


def _general_purpose(integrator: Callable, **options: float) -> Callable[[Example], Trajectory]:
    def run(ex: Example) -> Trajectory:
        return Trajectory(ex.times, integrate(integrator, ex.model, ex.M0, ex.h, ex.n_steps, **options))

    return run


def _dop853(ex: Example) -> Trajectory:
    M = integrate(dop853, ex.model, ex.M0, ex.h, ex.n_steps, rtol=ex.ref_rtol, atol=ex.ref_atol)
    return Trajectory(ex.times, M)


def _ddb(retraction_for: Callable[[Example], Retraction]) -> Callable[[Example], Trajectory]:
    def run(ex: Example) -> Trajectory:
        retraction = retraction_for(ex)
        M, xi = ddb_solve(ex.model, retraction, ex.M0, ex.h, ex.n_steps)
        return Trajectory(ex.times, M, xi, retraction)

    return run


METHODS: dict[str, Method] = {
    "DOP853": Method("SciPy's DOP853, tight tolerances", "DOP853 (reference)", "built_in", "k-", _dop853),
    "RK4": Method("Runge-Kutta 4th order", "RK4", "rk4", "b--", _general_purpose(rk4)),
    "Lobatto IIIC": Method(
        "Lobatto IIIC (3-stage)", "Lobatto IIIC", "lobatto_iiic", "m-.", _general_purpose(lobatto_iiic)
    ),
    "DDB exp": Method(
        "Discrete double-bracket, exp retraction",
        "DDB (exp)",
        "ddb_exp",
        "g-.",
        _ddb(lambda ex: Exp(order=ex.truncation_order)),
    ),
    "DDB cay": Method(
        "Discrete double-bracket, Cayley retraction", "DDB (cay)", "ddb_cay", "r:", _ddb(lambda ex: Cayley())
    ),
}
REFERENCE = "DOP853"  # the method used as reference solution
