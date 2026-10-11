"""
Tests for all the compared methods, on short runs of the rigid body and the lattice.

The regression tests store the output of 5-step runs (in tests/test_methods/). If a change in the code modifies these results, the test fails: investigate before regenerating the data with --force-regen.
"""

from dataclasses import replace

import numpy as np
import pytest

from doublebracket import examples
from doublebracket.methods import METHODS

SHORT_RUNS = {
    "rigid": replace(examples.Scenario_A, h=6.0),  # 5 steps with a large h
    "lattice": replace(examples.Lattice_A, t_span=(0.0, 0.25), h=0.05),  # 5 steps
}


@pytest.mark.parametrize("method_key", METHODS)
@pytest.mark.parametrize("run", SHORT_RUNS)
def test_time_grid_and_initial_condition(run: str, method_key: str) -> None:
    ex = SHORT_RUNS[run]
    trajectory = METHODS[method_key].run(ex)
    assert np.array_equal(trajectory.t, ex.times) and len(ex.times) == 6
    assert trajectory.M.shape == (6, *ex.M0.shape)
    assert np.array_equal(trajectory.M[0], ex.M0)
    if method_key.startswith("DDB"):
        assert trajectory.xi.shape == (5, *ex.M0.shape)


@pytest.mark.parametrize("method_key", METHODS)
def test_regression(ndarrays_regression, method_key: str) -> None:
    """Rigid body. Omega_k = I^{-1} M_k."""
    ex = SHORT_RUNS["rigid"]
    trajectory = METHODS[method_key].run(ex)
    Omega = np.array([ex.model.inverse_legendre(Mk) for Mk in trajectory.M])
    ndarrays_regression.check({"t": trajectory.t, "Omega": Omega, "M": trajectory.M})


@pytest.mark.parametrize("method_key", METHODS)
def test_lattice_regression(ndarrays_regression, method_key: str) -> None:
    """Lattice. xi_k = LL M_k (the inverse Legendre transform)."""
    ex = SHORT_RUNS["lattice"]
    trajectory = METHODS[method_key].run(ex)
    xi = np.array([ex.model.inverse_legendre(Mk) for Mk in trajectory.M])
    ndarrays_regression.check({"t": trajectory.t, "xi": xi, "M": trajectory.M})
