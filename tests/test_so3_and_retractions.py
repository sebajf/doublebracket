"""Tests for the group layer (so3.py) and the retraction layer (retractions.py)."""

import numpy as np
import pytest
from scipy.linalg import expm

from doublebracket.interfaces import Retraction
from doublebracket.retractions import Cayley, Exp, b_exp, cayley, rodrigues
from doublebracket.so3 import SO3, cross, hat, vee

RETRACTIONS: list[Retraction] = [Cayley(), Exp()]
rng = np.random.default_rng(0)


def test_hat_vee_and_cross() -> None:
    x, y = rng.normal(size=(2, 16, 3))
    assert np.allclose(hat(x) @ y[..., None], cross(x, y)[..., None], rtol=0, atol=1e-15)
    assert np.allclose(cross(x, y), np.cross(x, y), rtol=0, atol=1e-15)
    assert np.array_equal(vee(hat(x)), x)
    assert np.array_equal(hat(x), -np.swapaxes(hat(x), -1, -2))


def test_coadjoint_action() -> None:
    """Ad*_{gh} = Ad*_h Ad*_g, and <Ad*_g mu, x> = <mu, Ad_g x> with Ad_g x = g x."""
    G = SO3()
    g, h = rodrigues(rng.normal(size=(2, 5, 3)))
    mu, x = rng.normal(size=(2, 5, 3))
    assert np.allclose(G.coadjoint_action(g @ h, mu), G.coadjoint_action(h, G.coadjoint_action(g, mu)))
    assert np.allclose(np.sum(G.coadjoint_action(g, mu) * x, axis=-1), np.sum(mu * (g @ x[..., None])[..., 0], axis=-1))


def test_ad_star() -> None:
    """<ad*_xi mu, eta> = <mu, [xi, eta]>."""
    xi, mu, eta = rng.normal(size=(3, 3))
    assert np.dot(SO3().ad_star(xi, mu), eta) == pytest.approx(np.dot(mu, cross(xi, eta)))


@pytest.mark.parametrize("scale", [0.0, 1e-9, 0.1, 2.0, 6.0])
def test_closed_forms_of_exp_and_cay(scale: float) -> None:
    x = scale * rng.normal(size=(4, 3))
    X = hat(x)
    assert np.allclose(rodrigues(x), np.array([expm(Xi) for Xi in X]), rtol=0, atol=1e-12)
    expected_cay = np.linalg.solve(np.eye(3) - X / 2, np.eye(3) + X / 2)
    assert np.allclose(cayley(x), expected_cay, rtol=0, atol=1e-14)


@pytest.mark.parametrize("retraction", RETRACTIONS, ids=lambda r: r.name)
def test_tau_is_a_rotation(retraction: Retraction) -> None:
    R = retraction.tau(3 * rng.normal(size=(100, 3)))
    assert np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), rtol=0, atol=1e-14)
    assert np.allclose(np.linalg.det(R), 1, rtol=0, atol=1e-14)


@pytest.mark.parametrize("retraction", RETRACTIONS, ids=lambda r: r.name)
def test_dtau_inv_star_is_dual_of_inverse_right_trivialized_tangent(retraction: Retraction) -> None:
    """
    Compute the right trivialized tangent dtau_x by central finite differences, from T_x tau(eta) = hat(dtau_x eta) tau(x), and check <(dtau_x^{-1})^* mu, dtau_x eta> = <mu, eta>.
    """
    eps = 1e-6
    for scale in (1e-3, 0.3, 2.0):
        x = scale * rng.normal(size=3)
        dtau = np.zeros((3, 3))
        for j, e in enumerate(np.eye(3)):
            derivative = (retraction.tau(x + eps * e) - retraction.tau(x - eps * e)) / (2 * eps)
            dtau[:, j] = vee(derivative @ retraction.tau(x).T)
        B = np.column_stack([retraction.dtau_inv_star(x, e) for e in np.eye(3)])
        assert np.allclose(B.T @ dtau, np.eye(3), rtol=0, atol=1e-8)


@pytest.mark.parametrize("retraction", RETRACTIONS, ids=lambda r: r.name)
def test_dtau_inv_star_transport(retraction: Retraction) -> None:
    """tau(x)^T (dtau_x^{-1})^* mu = (dtau_{-x}^{-1})^* mu, that is, (dtau_{-x}^{-1})^* = Ad*_{tau(x)} (dtau_x^{-1})^*."""
    for scale in (1e-4, 0.1, 1.0, 3.0):
        x = scale * rng.normal(size=(5, 3))
        mu = rng.normal(size=(5, 3))
        lhs = SO3().coadjoint_action(retraction.tau(x), retraction.dtau_inv_star(x, mu))
        assert np.allclose(lhs, retraction.dtau_inv_star(-x, mu), rtol=0, atol=1e-14)


def test_exact_dexp_agrees_with_truncated_series() -> None:
    x, mu = np.array([0.03, -0.02, 0.05]), np.array([1.0, 2.0, 3.0])
    exact = Exp().dtau_inv_star(x, mu)
    assert np.allclose(exact, Exp(order=10).dtau_inv_star(x, mu), rtol=0, atol=1e-15)
    assert not np.allclose(exact, Exp(order=2).dtau_inv_star(x, mu), rtol=0, atol=1e-10)


def test_b_exp_is_continuous() -> None:
    """The series (θ < 1) and the closed form (θ >= 1) agree at θ = 1, and b(0) = 1/12."""
    assert b_exp(np.array([1 - 1e-12])) == pytest.approx(b_exp(np.array([1.0])), rel=1e-12)
    assert b_exp(np.array([0.0]))[0] == pytest.approx(1 / 12, rel=1e-15)
