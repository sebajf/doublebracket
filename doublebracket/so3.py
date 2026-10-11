"""
Group layer: the rotation group SO(3), and its powers SO(3)^N.

Elements of so(3) are identified with vectors x in R^3 through the hat map, and so(3)* with R^3 through the dot product. Then the bracket is the cross product, [x, y] = x × y.

Every function acts on the last axis (or the last two axes, for matrices), so the same code handles SO(3)^N: an array of shape (N, 3) is an element of so(3)^N, one row per factor, and an array of shape (N, 3, 3) is an element of SO(3)^N.
"""

import numpy as np

from doublebracket.array_types import Covec, CovecArray, GroupElement, GroupElementArray, Mat3x3, Vec


def hat(x: Vec) -> Mat3x3:
    """
    The hat map R^3 -> so(3),
                [  0   -x3   x2 ]
        x  ->   [  x3   0   -x1 ]
                [ -x2   x1   0  ]
    so that hat(x) y = x × y.
    """
    X = np.zeros((*x.shape, 3))
    X[..., 0, 1], X[..., 0, 2] = -x[..., 2], x[..., 1]
    X[..., 1, 0], X[..., 1, 2] = x[..., 2], -x[..., 0]
    X[..., 2, 0], X[..., 2, 1] = -x[..., 1], x[..., 0]
    return X


def vee(X: Mat3x3) -> Vec:
    """The inverse of the hat map, for a skew-symmetric matrix X."""
    return np.stack([X[..., 2, 1], X[..., 0, 2], X[..., 1, 0]], axis=-1)


def cross(x: Vec, y: Vec) -> Vec:
    """The cross product x × y = [x, y], the Lie bracket of so(3). (Faster than np.cross for small arrays.)"""
    z = np.empty(np.broadcast_shapes(x.shape, y.shape))
    z[..., 0] = x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1]
    z[..., 1] = x[..., 2] * y[..., 0] - x[..., 0] * y[..., 2]
    z[..., 2] = x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    return z


class SO3:
    """
    SO(3), or SO(3)^N when the arrays have a leading axis of length N.

    Group elements are rotation matrices R (one per factor), and the product is the matrix product.
    """

    def coadjoint_action(self, g: GroupElement, mu: Covec) -> Covec:
        """
        Ad*_R mu = R^T mu (for each factor).

        Indeed, Ad_R x = R x, since R hat(x) R^T = hat(R x), and the dual of R is R^T.
        """
        return np.einsum("...ji,...j->...i", g, mu)

    def ad_star(self, xi: Vec, mu: Covec) -> Covec:
        """ad*_xi mu = mu × xi (for each factor), since <mu, xi × eta> = <mu × xi, eta>."""
        return cross(mu, xi)


def spatial_momenta(R: GroupElementArray, M: CovecArray) -> CovecArray:
    """
    The spatial angular momenta pi = R M = Ad*_{R^{-1}} M (for each factor and time).

    For a left-invariant system without dissipation, pi is conserved (Noether's theorem).
    """
    return np.einsum("...ab,...b->...a", R, M)
