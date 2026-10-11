"""
Type aliases for the arrays used throughout the code.

All examples live on G = SO(3)^N, with N = 1 for the single rigid body. We identify
    g = so(3)^N  with (R^3)^N  through the hat map (componentwise),
    g* = (so(3)^N)*  with (R^3)^N  through the dot product,
so an element of g or g* is an array of shape (N, 3), with row i for body i, or of shape (3,) for a single rigid body. The aliases below use jaxtyping's "*sites" to cover both cases. Vectors and covectors are the same type in practice; the distinction is only for readability.
"""

from jaxtyping import Float
from numpy import ndarray

# One state
type Vec = Float[ndarray, "*sites 3"]  # element of g, e.g. the velocity xi (Omega for the rigid body)
type Covec = Float[ndarray, "*sites 3"]  # element of g*, e.g. the momentum M
type GroupElement = Float[ndarray, "*sites 3 3"]  # element of SO(3)^N: one rotation matrix per body
type Mat3x3 = Float[ndarray, "*sites 3 3"]  # any 3x3 matrix per body, e.g. hat(x) or an inertia tensor
type Matrix = Float[ndarray, "d d"]  # a linear map on (R^3)^N, flattened, d = 3N
type SiteScalars = Float[ndarray, "N"]  # one number per body, e.g. the Casimirs

# Time series: the first axis is time, t_k = t_0 + k h
type TimeArray = Float[ndarray, "n"]
type VecArray = Float[ndarray, "n *sites 3"]
type CovecArray = Float[ndarray, "n *sites 3"]
type GroupElementArray = Float[ndarray, "n *sites 3 3"]
type ScalarArray = Float[ndarray, "n"]  # one number per time step, e.g. the energy

# Flattened states, as seen by the general-purpose ODE integrators
type FlatState = Float[ndarray, "d"]
type FlatStateArray = Float[ndarray, "n d"]
