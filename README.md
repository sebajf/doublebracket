# Discrete double bracket dissipation

Discrete double-bracket (DDB) integrators: code for the paper

> A. Bloch, S. J. Ferraro, D. Martín de Diego, S. Bharadwaj, *Discrete variational calculus for double-bracket dissipation*, [arXiv:2604.26049](https://arxiv.org/abs/2604.26049).

The DDB methods integrate Euler–Poincaré (equivalently, Lie–Poisson) systems on a Lie algebra $\mathfrak g$ with double-bracket dissipation,

$$\frac{d\mu}{dt} = \mathrm{ad}^*_{\xi + \varphi(\xi)}\mu, \qquad \mu = \frac{\partial l}{\partial \xi},$$

so that the momentum stays on its coadjoint orbit while the energy is not preserved. The discrete methods preserve the coadjoint orbits exactly (up to round-off). The code compares them with RK4, 3-stage Lobatto IIIC and a DOP853 reference solution, on two examples:

- the free rigid body on $SO(3)$ (Section 6 of the paper);
- a $4\times4$ periodic lattice of rigid bodies on $SO(3)^{16}$, coupled through their momenta as in Arnaudon and Takao, *Networks of coadjoint orbits* (J. Geom. Mech., 2019).

## Installation

The project uses [uv](https://docs.astral.sh/uv/), which installs the right Python version and all the dependencies. If you do not have uv yet, install it with

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(see the uv documentation for Windows and other options). Then

```bash
git clone https://github.com/sebajf/doublebracket.git
cd doublebracket
uv sync
```

`uv sync` creates a virtual environment in `.venv/` with Python 3.13, the dependencies, the development tools (pytest, ruff) and the `doublebracket` package itself. Prefix commands with `uv run` to use this environment.

Two optional external programs:
- **ffmpeg**, for the videos (for instance `sudo apt install ffmpeg` or `brew install ffmpeg`);
- **Chrome or Chromium**, which plotly uses to save the 3D sphere figures as PNG. If it is not installed, run `uv run plotly_get_chrome`.

## Running the simulations

```bash
uv run run_rigid_body.py   # rigid body: Scenarios A, B and C of the paper (about 2 minutes)
uv run run_lattice.py      # 4x4 lattice: examples Lattice_A, Lattice_B, Lattice_cons (about 5 minutes; Lobatto IIIC takes most of it)
```

The settings are at the top of each script: which examples to run, which methods to skip (Lobatto IIIC is the slowest), whether to run the h-sweep, and which videos to make. The examples themselves (inertia tensors, dissipation coefficient, initial conditions, time span, step size) are defined in [`doublebracket/examples.py`](doublebracket/examples.py).

All output goes to `plots/<example name>/`:

| Output | Content                                                                                                                         |
| --- | --- |
| `timings.txt`, `summary.txt` | run time of each method; errors against the reference, Casimir drift, energy monotonicity                                       |
| `momentum_difference_norms.svg` | $\lVert M_k - M_k^{\mathrm{ref}}\rVert$ against time                                                                            |
| `casimir_drift_all_methods.svg` | $\max_i \lvert C_i(M_k) - C_i(M_0)\rvert$: drift from initial Casimir value                                                     |
| `energy_*.svg` | the Hamiltonian $H(M_k)$, its error, and the energy-dissipation residual $r_k = H(M_{k+1}) - H(M_k) - h\dot H(M_k)$             |
| `casimir_*_methods.svg`, `distance_to_*.svg`, `limit_M.txt` | rigid body only: the Casimir, and the approach to the limit set (a pair of points or a great circle)                            |
| `ddb_cay/` | rigid body only: figures of the DDB (cay) run, including the trajectory on the Casimir sphere (`.png`, and interactive `.html`) |
| `per_quantity/`, `per_method/`, `hsweep_summary.txt` | errors at the final time against the step size $h$ (from the h-sweep)                                                           |
| `<method>/*.mp4` | videos: the rotating bodies and their angular momenta                                                                           |

The h-sweep runs every method for a range of step sizes. For the rigid body it takes a long time, so its results are saved in `plots/<example name>/hsweep_summary.pkl` and the figures are drawn from that file. The repository includes these files for Scenarios A, B, C and Lattice_A, so the scripts draw the h-sweep figures without running the sweep. To run the sweep again, set `RUN_HSWEEP = True`, and then set it back to `False`.

The folder [`paper_figures/`](paper_figures/) contains the figures of Section 6 of the paper, as produced by `run_rigid_body.py` and renamed with the scenario as prefix (for instance `plots/Scenario_A/casimir_all_methods.svg` is `A-casimir_all_methods.svg`).

### Using the integrators in your own code

```python
import numpy as np

from doublebracket.ddb import ddb_solve, reconstruct
from doublebracket.models import rigid_body
from doublebracket.retractions import Cayley

model = rigid_body(Itensor=np.diag([1.0, 1.3, 0.5]), alpha=0.5)
M0 = model.legendre(np.array([0.0, 0.05, 1.0]))  # M = I Omega
M, xi = ddb_solve(model, Cayley(), M0, h=0.1, n_steps=300)  # momenta M_0..M_300 and stage velocities xi_0..xi_299
R = reconstruct(Cayley(), np.eye(3), xi, h=0.1)  # attitudes R_{k+1} = R_k cay(h xi_k)
```

## How the code is organized

The DDB method needs three ingredients: a Lie group, a retraction map on it, and a model (Lagrangian and dissipation). Section 5 of the paper describes them, and the code has one layer for each of them, plus a solver layer that uses only those ingredients. The protocols in [`interfaces.py`](doublebracket/interfaces.py) list what each layer provides.

| Layer      | File                                             | Contents                                                                                                                                                                                       |
| ---------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| group      | [`so3.py`](doublebracket/so3.py)                 | the hat map, the cross product (the bracket of $\mathfrak{so}(3)$), $\mathrm{Ad}^*_R\mu = R^T\mu$, $\mathrm{ad}^*_\xi\mu = \mu\times\xi$                                                       |
| retraction | [`retractions.py`](doublebracket/retractions.py) | `Cayley` and `Exp` ($\tau$ by the Cayley and Rodrigues formulas, and $(\mathrm d\tau_x^{-1})^*$; for `Exp`, exact or truncated at a given order)                                               |
| model      | [`models.py`](doublebracket/models.py)           | `RigidBodySystem`: the Lagrangian, the Legendre transform and its inverse, the Hamiltonian and energy, the dissipation map $\varphi$, the Casimirs; the continuous equations `lie_poisson_rhs` |
| solver     | [`ddb.py`](doublebracket/ddb.py)                 | Algorithm 1 of the paper: `solve_stage_equations`, `ddb_step`, `ddb_solve`, and `reconstruct`                                                                                                  |

All the examples live on $SO(3)^N$, with $N = 1$ for the single rigid body. An element of $\mathfrak g$ or $\mathfrak g^*$ is an array of shape `(N, 3)`, or `(3,)` for one body, and a group element is an array of shape `(N, 3, 3)`, or `(3, 3)`. The functions of the group and retraction layers act on the last axis, so the same code handles $SO(3)$ and $SO(3)^N$.

The other files:

| File | Contents |
| --- | --- |
| [`array_types.py`](doublebracket/array_types.py) | type aliases for the arrays |
| [`integrators.py`](doublebracket/integrators.py) | RK4, Lobatto IIIC and DOP853, applied to the continuous equations for $\mu$ |
| [`examples.py`](doublebracket/examples.py) | the examples (`Scenario_A`, ..., `Lattice_A`, ...); the input is checked here, once |
| [`methods.py`](doublebracket/methods.py) | the methods compared, with their names and plot styles |
| [`diagnostics.py`](doublebracket/diagnostics.py) | energy, Casimir drift, energy-dissipation residual, distance to the limit set |
| [`compare.py`](doublebracket/compare.py), [`hsweep.py`](doublebracket/hsweep.py) | comparison runs and h-sweeps, with their text output |
| [`plots.py`](doublebracket/plots.py), [`animation.py`](doublebracket/animation.py) | figures and videos |

### The examples

A rigid body and a lattice of $16$ rigid bodies at the sites of a $4\times4$ discrete torus (see the paper for the mathematical formulation).  Both examples are instances of `RigidBodySystem`.
### Adding an example

- **Same group, new parameters or initial conditions:** add an instance of `RigidBodyExample` or `LatticeExample` in `examples.py`. Its variable name becomes the name of the output folder.
- **Another system on $SO(3)^N$ with a quadratic Lagrangian:** construct a `RigidBodySystem` with its matrix $\mathbb I$ (see `torus_lattice` in `models.py`).
- **Another group** (for instance, the rigid body on $SO(4)$): write a group class and retraction classes for it, following `so3.py` and `retractions.py`, and a model following the `Model` protocol of `interfaces.py`. `ddb.py` and `integrators.py` work unchanged.

## Tests

This is for development and for adding new examples.

```bash
uv run pytest
```

The tests check the identities of the paper (for instance, $(\mathrm d\tau_x^{-1})^*$ against a finite-difference computation of $\mathrm d\tau_x$, exact Casimir preservation, monotone energy decrease, exact conservation of the spatial momenta without dissipation), and they include regression tests: 5-step runs of every method on both examples, whose output is stored in `tests/test_methods/`. If a change modifies these results, the regression tests fail; regenerate the stored data (`uv run pytest --force-regen`) only after checking that the change is intended.

## Use of generative AI

A generative AI tool (Claude Opus 5.5, Anthropic) was used to write the lattice simulations, based on the existing code for the single rigid body on SO(3), and to polish and refactor the code: the layered organization of the package, the documentation and the tests. The authors reviewed the changes and checked that the refactored code reproduces the results of the original code, and they take full responsibility for its content.

## License

The code is distributed under the MIT license; see [`LICENSE`](LICENSE).
