"""
The free rigid body with double-bracket dissipation (Section 6 of the paper): comparison of the methods, convergence in h, and videos.

Run with
    uv run run_rigid_body.py
The examples are defined in doublebracket/examples.py, and the output goes to plots/<example name>/. The settings are below.
"""

import os

import matplotlib
import numpy as np

from doublebracket import examples
from doublebracket.animation import create_animations
from doublebracket.compare import compare_methods
from doublebracket.hsweep import hsweep, plot_hsweep, sweep_path

matplotlib.use("Agg")  # save figures to files, without opening windows
matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "figure.figsize": (6.4, 4.8),
        "legend.loc": "best",
    }
)

# Settings
EXAMPLES = [examples.Scenario_A, examples.Scenario_B, examples.Scenario_C]  # Scenarios A, B, C of the paper
DISABLED_METHODS: tuple[str, ...] = ()  # e.g. ("Lobatto IIIC",), the slowest method
# The h-sweep is slow (hours for Scenarios B and C). Its results are saved, and the figures are made from the saved file,
# so run it once and then set RUN_HSWEEP = False.
RUN_HSWEEP = False
HRANGE = np.logspace(0, -3, 10)  # from h = 1 to h = 0.001, three values per decade
# Videos of the DDB (cay) run: rotating box and angular momentum. They need ffmpeg and take a few minutes.
VIDEO_EXAMPLES: list[examples.RigidBodyExample] = []  # e.g. [examples.For_Video_A, examples.For_Video_C]


if __name__ == "__main__":
    for example in EXAMPLES:
        compare_methods(example, DISABLED_METHODS)
        if RUN_HSWEEP:
            hsweep(example, HRANGE, DISABLED_METHODS)
        if os.path.exists(sweep_path(example)):
            plot_hsweep(example)
    for example in VIDEO_EXAMPLES:
        create_animations(example, "DDB cay", momentum_frame="body", fps=60, speedup=10.0)
