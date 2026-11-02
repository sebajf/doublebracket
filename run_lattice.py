"""
The 4x4 periodic lattice of rigid bodies with momentum coupling and double-bracket dissipation: comparison of the methods, convergence in h, and videos.

Run with
    uv run run_lattice.py
The examples are defined in doublebracket/examples.py (Lattice_A, Lattice_B), and the output goes to plots/<example name>/. The settings are below.
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
EXAMPLES = [
    examples.Lattice_A,
    examples.Lattice_B,
    examples.Lattice_B_T400,
    examples.Lattice_low_dissipation,
    examples.Lattice_high_dissipation,
]
# Lobatto IIIC is the slowest method (about 50 ms per step for the 4x4 lattice); add it here to skip it.
DISABLED_METHODS: tuple[str, ...] = ()
# The h-sweep takes a few minutes. Its results are saved, and the figures are made from the saved file.
HSWEEP_EXAMPLES = [examples.Lattice_A, examples.Lattice_high_dissipation]
RUN_HSWEEP = False
HRANGE = np.logspace(np.log10(0.2), np.log10(0.002), 7)
# Videos of the DDB (cay) run: rotating boxes and body angular momenta. They need ffmpeg and take a few minutes.
VIDEO_EXAMPLES: list[examples.LatticeExample] = []  # e.g. [examples.Lattice_B]
# Stills for the paper: the first and last frames of the body momenta video, without labels (PDF).
STILL_EXAMPLES: list[examples.LatticeExample] = []  # e.g. [examples.Lattice_B]


if __name__ == "__main__":
    for example in EXAMPLES:
        compare_methods(example, DISABLED_METHODS)
    for example in HSWEEP_EXAMPLES:
        if RUN_HSWEEP:
            hsweep(example, HRANGE, DISABLED_METHODS)
        if os.path.exists(sweep_path(example)):
            plot_hsweep(example)
    for example in VIDEO_EXAMPLES:
        create_animations(example, "DDB cay", momentum_frame="body", fps=30, speedup=10.0)
    for example in STILL_EXAMPLES:
        create_animations(example, "DDB cay", boxes=False, momentum_frame="body", video=False, stills=True)
