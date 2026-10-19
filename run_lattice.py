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

# Settings
EXAMPLES = [examples.Lattice_A, examples.Lattice_B]
# Lobatto IIIC is the slowest method (about 50 ms per step for the 4x4 lattice); add it here to skip it.
DISABLED_METHODS: tuple[str, ...] = ()
# The h-sweep takes a few minutes. Its results are saved, and the figures are made from the saved file.
HSWEEP_EXAMPLE = examples.Lattice_A
RUN_HSWEEP = False
HRANGE = np.logspace(np.log10(0.2), np.log10(0.002), 7)
# Videos of the DDB (cay) run: rotating boxes and spatial angular momenta. They need ffmpeg and take a few minutes.
VIDEO_EXAMPLES: list[examples.LatticeExample] = []  # e.g. [examples.Lattice_B]


if __name__ == "__main__":
    for example in EXAMPLES:
        compare_methods(example, DISABLED_METHODS)
    if RUN_HSWEEP:
        hsweep(HSWEEP_EXAMPLE, HRANGE, DISABLED_METHODS)
    if os.path.exists(sweep_path(HSWEEP_EXAMPLE)):
        plot_hsweep(HSWEEP_EXAMPLE)
    for example in VIDEO_EXAMPLES:
        create_animations(example, "DDB cay", momentum_frame="spatial", fps=30, speedup=10.0)
