"""
Convergence in the step size h: run every method for a range of h values, and plot the errors at the final time against h.

The sweep is slow, so hsweep saves a summary of each run (final momentum and a few scalars, not the whole trajectory) to plots/<example name>/hsweep_summary.pkl, and plot_hsweep makes the figures from that file.
"""

import os
import pickle
from dataclasses import replace
from time import perf_counter
from typing import TypedDict

import numpy as np

from doublebracket.array_types import Covec
from doublebracket.ddb import StageSolverError
from doublebracket.diagnostics import casimir_drift, energies, energy_dissipation_residual, largest_inertia_axes
from doublebracket.examples import Example, RigidBodyExample
from doublebracket.methods import METHODS, REFERENCE
from doublebracket.plots import plot_error_vs_h


class RunSummary(TypedDict):
    h: float
    N: int  # number of steps
    M_final: Covec
    energy_final: float
    casimir_error_final: float  # max_i |C_i(M_N) - C_i(M_0)|
    casimir_drift_max: float  # the same, maximized over time
    energy_increase_max: float  # max_k (H(M_{k+1}) - H(M_k)); <= 0 means that the energy is nonincreasing
    energy_residual_max: float  # max_k |r_k|, see diagnostics.energy_dissipation_residual
    runtime: float  # seconds


class Sweep(TypedDict):
    M_ref_final: Covec  # the reference (DOP853) solution at the final time
    energy_ref_final: float
    runs: dict[tuple[str, int], RunSummary]  # keys: (method key, index into hrange)


def sweep_path(example: Example) -> str:
    return f"plots/{example.name}/hsweep_summary.pkl"


def hsweep(example: Example, hrange: np.ndarray, disabled_methods: tuple[str, ...] = ()) -> None:
    """
    Run every method except the reference and disabled_methods, for each h in hrange, and save the summaries. The step actually used is T/N, with N = round(T/h) steps. The reference solution is computed once. A DDB run whose stage equations cannot be solved is reported and left out.
    """
    model = example.model
    T = example.t_span[1] - example.t_span[0]
    M_ref = METHODS[REFERENCE].run(replace(example, h=T)).M  # output at t_0 and t_0 + T only
    sweep: Sweep = {"M_ref_final": M_ref[-1], "energy_ref_final": model.hamiltonian(M_ref[-1]), "runs": {}}

    for index, h in enumerate(hrange):
        N = round(T / h)
        run_example = replace(example, h=T / N)
        print(f"Running with N = {N}, h = {T / N:.6f} ({index + 1}/{len(hrange)})")
        for key in METHODS:
            if key == REFERENCE or key in disabled_methods:
                continue
            print(f"    Solving with {METHODS[key].full_name}...", end="", flush=True)
            start = perf_counter()
            try:
                M = METHODS[key].run(run_example).M
            except StageSolverError as error:
                print(f" FAILED, left out of the sweep: {error}")
                continue
            runtime = perf_counter() - start
            print(f" Time: {runtime:.2f}")
            energy = energies(model, M)
            drift = casimir_drift(model, M)
            sweep["runs"][(key, index)] = {
                "h": T / N,
                "N": N,
                "M_final": M[-1],
                "energy_final": energy[-1],
                "casimir_error_final": drift[-1],
                "casimir_drift_max": drift.max(),
                "energy_increase_max": np.diff(energy).max(),
                "energy_residual_max": np.abs(energy_dissipation_residual(model, M, T / N)).max(),
                "runtime": runtime,
            }

    os.makedirs(os.path.dirname(sweep_path(example)), exist_ok=True)
    with open(sweep_path(example), "wb") as file:
        pickle.dump(sweep, file)
    print(f"Sweep results saved as '{sweep_path(example)}'")


def plot_hsweep(example: Example, h_limits: tuple[float, float] = (0.0, np.inf)) -> None:
    """
    Figures and a table from the saved sweep, for the h values within h_limits:
    - per_quantity/: one quantity for all methods: the final momentum, energy and Casimir errors, and the maximum over time of the Casimir drift and of the energy-dissipation residual;
    - per_method/: the final errors of each method together;
    - hsweep_summary.txt: all the numbers.
    """
    with open(sweep_path(example), "rb") as file:
        sweep: Sweep = pickle.load(file)
    runs = {key: run for key, run in sweep["runs"].items() if h_limits[0] <= run["h"] <= h_limits[1]}
    method_keys = [key for key in METHODS if any(method_key == key for method_key, _ in runs)]
    outdir = f"plots/{example.name}"

    def by_method(quantity) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """For each method, the h values (increasing) and the values of quantity(run)."""
        series = {}
        for key in method_keys:
            rows = sorted((run["h"], quantity(run)) for (method_key, _), run in runs.items() if method_key == key)
            series[key] = (np.array([h for h, _ in rows]), np.array([value for _, value in rows]))
        return series

    quantities = {
        "final_M_error": (
            lambda run: np.linalg.norm(run["M_final"] - sweep["M_ref_final"]),
            r"$\|M_{N} - M_{N}^{\mathrm{ref}}\|$",
            "Final momentum error vs time step size",
        ),
        "final_energy_error": (
            lambda run: abs(run["energy_final"] - sweep["energy_ref_final"]),
            r"$|H(M_{N})-H(M_{N}^{\mathrm{ref}})|$",
            "Final energy error vs time step size",
        ),
        "final_casimir_error": (
            lambda run: run["casimir_error_final"],
            r"$\max_i |C_i(M_{N})-C_i(M_0)|$",
            "Final Casimir error vs time step size",
        ),
        "casimir_drift": (
            lambda run: run["casimir_drift_max"],
            r"$\max_k \max_i |C_i(M_k)-C_i(M_0)|$",
            "Casimir drift vs time step size",
        ),
        "energy_residual": (
            lambda run: run["energy_residual_max"],
            r"$\max_k |r_k|$",
            "Energy-dissipation residual vs time step size",
        ),
    }
    series = {name: by_method(quantity) for name, (quantity, _, _) in quantities.items()}
    for name, (_, ylabel, title) in quantities.items():
        order_lines = example.hsweep_order_lines if name == "final_M_error" else ()
        plot_error_vs_h(series[name], ylabel, title, f"{outdir}/per_quantity/hsweep_{name}_vs_h.svg", order_lines)

    if isinstance(example, RigidBodyExample):
        # The minimum of the energy on the Casimir sphere |M| = |M_0|, attained on the axes of largest moment of inertia
        axis = largest_inertia_axes(example.Itensor)[0]
        M_min = np.zeros(3)
        M_min[axis] = np.linalg.norm(example.M0)
        min_energy = example.model.hamiltonian(M_min)
        plot_error_vs_h(
            by_method(lambda run: abs(run["energy_final"] - min_energy)),
            r"$|H(M_{N}) - \mathrm{min}_{\mathcal{S}}H|$",
            "Final energy minus terminal energy vs time step size",
            f"{outdir}/per_quantity/hsweep_final_energy_minus_min_vs_h.svg",
        )

    styles = {  # name: (linespec, label)
        "final_M_error": ("k-", quantities["final_M_error"][1]),
        "final_casimir_error": ("b--", quantities["final_casimir_error"][1]),
        "final_energy_error": ("r:", quantities["final_energy_error"][1]),
    }
    for key in method_keys:
        plot_error_vs_h(
            {name: series[name][key] for name in styles},
            "Error",
            f"Final errors vs time step size ({METHODS[key].label})",
            f"{outdir}/per_method/hsweep_final_errors_vs_h_{METHODS[key].short_name}.svg",
            order_lines=(example.hsweep_method_order_lines[key],) if key in example.hsweep_method_order_lines else (),
            styles=styles,
        )

    table = [
        f"h-sweep for {example.name} (T = {example.t_span[1] - example.t_span[0]}); reference H(M_ref(T)) = {sweep['energy_ref_final']:.10f}",
        f"{'method':14s} {'N':>6s} {'h':>10s} {'error(T)':>10s} {'E err(T)':>10s} {'Cas drift':>10s} {'max dE':>10s} {'max|r_k|':>10s} {'time (s)':>9s}",
    ]
    for (key, _), run in sorted(runs.items(), key=lambda item: (item[0][0], item[1]["h"])):
        table.append(
            f"{key:14s} {run['N']:6d} {run['h']:10.6f} {np.linalg.norm(run['M_final'] - sweep['M_ref_final']):10.3e} "
            f"{abs(run['energy_final'] - sweep['energy_ref_final']):10.3e} {run['casimir_drift_max']:10.1e} "
            f"{run['energy_increase_max']:10.2e} {run['energy_residual_max']:10.3e} {run['runtime']:9.2f}"
        )
    print("\n".join(table))
    with open(f"{outdir}/hsweep_summary.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(table) + "\n")
