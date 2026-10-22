"""
Run all the methods on an example and compare them: timings, a text summary, and figures in plots/<example name>/.

For the rigid body, also the figures of Section 6 of the paper: Casimir, distance to the limit set, and the trajectory on the Casimir sphere.
"""

import os
from time import perf_counter

import numpy as np

from doublebracket.diagnostics import (
    casimir_drift,
    distance_to_great_circle,
    energies,
    energy_dissipation_residual,
    equilibrium_residual,
    largest_inertia_axes,
    momentum_error,
)
from doublebracket.examples import Example, LatticeExample, RigidBodyExample
from doublebracket.methods import METHODS, REFERENCE, Trajectory
from doublebracket.plots import (
    plot_energy_levels_on_sphere,
    plot_rigid_body_run,
    plot_time_series,
    plot_trajectory_on_sphere,
)


def run_methods(example: Example, method_keys: list[str]) -> tuple[dict[str, Trajectory], dict[str, float]]:
    """Run the given methods (the reference must be among them), and return their trajectories and run times."""
    trajectories, timings = {}, {}
    for key in method_keys:
        print(f"Solving with {METHODS[key].full_name}...")
        start = perf_counter()
        trajectories[key] = METHODS[key].run(example)
        timings[key] = perf_counter() - start
    return trajectories, timings


def write_lines(lines: list[str], path: str) -> None:
    print("\n" + "\n".join(lines))
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print(f"Saved as '{path}'")


def compare_methods(example: Example, disabled_methods: tuple[str, ...] = ()) -> dict[str, Trajectory]:
    """
    Run all methods except disabled_methods, write timings.txt and summary.txt, and plot (against time) the momentum error and the energy error with respect to the reference, the Casimir drift, the energy, and the energy-dissipation residual. Returns the trajectories.
    """
    model = example.model
    trajectories, timings = run_methods(example, [key for key in METHODS if key not in disabled_methods])
    outdir = f"plots/{example.name}"
    os.makedirs(outdir, exist_ok=True)

    write_lines(
        ["Execution times (seconds):"] + [f"  {METHODS[key].label:20s}: {timings[key]:.6f}" for key in trajectories],
        f"{outdir}/timings.txt",
    )

    t = example.times
    M_ref = trajectories[REFERENCE].M
    energy_ref = energies(model, M_ref)
    energy = {key: energies(model, tr.M) for key, tr in trajectories.items()}
    error = {key: momentum_error(tr.M, M_ref) for key, tr in trajectories.items()}
    drift = {key: casimir_drift(model, tr.M) for key, tr in trajectories.items()}
    residual = {key: energy_dissipation_residual(model, tr.M, example.h) for key, tr in trajectories.items()}

    summary = [
        f"Example {example.name}: alpha = {example.alpha}, t_span = {example.t_span}, h = {example.h}",
        f"H(M_0) = {energy_ref[0]:.10f}, H(M_ref(T)) = {energy_ref[-1]:.10f} (DOP853, rtol = {example.ref_rtol:g}, atol = {example.ref_atol:g})",
        "",
        (
            "Per method: ||M - M_ref|| and |H(M) - H(M_ref)| at T; max over time of the Casimir drift max_i |C_i(M_k) - C_i(M_0)|; "
            "largest energy increment max_k (H(M_{k+1}) - H(M_k)) (<= 0 means that the energy is nonincreasing); "
            "max_k |r_k| of the energy-dissipation residual, and the same divided by h^2."
        ),
    ]
    for key in trajectories:
        max_residual = np.abs(residual[key]).max()
        summary.append(
            f"  {METHODS[key].label:20s}: error(T) = {error[key][-1]:.3e}, "
            f"energy error(T) = {abs(energy[key][-1] - energy_ref[-1]):.3e}, "
            f"Casimir drift = {drift[key].max():.1e}, "
            f"max energy increment = {np.diff(energy[key]).max():.2e}, "
            f"max |r_k| = {max_residual:.3e} ({max_residual / example.h**2:.3f} h^2)"
        )
    write_lines(summary, f"{outdir}/summary.txt")

    others = [key for key in trajectories if key != REFERENCE]
    plot_time_series(
        {key: (t, error[key]) for key in others},
        r"$\|M - M_{\mathrm{DOP853}}\|$",
        f"Momentum difference vs built-in ODE solver ({METHODS[REFERENCE].full_name})",
        f"{outdir}/momentum_difference_norms.svg",
    )
    plot_time_series(
        {key: (t, drift[key]) for key in trajectories},
        r"$\max_i |C_i(M_k) - C_i(M_0)|$",
        "Casimir drift (distance to the initial coadjoint orbit)",
        f"{outdir}/casimir_drift_all_methods.svg",
    )
    if example.alpha > 0:
        plot_time_series(
            {key: (t, energy[key]) for key in trajectories},
            r"$H(M)$",
            "Energy (all methods)",
            f"{outdir}/energy_all_methods.svg",
            logscale=False,
        )
        plot_time_series(
            {key: (t, np.abs(energy[key] - energy_ref)) for key in others},
            r"$|H(M) - H(M^{\mathrm{DOP853}})|$",
            "Energy difference vs built-in ODE solver",
            f"{outdir}/energy_difference_all_methods.svg",
        )
    else:
        plot_time_series(
            {key: (t, np.abs(energy[key] - energy[key][0])) for key in trajectories},
            r"$|H(M) - H(M_0)|$",
            "Energy drift (no dissipation)",
            f"{outdir}/energy_drift_all_methods.svg",
        )
    plot_time_series(
        {key: (t[:-1], np.abs(residual[key])) for key in trajectories},
        r"$|r_k|$",
        r"Energy-dissipation residual $r_k = H(M_{k+1}) - H(M_k) - h\,\dot H(M_k)$",
        f"{outdir}/energy_dissipation_residual_all_methods.svg",
    )

    if isinstance(example, RigidBodyExample):
        compare_rigid_body(example, trajectories)
    if isinstance(example, LatticeExample) and example.alpha > 0:
        compare_lattice(example, trajectories)
    return trajectories


def compare_lattice(example: LatticeExample, trajectories: dict[str, Trajectory]) -> None:
    """
    The residual max_i |mu_i × xi_i| of the set E of equilibria, over the second half of the run (for t in [200, 400] if t_span = (0, 400)).
    """
    model = example.model
    t = example.times
    late = t >= (example.t_span[0] + example.t_span[1]) / 2
    plot_time_series(
        {key: (t[late], equilibrium_residual(model, tr.M)[late]) for key, tr in trajectories.items()},
        r"$\max_i \|\mu_i \times \xi_i\|$",
        r"Residual of the equilibrium set $\mathcal{E}$ (second half of the simulation)",
        f"plots/{example.name}/equilibrium_residual_late_all_methods.svg",
    )


def compare_rigid_body(example: RigidBodyExample, trajectories: dict[str, Trajectory]) -> None:
    """
    The rigid-body figures of Section 6: the Casimir C(M) = |M|^2 / 2, the distance to the limit set at late times, and the figures of the DDB (cay) run, including its trajectory on the Casimir sphere.

    The trajectories tend to the minimum-energy set on the Casimir sphere |M| = |M_0|: the points ±|M_0| e_j if j is the only axis of largest moment of inertia, or a great circle if there are two such axes.
    """
    model = example.model
    outdir = f"plots/{example.name}"
    t = example.times
    best = [key for key in ("DOP853", "DDB exp", "DDB cay") if key in trajectories]
    for keys, suffix, title in ((list(trajectories), "all", "all methods"), (best, "best", "selected methods")):
        plot_time_series(
            {key: (t, model.casimirs(trajectories[key].M)) for key in keys},
            r"$C(M) = \frac{1}{2}\|M\|^2$",
            f"Casimir evolution ({title})",
            f"{outdir}/casimir_{suffix}_methods.svg",
            logscale=False,
        )

    radius = float(np.linalg.norm(example.M0))
    axes = largest_inertia_axes(example.Itensor)
    late = t >= example.t_span[0] + 0.9 * (example.t_span[1] - example.t_span[0])
    final = {key: tr.M[-1] for key, tr in trajectories.items()}
    if len(axes) == 1:
        # The limit point: ±|M_0| e_j, with the sign of the reference solution
        limit = np.zeros(3)
        limit[axes[0]] = radius * np.sign(final[REFERENCE][axes[0]])
        lines = ["Final momentum differences against exact limit (isolated point):"] + [
            f"  {METHODS[key].label:20s}: ||M_last-limit|| = {np.linalg.norm(M_N - limit):.6e}, "
            f"collinearity = {np.linalg.norm(np.cross(M_N, limit)):.6e}, "
            f"| ||M_last||-||limit|| | = {abs(np.linalg.norm(M_N) - radius):.6e}"
            for key, M_N in final.items()
        ]
        plot_time_series(
            {key: (t[late], np.linalg.norm(tr.M - limit, axis=1)[late]) for key, tr in trajectories.items()},
            r"dist$(M_k,\text{limit point})$",
            "Distance to exact limit point",
            f"{outdir}/distance_to_limit_point_all_methods.svg",
        )
    elif len(axes) == 2:
        # The limit set: the great circle of radius |M_0| orthogonal to the axis of smallest moment of inertia
        normal_axis = int(np.argmin(np.diag(example.Itensor)))
        lines = ["Final momentum differences against exact limit (great circle):"] + [
            f"  {METHODS[key].label:20s}: distance from M_last to plane of great circle = {abs(M_N[normal_axis]):.6e}"
            for key, M_N in final.items()
        ]
        plot_time_series(
            {
                key: (t[late], distance_to_great_circle(tr.M, radius, normal_axis)[late])
                for key, tr in trajectories.items()
            },
            r"dist$(M_k,\text{great circle})$",
            "Distance to great circle",
            f"{outdir}/distance_to_great_circle_all_methods.svg",
        )
        plot_time_series(
            {key: (t[late], np.abs(tr.M[late, normal_axis])) for key, tr in trajectories.items()},
            rf"$|M_{{{normal_axis + 1}}}|$",
            "Distance to plane of great circle",
            f"{outdir}/distance_to_plane_of_great_circle_all_methods.svg",
        )
    else:
        lines = ["Homogeneous body: every point of the Casimir sphere is an equilibrium."]
    write_lines(lines, f"{outdir}/limit_M.txt")

    method_key = "DDB cay"
    if method_key in trajectories:
        method_dir = f"{outdir}/{METHODS[method_key].short_name}"
        plot_rigid_body_run(example, method_key, trajectories[method_key], method_dir)
        plot_trajectory_on_sphere(
            example, trajectories[method_key].M, f"{method_dir}/mk_evolution_3d_{METHODS[method_key].short_name}.png"
        )
        plot_energy_levels_on_sphere(example, trajectories[method_key].M, f"{outdir}/energy_levels_on_sphere.png")
