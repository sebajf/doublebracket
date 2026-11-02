"""
Static figures: comparisons of the methods over time and over the step size h, and figures of a single rigid-body run.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import colors as mcolors
from matplotlib import transforms

from doublebracket.array_types import CovecArray, Mat3x3, TimeArray
from doublebracket.diagnostics import largest_inertia_axes
from doublebracket.examples import RigidBodyExample
from doublebracket.methods import METHODS, Trajectory


def save_figure(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"Plot saved as '{path}'")


def plot_time_series(
    series: dict[str, tuple[TimeArray, np.ndarray]],
    ylabel: str,
    title: str,
    path: str,
    logscale: bool = True,
    labels: dict[str, str] | None = None,
    legend_bottom: float | None = None,
) -> None:
    """
    One curve per method, from a dict mapping method keys to (times, values). labels replaces the legend labels of some methods. If legend_bottom is given, the legend is placed at the right, with its bottom edge at y = legend_bottom (in data coordinates).
    """
    plt.figure(figsize=(8, 3.2))
    for method_key, (t, values) in series.items():
        method = METHODS[method_key]
        label = labels.get(method_key, method.label) if labels else method.label
        plt.plot(t, values, method.linespec, linewidth=1.5, label=label)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if logscale:
        plt.yscale("log")
    if legend_bottom is None:
        plt.legend()
    else:
        ax = plt.gca()
        to_axes_x_data_y = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        plt.legend(loc="lower right", bbox_to_anchor=(1, legend_bottom), bbox_transform=to_axes_x_data_y)
    plt.tight_layout()
    save_figure(path)


def plot_error_vs_h(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    ylabel: str,
    title: str,
    path: str,
    order_lines: tuple[tuple[int, float], ...] = (),
    styles: dict[str, tuple[str, str]] | None = None,
) -> None:
    """
    Log-log plot of errors against h, from a dict mapping keys to (h values, errors). The keys are method keys, unless styles maps them to (linespec, label). Each (order, factor) in order_lines adds the dotted reference line factor * h^order.
    """
    plt.figure(figsize=(8, 4))
    for key, (h_values, errors) in series.items():
        linespec, label = styles[key] if styles else (METHODS[key].linespec, METHODS[key].label)
        plt.loglog(h_values, errors, linespec, marker="D", linewidth=1.5, label=label)
    if order_lines and series:
        h_values = next(iter(series.values()))[0]
        for order, factor in order_lines:
            line = factor * h_values**order
            plt.loglog(h_values, line, ":k", linewidth=1)
            middle = len(h_values) // 2
            plt.text(h_values[middle], line[middle] * 1.2, rf"$\propto h^{{{order}}}$", ha="center", va="bottom")
    plt.xlabel(r"$h$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_figure(path)


# Figures of a single rigid-body run


def plot_rigid_body_run(example: RigidBodyExample, method_key: str, trajectory: Trajectory, outdir: str) -> None:
    """The Casimir, the components of Omega_k = I^{-1} M_k and of M_k, and the Lagrangian l(Omega_k), for one method."""
    method = METHODS[method_key]
    model = example.model
    t, M = trajectory.t, trajectory.M
    Omega = np.array([model.inverse_legendre(Mk) for Mk in M])

    def new_figure(ylabel: str, title: str) -> None:
        plt.figure(figsize=(8, 4.5))
        plt.xlabel("t")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)

    new_figure(r"$C(M_k)=\frac{1}{2}\|M_k\|^2$", f"Casimir for {method.full_name}")
    plt.plot(t, model.casimirs(M), lw=2.0, color="tab:blue")
    plt.tight_layout()
    save_figure(f"{outdir}/casimir_{method.short_name}.svg")

    new_figure(r"$\Omega_k$", r"Components of $\Omega_k = I^{-1} M_k$")
    for i in range(3):
        plt.plot(t, Omega[:, i], lw=2.0, label=rf"$\Omega_{{k,{i + 1}}}$")
    plt.legend()
    plt.tight_layout()
    save_figure(f"{outdir}/Omega_{method.short_name}.svg")

    new_figure(r"$M_k$", r"Components of $M_k$")
    for i in range(3):
        plt.plot(t, M[:, i], lw=2.0, label=rf"$M_{{k,{i + 1}}}$")
    radius = float(np.linalg.norm(M[0]))
    plt.axhline(y=radius, color="k", linestyle=":", linewidth=1.5, label=r"$\pm\|M_0\|$")
    plt.axhline(y=-radius, color="k", linestyle=":", linewidth=1.5)
    plt.legend()
    plt.tight_layout()
    save_figure(f"{outdir}/M_{method.short_name}.svg")

    new_figure(r"$l(\Omega_k)$", r"Evolution of the Lagrangian")
    plt.plot(t, [model.lagrangian(Omega_k) for Omega_k in Omega], lw=2.0, color="tab:green")
    plt.tight_layout()
    save_figure(f"{outdir}/lagrangian_{method.short_name}.svg")


def energy_levels_on_sphere(Itensor: Mat3x3, radius: float, great_circle: bool) -> go.Figure:
    """
    A plotly figure of the Casimir sphere |M| = radius with level curves of the energy H(M) = 1/2 <M, I^{-1} M>, for a diagonal inertia tensor. If great_circle is set (two equal largest moments of inertia), the minimum-energy great circle is drawn explicitly (being a minimum, it often does not get automatically drawn as a level curve).
    """
    inverse_moments = 1 / np.diag(Itensor)

    # Sphere mesh
    u = np.linspace(0.0, 2.0 * np.pi, 70)
    v = np.linspace(0.0, np.pi, 40)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = radius * np.outer(np.ones_like(u), np.cos(v))

    # Energy on the sphere, in spherical coordinates (phi, theta)
    theta = np.linspace(1e-3, np.pi - 1e-3, 500)
    phi = np.linspace(0.0, 2.0 * np.pi, 1000)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    unit_vectors = np.stack(
        [np.sin(theta_grid) * np.cos(phi_grid), np.sin(theta_grid) * np.sin(phi_grid), np.cos(theta_grid)], axis=-1
    )
    energy = 0.5 * radius**2 * unit_vectors**2 @ inverse_moments

    e_min, e_max = float(energy.min()), float(energy.max())
    levels = np.linspace(e_min, e_max, 17)[1:-1]
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=e_min, vmax=e_max)

    # Use matplotlib's contouring only to extract the level curves in (phi, theta)
    figure_for_contours, axes_for_contours = plt.subplots()
    contour_set = axes_for_contours.contour(phi_grid, theta_grid, energy, levels=levels)
    level_curves = [(float(level), list(segments)) for level, segments in zip(levels, contour_set.allsegs)]
    plt.close(figure_for_contours)
    if great_circle:
        equator = np.column_stack((phi, np.full_like(phi, np.pi / 2)))
        level_curves.append((e_min, [equator]))

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=x_sphere,
            y=y_sphere,
            z=z_sphere,
            opacity=0.9,
            colorscale=[[0.0, "lightskyblue"], [1.0, "lightskyblue"]],
            contours=dict(
                x=dict(show=False, highlight=False),
                y=dict(show=False, highlight=False),
                z=dict(show=False, highlight=False),
            ),
            showscale=False,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    for level, segments in level_curves:
        color = mcolors.to_hex(cmap(norm(level)), keep_alpha=False)
        for segment in segments:
            if len(segment) < 2:
                continue
            phi_s, theta_s = segment[:, 0], segment[:, 1]
            figure.add_trace(
                go.Scatter3d(
                    x=radius * np.sin(theta_s) * np.cos(phi_s),
                    y=radius * np.sin(theta_s) * np.sin(phi_s),
                    z=radius * np.cos(theta_s),
                    mode="lines",
                    line=dict(color=color, width=4),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # An invisible trace, only for the energy colorbar
    figure.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(
                size=0,
                color=[e_min, e_max],
                colorscale=[[i / 255.0, mcolors.to_hex(cmap(i / 255.0))] for i in range(256)],
                cmin=e_min,
                cmax=e_max,
                colorbar=dict(
                    title=dict(text="Energy", font=dict(size=20)),
                    tickfont=dict(size=20),
                    len=0.8,
                    thickness=20,
                    x=1.02,
                    y=0.5,
                    yanchor="middle",
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    def axis(label: str) -> dict:
        return dict(
            title=dict(text=label, font=dict(size=20)),
            tickfont=dict(size=15),
            range=[-radius, radius],
            showspikes=False,
        )

    camera = dict(eye=dict(x=-1.2, y=1.2, z=1.2)) if great_circle else dict(eye=dict(x=1.2, y=-1.2, z=1.2))
    figure.update_layout(
        hovermode=False,
        scene=dict(xaxis=axis("M₁"), yaxis=axis("M₂"), zaxis=axis("M₃"), aspectmode="cube", camera=camera),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return figure


def plot_energy_levels_on_sphere(example: RigidBodyExample, M: CovecArray, path: str) -> None:
    """The Casimir sphere through M_0 with the energy level curves (no trajectory), saved as path (a .png)."""
    great_circle = len(largest_inertia_axes(example.Itensor)) == 2
    figure = energy_levels_on_sphere(example.Itensor, float(np.linalg.norm(M[0])), great_circle)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.write_image(path, width=1000, height=800, scale=2)
    print(f"Plot saved as '{path}'")


def plot_trajectory_on_sphere(example: RigidBodyExample, M: CovecArray, path: str) -> None:
    """The trajectory M_k on the Casimir sphere with the energy level curves, saved as path (a .png) and as an interactive .html file."""
    great_circle = len(largest_inertia_axes(example.Itensor)) == 2
    figure = energy_levels_on_sphere(example.Itensor, float(np.linalg.norm(M[0])), great_circle)
    figure.add_trace(
        go.Scatter3d(
            x=M[:, 0],
            y=M[:, 1],
            z=M[:, 2],
            mode="lines",
            line=dict(color="red", width=5),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.write_image(path, width=1000, height=800, scale=2)
    inertia = ", ".join(f"{value:g}" for value in np.diag(example.Itensor))
    figure.update_layout(
        title=dict(text=f"Evolution of Mₖ, I = diag({inertia})", font=dict(size=24), x=0.5, xanchor="center"),
        margin=dict(l=0, r=70, b=0, t=40),
    )
    figure.write_html(path.removesuffix(".png") + ".html")
    print(f"Plot saved as '{path}' (and .html)")
