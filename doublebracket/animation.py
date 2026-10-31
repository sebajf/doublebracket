"""
Videos of a run (they need ffmpeg): the rotating bodies, drawn as boxes, and their angular momentum vectors.

A rigid-body example is drawn as a 1 x 1 lattice. The videos are interpolated from the simulation grid to the frame times, so the frame rate does not affect the simulation.
"""

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import ndarray
from scipy.spatial.transform import Rotation, Slerp

from doublebracket.array_types import CovecArray, GroupElementArray, ScalarArray, TimeArray, Vec
from doublebracket.ddb import reconstruct
from doublebracket.diagnostics import energies
from doublebracket.examples import Example, LatticeExample, RigidBodyExample
from doublebracket.methods import METHODS
from doublebracket.retractions import Exp
from doublebracket.so3 import spatial_momenta


def box_dimensions(principal_moments: Vec) -> Vec:
    """
    The edges (a, b, c) of a homogeneous box of unit mass with the given principal moments of inertia, from
        I_1 = (b^2 + c^2) / 12,    I_2 = (a^2 + c^2) / 12,    I_3 = (a^2 + b^2) / 12.
    """
    I1, I2, I3 = principal_moments
    squares = 6 * np.array([I2 + I3 - I1, I1 + I3 - I2, I1 + I2 - I3])
    if np.any(squares <= 0):
        raise ValueError(
            "Unphysical principal moments of inertia: they do not satisfy the triangle inequality"
        )
    return np.sqrt(squares)


def box_faces(edges: Vec) -> Float[ndarray, "6 4 3"]:
    """The 6 faces (4 vertices each) of a box centered at the origin with the given edges. Faces 0, 1 are normal to e1, faces 2, 3 to e2, and faces 4, 5 to e3."""
    a, b, c = edges / 2
    vertices = np.array(
        [[-a, -b, -c], [a, -b, -c], [a, b, -c], [-a, b, -c], [-a, -b, c], [a, -b, c], [a, b, c], [-a, b, c]]
    )
    return vertices[[[0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 2, 3], [4, 5, 6, 7]]]


def _frame_times(t: TimeArray, fps: int, speedup: float) -> TimeArray:
    """Frame times with speedup units of simulated time per second of video."""
    numframes = int(np.round((t[-1] - t[0]) * fps / speedup)) + 1
    return np.clip(t[0] + speedup / fps * np.arange(numframes), t[0], t[-1])


def _lattice_axes(side: int, spacing: float, half_width: float, title: str):
    """A 3D figure for a side x side array of sites; site i = side*a + b is centered at (spacing b, -spacing a, 0)."""
    figure = plt.figure(figsize=(10, 9))
    ax = figure.add_subplot(111, projection="3d")
    limits = [
        (-half_width, spacing * (side - 1) + half_width),
        (-spacing * (side - 1) - half_width, half_width),
        (-half_width, half_width),
    ]
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect([upper - lower for lower, upper in limits])
    ax.set_axis_off()
    ax.view_init(elev=35, azim=-60)
    ax.set_title(title)
    centers = np.array([[spacing * b, -spacing * a, 0.0] for a in range(side) for b in range(side)])
    return figure, ax, centers


def _save(figure, update, numframes: int, fps: int, filename: str) -> None:
    def update_with_progress(frame: int) -> list:
        if frame % max(1, numframes // 10) == 0:
            print(f"    Progress: {100 * frame / numframes:.0f}%")
        return update(frame)

    print(f"  {numframes} frames ({fps} fps, {numframes / fps:.1f} s of video). Rendering...")
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    animation = FuncAnimation(figure, update_with_progress, frames=numframes, interval=1000 / fps, blit=False)
    animation.save(filename, writer=FFMpegWriter(fps=fps, bitrate=4000, codec="libx264"), dpi=100)
    plt.close(figure)
    print(f"  Animation saved as '{filename}'")


def _save_stills(figure, ax, update, numframes: int, hidden: list, filename: str) -> None:
    """
    Save the first and last frames as <filename without .mp4>_first.pdf and _last.pdf, without the title and the artists in hidden (text labels, trails).

    Both are cropped to the same box, the smallest one that holds the drawing in either frame (bbox_inches="tight" keeps the whole 3D axes).
    """
    title = ax.get_title()
    ax.set_title("")
    for artist in hidden:
        artist.set_visible(False)
    frames = {"first": 0, "last": numframes - 1}
    rows, cols = [], []
    for frame in frames.values():
        update(frame)
        figure.canvas.draw()
        drawn = np.any(np.asarray(figure.canvas.buffer_rgba())[..., :3] < 250, axis=-1)
        rows.extend(np.flatnonzero(drawn.any(axis=1))[[0, -1]])
        cols.extend(np.flatnonzero(drawn.any(axis=0))[[0, -1]])
    height = drawn.shape[0]
    pad = 4  # pixels
    box = Bbox.from_extents(min(cols) - pad, height - max(rows) - pad, max(cols) + pad, height - min(rows) + pad)
    box = box.transformed(figure.dpi_scale_trans.inverted())  # inches
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    stem = filename.removesuffix(".mp4")
    for suffix, frame in frames.items():
        update(frame)
        figure.savefig(f"{stem}_{suffix}.pdf", bbox_inches=box)
        print(f"  Still saved as '{stem}_{suffix}.pdf'")
    ax.set_title(title)
    for artist in hidden:
        artist.set_visible(True)


def _time_label(t: float, energy: float) -> str:
    return f"t = {t:.2f},   H(M) = {energy:.4f}"


def animate_boxes(
    t: TimeArray,
    R: Float[ndarray, "n N 3 3"],
    I_list: Float[ndarray, "N 3 3"],
    energy: ScalarArray,
    filename: str,
    title: str = "",
    fps: int = 30,
    speedup: float = 10.0,
    max_edge: float = 1.1,
    spacing: float = 1.8,
) -> None:
    """
    Video of N = side^2 rotating boxes; body i is drawn at row a, column b (i = side*a + b), with attitude R[:, i].

    Each box is the homogeneous box with the (diagonal) inertia tensor I_list[i]. All boxes are drawn at the same scale, with longest edge max_edge, and their faces are colored by body axis: e1 red, e2 green, e3 blue. The rotations are interpolated at the frame times by slerp.
    """
    print(f"\nCreating animation '{filename}'...")
    n_sites = len(I_list)
    side = round(np.sqrt(n_sites))
    edges = np.array([box_dimensions(np.diag(Itensor)) for Itensor in I_list])
    faces = [box_faces(e * max_edge / edges.max()) for e in edges]

    frame_times = _frame_times(t, fps, speedup)
    R_frames = np.stack(
        [Slerp(t, Rotation.from_matrix(R[:, i]))(frame_times).as_matrix() for i in range(n_sites)], axis=1
    )
    energy_frames = np.interp(frame_times, t, energy)

    figure, ax, centers = _lattice_axes(side, spacing, max_edge / 2 + 0.1, title)

    def rotated_faces(frame: int) -> list:
        return [face for i in range(n_sites) for face in faces[i] @ R_frames[frame, i].T + centers[i]]

    collection = Poly3DCollection(
        rotated_faces(0),
        facecolors=(["tab:red"] * 2 + ["tab:green"] * 2 + ["tab:blue"] * 2) * n_sites,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    ax.add_collection3d(collection)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=13)
    ax.text2D(
        0.02,
        0.02,
        "Face colors: body axes $e_1$ (red), $e_2$ (green), $e_3$ (blue)",
        transform=ax.transAxes,
        fontsize=10,
    )

    def update(frame: int) -> list:
        collection.set_verts(rotated_faces(frame))
        time_text.set_text(_time_label(frame_times[frame], energy_frames[frame]))
        return [collection, time_text]

    _save(figure, update, len(frame_times), fps, filename)


def animate_momenta(
    t: TimeArray,
    M: Float[ndarray, "n N 3"],
    energy: ScalarArray,
    filename: str,
    frame: Literal["body", "spatial"],
    title: str = "",
    fps: int = 30,
    speedup: float = 10.0,
    spacing: float = 1.8,
    max_radius: float = 0.65,
    trail_time: float = 3.0,
    video: bool = True,
    stills: bool = False,
) -> None:
    """
    Video of the angular momentum vectors of N = side^2 bodies, laid out as in animate_boxes. The longest vector is drawn with length max_radius, and a trail shows the path of each tip over the last trail_time time units.

    - frame="body": M holds the body momenta M_i. They move on spheres (the coadjoint orbits |M_i| = const, drawn as wireframes); short colored stubs mark the body axes (e1 red, e2 green, e3 blue).
    - frame="spatial": M holds the spatial momenta pi_i = R_i M_i, which are constant without dissipation. Gray segments mark the spatial axes.

    The vectors are interpolated linearly at the frame times, and rescaled to the interpolated norm.
    """
    print(f"\nCreating animation '{filename}'...")
    n_sites = M.shape[1]
    side = round(np.sqrt(n_sites))
    frame_times = _frame_times(t, fps, speedup)
    M_frames = np.apply_along_axis(lambda values: np.interp(frame_times, t, values), 0, M)
    norms = np.linalg.norm(M, axis=-1)
    norms_frames = np.apply_along_axis(lambda values: np.interp(frame_times, t, values), 0, norms)
    M_frames *= (norms_frames / np.linalg.norm(M_frames, axis=-1))[..., None]
    scale = max_radius / norms.max()
    energy_frames = np.interp(frame_times, t, energy)
    trail_frames = max(1, int(np.round(trail_time * fps / speedup)))

    figure, ax, centers = _lattice_axes(side, spacing, max_radius + 0.1, title)
    tips = centers + scale * M_frames  # shape (numframes, N, 3)

    # Static parts: orbit spheres and body axes (body frame), or spatial axes (spatial frame)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 73), np.linspace(0, np.pi, 37))
    unit_sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
    for i in range(n_sites):
        r = scale * norms[0, i]
        for axis, color in enumerate(["tab:red", "tab:green", "tab:blue"]):
            segment = np.zeros((2, 3))
            if frame == "body":
                segment[:, axis] = [r, 1.15 * r]
                ax.plot3D(*(centers[i] + segment).T, color=color, linewidth=2)
            else:
                segment[:, axis] = [-r, r]
                ax.plot3D(*(centers[i] + segment).T, color="gray", linewidth=0.6, alpha=0.5)
        if frame == "body":
            ax.plot_wireframe(*(centers[i][:, None, None] + r * unit_sphere), rstride=6, cstride=6, color="gray", linewidth=0.3, alpha=0.3)

    # Moving parts
    arrows = [ax.plot3D([], [], [], color="black", linewidth=2)[0] for _ in range(n_sites)]
    heads = [ax.plot3D([], [], [], marker="o", markersize=4, color="black")[0] for _ in range(n_sites)]
    trails = [ax.plot3D([], [], [], color="tab:orange", linewidth=1, alpha=0.8)[0] for _ in range(n_sites)]
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=13)
    legend = (
        "Arrows: body angular momenta $M_i$. Body axes: $e_1$ (red), $e_2$ (green), $e_3$ (blue)"
        if frame == "body"
        else r"Arrows: spatial angular momenta $\pi_i = R_i M_i$. Gray: spatial axes"
    )
    legend_text = ax.text2D(0.02, 0.02, legend, transform=ax.transAxes, fontsize=10)

    def update(k: int) -> list:
        first = max(0, k - trail_frames)
        for i in range(n_sites):
            arrows[i].set_data_3d(*np.stack([centers[i], tips[k, i]]).T)
            heads[i].set_data_3d(*tips[k, i][:, None])
            trails[i].set_data_3d(*tips[first : k + 1, i].T)
        time_text.set_text(_time_label(frame_times[k], energy_frames[k]))
        return [*arrows, *heads, *trails, time_text]

    if stills:
        _save_stills(figure, ax, update, len(frame_times), [time_text, legend_text, *trails], filename)
    if video:
        _save(figure, update, len(frame_times), fps, filename)
    else:
        plt.close(figure)


def attitudes(example: Example, method_key: str) -> tuple[TimeArray, CovecArray, GroupElementArray]:
    """
    Run a method and reconstruct the attitudes from R_0 = I, by R_{k+1} = R_k tau(h xi_k).

    For the DDB methods, tau is the method's retraction and xi_k are its stage velocities, so R_k are the discrete configurations of the method. For the other methods, tau = exp and xi_k = dH/dmu(M_k).
    Returns (t, M, R), with R of shape (n, *M0.shape[:-1], 3, 3).
    """
    print(f"Solving with {METHODS[method_key].full_name}...")
    trajectory = METHODS[method_key].run(example)
    R0 = np.broadcast_to(np.eye(3), (*example.M0.shape[:-1], 3, 3))
    if trajectory.xi is not None:
        R = reconstruct(trajectory.retraction, R0, trajectory.xi, example.h)
    else:
        xi = np.array([example.model.inverse_legendre(Mk) for Mk in trajectory.M[:-1]])
        R = reconstruct(Exp(), R0, xi, example.h)
    return trajectory.t, trajectory.M, R


def create_animations(
    example: Example,
    method_key: str = "DDB cay",
    boxes: bool = True,
    momenta: bool = True,
    momentum_frame: Literal["body", "spatial"] = "spatial",
    fps: int = 30,
    speedup: float = 10.0,
    video: bool = True,
    stills: bool = False,
) -> None:
    """
    Videos of one method's run, saved in plots/<example name>/<method short name>/:
    rotating_boxes_<method>.mp4 (if boxes) and <momentum_frame>_momenta_<method>.mp4 (if momenta).
    With stills, the first and last frames of the momenta video are also saved, without labels or trails, as <video name>_first.pdf and <video name>_last.pdf. With video=False, only the stills are made.
    """
    method = METHODS[method_key]
    t, M, R = attitudes(example, method_key)
    energy = energies(example.model, M)
    outdir = f"plots/{example.name}/{method.short_name}"
    if isinstance(example, LatticeExample):
        title = f"{example.side}x{example.side} lattice of rigid bodies, {method.label}, alpha = {example.alpha}"
        I_list = example.I_list
    else:
        assert isinstance(example, RigidBodyExample)
        title = f"Rigid body, {method.label}, alpha = {example.alpha}"
        I_list, M, R = example.Itensor[None], M[:, None], R[:, None]  # one body

    if boxes:
        animate_boxes(t, R, I_list, energy, f"{outdir}/rotating_boxes_{method.short_name}.mp4", title, fps, speedup)
    if momenta:
        vectors = spatial_momenta(R, M) if momentum_frame == "spatial" else M
        animate_momenta(
            t,
            vectors,
            energy,
            f"{outdir}/{momentum_frame}_momenta_{method.short_name}.mp4",
            momentum_frame,
            title,
            fps,
            speedup,
            video=video,
            stills=stills,
        )
