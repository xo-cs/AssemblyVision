"""3D and 4D views of the body3d stage (Week 11): plotly figures from <run>/body3d/*.npz.

The joints are in GVHMR's gravity aligned world frame of one piece of a shot (y up, metres; the frame
starts again at every cut), so every view is per piece. Plotly's vertical axis is the world y.
- density_3d: where the hands (wrists) and feet (ankles) spend time, as voxels of voxel_m (marker size
  and colour = seconds in the voxel).
- floor_map: the feet on the ground plane (the 5th percentile of the ankle height), a 2D histogram in
  metres, seconds per cell.
- skeleton_4d: the 17 joint skeleton animated over the piece (at most anim_fps frames per second),
  with a trail of the wrists over trail_s; the slider shows the video time.
"""
from __future__ import annotations

import numpy as np

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]
HANDS, FEET = (9, 10), (15, 16)
# colour scales without a white end, so short stays stay visible on the light background
SCALES = {"hands": [[0, "#f6a6a1"], [1, "#9b1c1c"]], "feet": [[0, "#9ecae1"], [1, "#08306b"]]}


def ground_level(J: np.ndarray) -> float:
    """The ground plane: the 5th percentile of the ankle height (GVHMR's y = 0 is close to it)."""
    a = J[:, FEET, 1].ravel()
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 5)) if len(a) else 0.0


def _ground(J: np.ndarray, pad: float = 0.3):
    """A light ground plane under the motion (plotly axes: x, z, height)."""
    import plotly.graph_objects as go
    pts = J.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1)]
    x0, x1 = pts[:, 0].min() - pad, pts[:, 0].max() + pad
    z0, z1 = pts[:, 2].min() - pad, pts[:, 2].max() + pad
    g = ground_level(J)
    return go.Mesh3d(x=[x0, x1, x1, x0], y=[z0, z0, z1, z1], z=[g] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                     color="#b0b7c3", opacity=0.35, name="ground", hoverinfo="skip", showlegend=True)


def clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _voxels(pts: np.ndarray, voxel_m: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """(voxel centres [M, 3], seconds [M]) of points [N, 3]."""
    pts = pts[np.isfinite(pts).all(1)]
    if not len(pts):
        return np.zeros((0, 3)), np.zeros(0)
    keys, counts = np.unique(np.floor(pts / voxel_m).astype(np.int64), axis=0, return_counts=True)
    return (keys + 0.5) * voxel_m, counts / fps


def density_3d(piece: dict, fps: float, voxel_m: float = 0.1):
    import plotly.graph_objects as go
    J = piece["joints"]
    fig = go.Figure(_ground(J))
    for name, idx in (("hands", HANDS), ("feet", FEET)):
        c, s = _voxels(J[:, idx].reshape(-1, 3), voxel_m, fps)
        if not len(c):
            continue
        fig.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 2], z=c[:, 1], mode="markers", name=f"{name} (s per voxel)",
                                   marker=dict(size=4 + 12 * np.sqrt(s / s.max()), color=s, colorscale=SCALES[name],
                                               cmin=0, cmax=float(s.max()), opacity=0.85, showscale=False,
                                               line=dict(width=0)),
                                   hovertemplate="%{marker.color:.1f} s<extra>" + name + "</extra>"))
    fig.update_layout(scene=dict(xaxis_title="x (m)", yaxis_title="z (m)", zaxis_title="height (m)", aspectmode="data"),
                      margin=dict(l=0, r=0, t=30, b=0), title=f"Hands and feet, {voxel_m * 100:.0f} cm voxels", height=560)
    return fig


def floor_map(piece: dict, fps: float, cell_m: float = 0.1):
    import plotly.graph_objects as go
    feet = piece["joints"][:, FEET].reshape(-1, 3)
    feet = feet[np.isfinite(feet).all(1)]
    ground = ground_level(piece["joints"])
    if not len(feet):
        return go.Figure()
    x, z = feet[:, 0], feet[:, 2]
    def edges(v):                                       # at least one cell, the last edge past the maximum
        a = np.floor(v.min() / cell_m) * cell_m
        return a + cell_m * np.arange(int(np.floor((v.max() - a) / cell_m)) + 2)
    bx, bz = edges(x), edges(z)
    h, _, _ = np.histogram2d(z, x, bins=[bz, bx])
    sec = np.where(h > 0, h / fps / 2, np.nan)            # cells never stepped on stay empty (not "0 s" black)
    xs, zs = bx[:-1] + cell_m / 2, bz[:-1] + cell_m / 2
    # the hover text is made here: a %{z:.1f} template is shown literally by some plotly.js builds (Gradio's)
    text = [[f"x {x:.1f} m, z {z:.1f} m: {v:.1f} s" if np.isfinite(v) else "" for x, v in zip(xs, row)]
            for z, row in zip(zs, sec)]
    fig = go.Figure(go.Heatmap(z=sec, x=xs, y=zs, colorscale="YlOrRd", text=text, hoverinfo="text",
                               colorbar=dict(title="s", x=1.0, xpad=4, thickness=14)))
    fig.update_layout(title=f"Feet on the ground plane (y = {ground:.2f} m), {cell_m * 100:.0f} cm cells, "
                            f"seconds per cell", plot_bgcolor="#f4f5f7",
                      xaxis_title="x (m)", yaxis_title="z (m)", yaxis_scaleanchor="x", height=460,
                      margin=dict(l=40, r=90, t=60, b=40), modebar=dict(orientation="v"))   # the toolbar clear of the colour bar
    return fig


def skeleton_4d(piece: dict, fps: float, anim_fps: float = 10.0, trail_s: float = 1.0):
    import plotly.graph_objects as go
    J, frames = piece["joints"], piece["frames"]
    step = max(1, int(round(fps / anim_fps)))
    idx = np.arange(0, len(J), step)
    lo, hi = np.nanmin(J.reshape(-1, 3), 0), np.nanmax(J.reshape(-1, 3), 0)
    trail = max(1, int(round(trail_s * fps)))

    def traces(i):
        p = J[i]
        bx, by, bz = [], [], []
        for a, b in BONES:
            bx += [p[a, 0], p[b, 0], None]
            by += [p[a, 2], p[b, 2], None]
            bz += [p[a, 1], p[b, 1], None]
        out = [go.Scatter3d(x=bx, y=by, z=bz, mode="lines", line=dict(width=6, color="#3b6ea5"), name="skeleton"),
               go.Scatter3d(x=p[:, 0], y=p[:, 2], z=p[:, 1], mode="markers", marker=dict(size=3, color="#1d3557"),
                            name="joints")]
        a = max(0, i - trail)
        for j, col in zip(HANDS, ("#e63946", "#f4a261")):
            t = J[a:i + 1, j]
            out.append(go.Scatter3d(x=t[:, 0], y=t[:, 2], z=t[:, 1], mode="lines", line=dict(width=3, color=col),
                                    name="left wrist trail" if j == 9 else "right wrist trail"))
        return out
    t0 = frames[0] / fps
    fig = go.Figure(data=[*traces(idx[0]), _ground(J)],
                    frames=[go.Frame(data=traces(i), name=clock(frames[i] / fps), traces=list(range(4))) for i in idx])
    fig.update_layout(
        scene=dict(xaxis=dict(range=[lo[0], hi[0]], title="x (m)"), yaxis=dict(range=[lo[2], hi[2]], title="z (m)"),
                   zaxis=dict(range=[min(lo[1], 0), hi[1]], title="height (m)"), aspectmode="data"),
        height=600, margin=dict(l=0, r=0, t=30, b=0), title=f"Skeleton from {clock(t0)} (video time)",
        updatemenus=[dict(type="buttons", x=0, y=0, xanchor="left", yanchor="top", buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=1000 / anim_fps, redraw=True),
                                                                  fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(x=0.1, len=0.9, currentvalue=dict(prefix="video time "), steps=[
            dict(label=clock(frames[i] / fps), method="animate",
                 args=[[clock(frames[i] / fps)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for i in idx])])
    return fig


def player_data(piece: dict, fps: float, voxel_m: float = 0.1, cell_m: float = 0.1) -> dict:
    """The 3D player's data of one part (ui/static/player.js): joints per frame, the voxels the hands
    (kind 0) and feet (kind 1) visit and their ids per frame, the floor cells of the feet per frame."""
    J = np.asarray(piece["joints"], np.float64)
    L = len(J)
    ok = np.isfinite(J).all(-1)
    pts = J[:, HANDS + FEET]                                   # [L, 4, 3]
    keys = np.floor(pts / voxel_m).astype(np.int64)
    kinds = np.array([0, 0, 1, 1])
    table: dict = {}
    vox_ids = np.full((L, 4), -1, np.int64)
    for f in range(L):
        for j in range(4):
            if ok[f, HANDS + FEET][j]:
                k = (int(kinds[j]), *map(int, keys[f, j]))
                vox_ids[f, j] = table.setdefault(k, len(table))
    order = sorted(table, key=table.get)
    centres = np.array([[(a + 0.5) * voxel_m for a in k[1:]] for k in order]) if order else np.zeros((0, 3))
    feet = J[:, FEET]
    fx, fz = feet[..., 0], feet[..., 2]
    good = np.isfinite(fx) & np.isfinite(fz)
    x0 = np.floor(np.nanmin(fx[good]) / cell_m) * cell_m if good.any() else 0.0
    z0 = np.floor(np.nanmin(fz[good]) / cell_m) * cell_m if good.any() else 0.0
    nx = int(np.floor((np.nanmax(fx[good]) - x0) / cell_m)) + 1 if good.any() else 1
    nz = int(np.floor((np.nanmax(fz[good]) - z0) / cell_m)) + 1 if good.any() else 1
    cx = np.floor((fx - x0) / cell_m).astype(np.int64)
    cz = np.floor((fz - z0) / cell_m).astype(np.int64)
    cell_ids = np.where(good, cz * nx + cx, -1)
    flat = J[ok].reshape(-1, 3) if ok.any() else np.zeros((1, 3))
    lo, hi = np.nanmin(J.reshape(-1, 3), 0), np.nanmax(J.reshape(-1, 3), 0)
    return {"fps": fps, "frames": [int(x) for x in piece["frames"]], "joints": np.round(np.nan_to_num(J), 3).ravel().tolist(),
            "voxel_m": voxel_m, "voxels": np.round(centres, 3).ravel().tolist(), "vox_kind": [k[0] for k in order],
            "vox_ids": vox_ids.tolist(), "cell_m": cell_m, "cells_x": nx, "cells_z": nz, "cell_ids": cell_ids.tolist(),
            "ground": round(ground_level(J), 3), "center": np.round((lo + hi) / 2, 3).tolist(),
            "extent": round(float(max(hi[0] - lo[0], hi[2] - lo[2], 1.0)), 3), "n_points": int(len(flat))}


def floor_png(piece: dict, fps: float, cell_m: float = 0.1, title: str = "") -> np.ndarray | None:
    """The floor map of one part as an RGB image (matplotlib): seconds per cell of the feet on the ground
    plane, in metres; cells never stepped on stay empty."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    feet = np.asarray(piece["joints"])[:, FEET].reshape(-1, 3)
    feet = feet[np.isfinite(feet).all(1)]
    if not len(feet):
        return None
    x, z = feet[:, 0], feet[:, 2]

    def edges(v):
        a = np.floor(v.min() / cell_m) * cell_m
        return a + cell_m * np.arange(int(np.floor((v.max() - a) / cell_m)) + 2)
    bx, bz = edges(x), edges(z)
    h, _, _ = np.histogram2d(z, x, bins=[bz, bx])
    sec = np.where(h > 0, h / fps / 2, np.nan)
    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=110)
    im = ax.imshow(sec, origin="lower", extent=[bx[0], bx[-1], bz[0], bz[-1]], cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("z (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#f4f5f7")
    ax.set_title(title, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("s", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
