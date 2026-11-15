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
    fig = go.Figure()
    for name, idx, scale in (("hands", HANDS, "Reds"), ("feet", FEET, "Blues")):
        c, s = _voxels(J[:, idx].reshape(-1, 3), voxel_m, fps)
        if not len(c):
            continue
        fig.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 2], z=c[:, 1], mode="markers", name=f"{name} (s per voxel)",
                                   marker=dict(size=3 + 12 * np.sqrt(s / s.max()), color=s, colorscale=scale,
                                               opacity=0.7, showscale=False),
                                   hovertemplate="%{marker.color:.1f} s<extra>" + name + "</extra>"))
    fig.update_layout(scene=dict(xaxis_title="x (m)", yaxis_title="z (m)", zaxis_title="height (m)", aspectmode="data"),
                      margin=dict(l=0, r=0, t=30, b=0), title=f"Hands and feet, {voxel_m * 100:.0f} cm voxels", height=560)
    return fig


def floor_map(piece: dict, fps: float, cell_m: float = 0.1):
    import plotly.graph_objects as go
    feet = piece["joints"][:, FEET].reshape(-1, 3)
    feet = feet[np.isfinite(feet).all(1)]
    ground = float(np.percentile(feet[:, 1], 5)) if len(feet) else 0.0
    if not len(feet):
        return go.Figure()
    x, z = feet[:, 0], feet[:, 2]
    def edges(v):                                       # at least one cell, the last edge past the maximum
        a = np.floor(v.min() / cell_m) * cell_m
        return a + cell_m * np.arange(int(np.floor((v.max() - a) / cell_m)) + 2)
    bx, bz = edges(x), edges(z)
    h, _, _ = np.histogram2d(z, x, bins=[bz, bx])
    fig = go.Figure(go.Heatmap(z=h / fps / 2, x=bx[:-1] + cell_m / 2, y=bz[:-1] + cell_m / 2, colorscale="Inferno",
                               colorbar=dict(title="s"), hovertemplate="x %{x:.1f} m, z %{y:.1f} m: %{z:.1f} s<extra></extra>"))
    fig.update_layout(title=f"Feet on the ground plane (y = {ground:.2f} m), {cell_m * 100:.0f} cm cells",
                      xaxis_title="x (m)", yaxis_title="z (m)", yaxis_scaleanchor="x", height=460,
                      margin=dict(l=40, r=10, t=40, b=40))
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
    fig = go.Figure(data=traces(idx[0]),
                    frames=[go.Frame(data=traces(i), name=clock(frames[i] / fps)) for i in idx])
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
