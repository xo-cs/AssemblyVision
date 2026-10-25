"""Metric formulas (LLD 3.11). Plain functions that read no file.

Segmentation metrics follow the evaluation code of MS TCN (Y. Abu Farha and
J. Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network for Action
Segmentation", CVPR 2019; eval.py in github.com/yabufarha/ms-tcn). That code
takes the edit score and the segmental F1@k from C. Lea et al., "Temporal
Convolutional Networks for Action Segmentation and Detection", CVPR 2017.
get_labels_start_end_time, levenstein, edit_score and f_score are ports of
those functions, including their details: the background class is removed for
edit and F1 but counted in frame accuracy, and the end of the last segment is
its last frame index (not one past it), as in the original.

Boundary error is not an MS TCN metric; boundary_mae is our definition.
"""
from __future__ import annotations

import numpy as np

BG = ("other",)


def frames_from_steps(steps: list, duration_s: float, fps: float, bg: str = "other") -> list[str]:
    """Label of every sampled frame i (time i / fps) in [0, duration); bg where no step covers it.
    steps: objects with label, start_s, end_s."""
    n = int(round(duration_s * fps))
    t = np.arange(n) / fps
    out = np.full(n, bg, dtype=object)
    for s in sorted(steps, key=lambda s: s.start_s):
        out[(t >= s.start_s) & (t < s.end_s)] = s.label
    return list(out)


def get_labels_start_end_time(frame_wise_labels, bg_class=BG):
    labels, starts, ends = [], [], []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row, n_col = len(p), len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float64)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    if norm:
        if max(m_row, n_col) == 0:      # both empty (not handled in the original): identical
            return 100.0
        return (1 - D[-1, -1] / max(m_row, n_col)) * 100
    return D[-1, -1]


def edit_score(recognized, ground_truth, norm=True, bg_class=BG) -> float:
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return float(levenstein(P, Y, norm))


def f_score(recognized, ground_truth, overlap, bg_class=BG) -> tuple[float, float, float]:
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
    tp, fp = 0, 0
    hits = np.zeros(len(y_label))
    y_start, y_end = np.array(y_start), np.array(y_end)
    for j in range(len(p_label)):
        if not y_label:                 # no GT segment (not handled in the original): all false
            fp += 1
            continue
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * np.array([p_label[j] == y_label[x] for x in range(len(y_label))])
        idx = int(np.array(IoU).argmax())
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    """As in MS TCN eval.py: 2PR/(P+R) * 100, 0 when a denominator is 0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall) * 100


def f1_overlap(recognized, ground_truth, overlap, bg_class=BG) -> float:
    return f1_from_counts(*f_score(recognized, ground_truth, overlap, bg_class))


def frame_accuracy(recognized, ground_truth) -> float:
    """Percent of frames with an equal label, background included (MS TCN)."""
    if len(ground_truth) == 0:
        return 0.0
    return 100.0 * sum(a == b for a, b in zip(recognized, ground_truth)) / len(ground_truth)


def step_boundaries(steps: list, duration_s: float, bg: str = "other", tol: float = 1e-6) -> list[float]:
    """Start and end of every non background step, without 0 and the duration."""
    out = set()
    for s in steps:
        if s.label == bg:
            continue
        for t in (s.start_s, s.end_s):
            if tol < t < duration_s - tol:
                out.add(round(float(t), 3))
    return sorted(out)


def boundary_mae(pred_steps: list, gt_steps: list, duration_s: float, bg: str = "other",
                 edge_tol_s: float = 0.0) -> tuple[float | None, list[dict]]:
    """Mean absolute boundary error (our definition): for every GT boundary the
    distance to the nearest predicted boundary, averaged. None without GT or
    predicted boundaries. edge_tol_s drops boundaries this close to 0 or the end."""
    gt_b = step_boundaries(gt_steps, duration_s, bg, edge_tol_s or 1e-6)
    pr_b = step_boundaries(pred_steps, duration_s, bg, edge_tol_s or 1e-6)
    if not gt_b or not pr_b:
        return None, [{"gt_s": g, "pred_s": None, "error_s": None} for g in gt_b]
    pr = np.array(pr_b)
    rows = []
    for g in gt_b:
        k = int(np.argmin(np.abs(pr - g)))
        rows.append({"gt_s": g, "pred_s": float(pr[k]), "error_s": round(abs(float(pr[k]) - g), 3)})
    return round(float(np.mean([r["error_s"] for r in rows])), 3), rows


def cohen_kappa(a, b) -> tuple[float, float, float]:
    """(kappa, observed agreement po, chance agreement pe) of two label sequences."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("kappa needs two non empty sequences of equal length")
    po = float(np.mean(a == b))
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    pe = float(sum(np.mean(a == l) * np.mean(b == l) for l in labels))
    if pe >= 1.0:
        return (1.0 if po >= 1.0 else 0.0), po, pe
    return (po - pe) / (1 - pe), po, pe
