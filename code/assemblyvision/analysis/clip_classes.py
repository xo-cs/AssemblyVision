"""Decoders without a phase order for the public train domain test (Week 12, rule w12-train-domain).

They take the cumulative class log likelihood [K, N+1] of the B zero shot setting
(EngineA.phase_likelihood: SigLIP full frame, SigLIP crop and VLM, equal weights) and use nothing else:
- best_class: one class for a clip;
- two_segments: two different classes and one change time, in either order;
- unordered: any number of segments of at least min_s, any class, a class may come back.
No file is read or written.
"""
from __future__ import annotations

import numpy as np


def best_class(ll: np.ndarray) -> tuple[int, float]:
    """(class, its lead over the second class in log likelihood)."""
    total = ll[:, -1] - ll[:, 0]
    order = np.argsort(-total, kind="stable")
    return int(order[0]), float(total[order[0]] - total[order[1]]) if len(order) > 1 else 0.0


def two_segments(ll: np.ndarray, fps: float, min_s: float = 2.0) -> tuple[int, int, int]:
    """(first class, second class, change frame): the best split with two different classes."""
    K, N = ll.shape[0], ll.shape[1] - 1
    if K < 2:
        raise ValueError("two segments need at least two classes")
    m = max(1, int(round(min_s * fps)))
    if N < 2 * m:
        m = max(1, N // 2)
    best = (-np.inf, 0, 1, m)
    for t in range(m, N - m + 1):
        head, tail = ll[:, t] - ll[:, 0], ll[:, N] - ll[:, t]
        a = np.argsort(-head, kind="stable")[:2]
        b = np.argsort(-tail, kind="stable")[:2]
        for i in a:
            for j in b:
                if i != j and head[i] + tail[j] > best[0]:
                    best = (float(head[i] + tail[j]), int(i), int(j), t)
    return best[1], best[2], best[3]


def unordered(ll: np.ndarray, fps: float, min_s: float = 5.0) -> list[tuple[int, int, int]]:
    """[(class, start frame, end frame)] covering all frames, every segment at least min_s long (a shorter
    video is one segment), two neighbours always of different classes. Exact dynamic programme."""
    K, N = ll.shape[0], ll.shape[1] - 1
    m = max(1, int(round(min_s * fps)))
    if K == 1 or N < m:
        return [(best_class(ll)[0], 0, N)]
    A = np.full((N + 1, K), -np.inf)
    start = np.zeros((N + 1, K), np.int32)
    prev = np.full((N + 1, K), -1, np.int32)
    keys = np.arange(K)
    for t in range(m, N + 1):
        u = t - m
        if u == 0:
            enter, pj = ll[:, t] - ll[:, 0], np.full(K, -1)
        else:
            j1 = int(np.argmax(A[u]))
            rest = A[u].copy()
            rest[j1] = -np.inf
            j2 = int(np.argmax(rest))
            pj = np.where(keys == j1, j2, j1)
            enter = A[u, pj] + ll[:, t] - ll[:, u]
        stay = A[t - 1] + ll[:, t] - ll[:, t - 1]
        take = enter > stay
        A[t] = np.where(take, enter, stay)
        start[t] = np.where(take, u, start[t - 1])
        prev[t] = np.where(take, pj, prev[t - 1])
    out = []
    k, t = int(np.argmax(A[N])), N
    while True:
        s = int(start[t, k])
        out.append((k, s, t))
        if s == 0:
            break
        k, t = int(prev[t, k]), s
    return out[::-1]
