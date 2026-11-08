"""RuleBasedEngine, Engine A (LLD 3.6): ordered process phases from boundary evidence.

Evidence on the run timeline (strengths are seg_a.strength_*):
  no_person  a stretch of at least seg_a.no_person_min_s with no stored track
             (title cards, but also close ups of parts; on screen text is never read)
  cut_pause  a shot cut with a subject pause of at least seg_a.long_pause_s nearby
  pause      such a long pause away from cuts (boundary at its middle)
  cut        any other shot cut

With seg_a.activity_change on, every point candidate (cut_pause, pause, cut)
gets strength = base + seg_a.ac_weight * activity change, where the change
(0 to 1, activity.py) compares the subject activity in the seg_a.ac_window_s
before and after the candidate. no_person keeps its strength. Off (default):
the strengths and the result are exactly the ones without this feature.

With seg_a.engine_b_weight > 0 (default 0 = off, nothing changes) the Engine B
boundary probability (boundary_tcn.py, on the run clock) is one more evidence
term: it is pooled by a centred moving average of engine_b.pool_s (0 = none,
so that a scorer trained on short actions does not fire on every action
change inside a phase), divided by its maximum over the active span (seg_a.engine_b_norm = "max",
so a model that is less confident on QiSens than on HA4M still reaches 1;
"none" uses it as it is), and every point
candidate gets strength += engine_b_weight * the highest such value within
seg_a.engine_b_window_s of it. With seg_a.engine_b_candidates the peaks of the
pooled probability (engine_b.peak_threshold, at least max(peak_min_gap_s,
pool_s) apart, not near another candidate) become extra point candidates of
kind engine_b with strength engine_b_weight * value. The report gives the firing rate:
peaks per minute of the raw probability and of the pooled, normalized evidence
(engine_b.peak_threshold; gap peak_min_gap_s, and max(peak_min_gap_s, pool_s) pooled).

With seg_a.embed_weight > 0 (default 0 = off) the Engine B embedding (the last
stage's unit length hidden features, boundary_tcn.change_score) gives a phase
level change score: at frame t the distance between the mean embedding of the
seg_a.embed_window_s before t and of the window after t, divided by its maximum
over the active span. Every point candidate gets strength += embed_weight * the
highest score within seg_a.engine_b_window_s of it.

With seg_a.start_detect (default off) phase 1 starts at the start of work, not at
the first person frame: the first frame t from which at least
seg_a.start_active_share of the frames in the next seg_a.start_sustain_s move
(motion energy >= pause_energy, the pause rule), moved on to the first moving frame
of that window. The time before it (an intro:
the worker is present but not working yet) is "other". If no such t lies within
seg_a.start_max_share of the active span, or the intro is shorter than
seg_a.start_min_intro_s (a short still start is not an intro), phase 1 starts at
the first person frame.

Phase mode (steps.step not empty): K phases in the order of steps.toml, each
at most once and never backwards. Leading and trailing stretches without a
person are other. Limits per phase: T = first to last frame with a person,
min_s = seg_a.min_factor * T / K and max_s = seg_a.max_factor * T / K, unless
steps.toml gives min_s or max_s for that phase.

  dp       (default) chooses exactly K - 1 boundaries that maximise the total
           strength with every phase inside its limits; ties go to the choice
           whose phase lengths are closest to T / K. Deviation from the LLD
           forward walk: a close up without a person inside a phase would end
           that phase in the forward walk, while dp weighs all candidates.
  forward  the LLD walk: phase k ends at the first no_person stretch at least
           min_s after it opened; if none comes before max_s, at the strongest
           other candidate in [min_s, max_s] (later on ties), else it is force
           closed at max_s with score 0 (TC-SEG-07). Phases never reached are
           missing, never invented. Also used when dp finds no valid choice.

Step score: the lower of the strengths at its start and end (run start and end
count 1.0). Stretches between steps longer than seg_a.other_min_s are other;
shorter ones join the step before.

Fallback (no vocabulary, AV-SEG-003): cut at the middle of every pause of at
least seg_a.fallback_pause_min_s, no_person stretches are other, the pieces are
segment_1..n.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import AvError
from ...phases import phase_defs_for
from ...storage.files import RunContext
from ...types import Step
from ..cleaning import SignalCleaner, _runs
from .activity import HOP_S, activity_change, video_stats
from .base import OTHER, SOURCE, RunTimeline, SegmentationEngine, check_contract


@dataclass
class Evidence:
    start: int        # previous phase ends here (frame)
    end: int          # next phase starts here (frame); equal to start for point evidence
    kind: str
    strength: float
    used: bool = False
    activity_change: float | None = None    # set only with seg_a.activity_change on
    engine_b: float | None = None           # set only with seg_a.engine_b_weight > 0
    embed_change: float | None = None       # set only with seg_a.embed_weight > 0


class RuleBasedEngine(SegmentationEngine):
    name = "A"

    def __init__(self, cfg, ctx: RunContext | None = None):
        self.cfg = cfg
        self.ctx = ctx
        self.s = cfg["seg_a"]
        self.cleaner = SignalCleaner(cfg)
        st = cfg["steps"]
        # phase names per data area: steps.toml for QiSens, Phase 1..N for public clips
        self.step_defs = phase_defs_for(list(st["step"]), ctx.source if ctx else None) if st["use"] else []
        self.vocabulary = [d["name"] for d in self.step_defs]
        self.report: dict = {}

    # evidence ---------------------------------------------------------------

    def pauses(self, tl: RunTimeline, min_s: float) -> list[tuple[int, int]]:
        still = np.nan_to_num(tl.energy, nan=np.inf) < self.s["pause_energy"]
        n = max(1, int(round(min_s * tl.fps)))
        return [(a, b) for a, b in _runs(still) if b - a >= n]

    def evidence(self, tl: RunTimeline, a: int, b: int) -> list[Evidence]:
        """Boundary candidates strictly inside the active span [a, b)."""
        s, fps = self.s, tl.fps
        n_min = int(round(s["no_person_min_s"] * fps))
        empty = [(p, q) for p, q in _runs(~tl.person[a:b]) if q - p >= n_min]
        empty = [(p + a, q + a) for p, q in empty]
        ev = [Evidence(p, q, "no_person", s["strength_no_person"]) for p, q in empty]

        def in_empty(f: int) -> bool:
            return any(p - 1 <= f <= q + 1 for p, q in empty)
        longp = [(p, q) for p, q in self.pauses(tl, s["long_pause_s"]) if a < (p + q) // 2 < b]
        consumed = set()
        w = int(round(s["cut_pause_window_s"] * fps))
        for c in tl.cuts:
            if not a < c < b or in_empty(c):
                continue
            near = [i for i, (p, q) in enumerate(longp) if p - w <= c <= q + w]
            if near:
                consumed.update(near)
                ev.append(Evidence(c, c, "cut_pause", s["strength_cut_pause"]))
            else:
                ev.append(Evidence(c, c, "cut", s["strength_cut"]))
        for i, (p, q) in enumerate(longp):
            m = (p + q) // 2
            if i not in consumed and not in_empty(m):
                ev.append(Evidence(m, m, "pause", s["strength_pause"]))
        ev.sort(key=lambda e: (e.start, e.end))
        if s["activity_change"] and tl.activity:
            self.add_activity_change(tl, ev, a, b)
        if s["engine_b_weight"] > 0 and tl.boundary_prob is not None:
            self.add_engine_b(tl, ev, a, b, empty)
            ev.sort(key=lambda e: (e.start, e.end))
        if s["embed_weight"] > 0 and tl.embedding is not None:
            self.add_embed_change(tl, ev, a, b)
        return ev

    def add_embed_change(self, tl: RunTimeline, ev: list[Evidence], a: int, b: int) -> None:
        from .boundary_tcn import change_score
        valid = tl.embedding_valid if tl.embedding_valid is not None else np.ones(tl.n_frames, bool)
        score = change_score(tl.embedding[:tl.n_frames], valid[:tl.n_frames], int(round(self.s["embed_window_s"] * tl.fps)))
        top = float(score[a:b].max()) if b > a else 0.0
        val = score / top if top > 0 else score
        w = int(round(self.s["engine_b_window_s"] * tl.fps))
        for e in ev:
            if e.kind == "no_person":
                continue
            lo, hi = max(a, e.start - w), min(b, e.end + w + 1)
            e.embed_change = round(float(val[lo:hi].max()) if hi > lo else 0.0, 4)
            e.strength = e.strength + self.s["embed_weight"] * e.embed_change

    def start_of_work(self, tl: RunTimeline, a: int, b: int) -> tuple[int, dict]:
        """(first frame of phase 1, report) under seg_a.start_detect (see the module doc)."""
        s = self.s
        moving = np.nan_to_num(tl.energy[:tl.n_frames], nan=-1.0) >= s["pause_energy"]
        n = max(1, int(round(s["start_sustain_s"] * tl.fps)))
        cs = np.r_[0, np.cumsum(moving)]
        last = a + int(s["start_max_share"] * (b - a))
        for t in range(a, min(last, b - n) + 1):
            if (cs[t + n] - cs[t]) >= s["start_active_share"] * n:
                t += int(np.argmax(moving[t:t + n]))           # the first moving frame of that window
                intro = round((t - a) / tl.fps, 3)
                if intro < s["start_min_intro_s"]:
                    return a, {"on": True, "found": False, "start_s": tl.t(a), "intro_s": 0.0, "candidate_s": intro,
                               "note": f"still start of {intro:g} s is shorter than start_min_intro_s"}
                return t, {"on": True, "found": True, "start_s": tl.t(t), "intro_s": intro}
        return a, {"on": True, "found": False, "start_s": tl.t(a), "intro_s": 0.0,
                   "note": f"no sustained activity within the first {s['start_max_share']:g} of the active span"}

    def engine_b_evidence(self, tl: RunTimeline, a: int, b: int) -> np.ndarray:
        """Pooled (engine_b.pool_s) and normalized (seg_a.engine_b_norm) Engine B probability [N]."""
        from .boundary_tcn import pool
        p = pool(np.nan_to_num(tl.boundary_prob[:tl.n_frames]), int(round(self.cfg["engine_b"]["pool_s"] * tl.fps)))
        if self.s["engine_b_norm"] == "max" and b > a:
            top = float(p[a:b].max())
            p = p / top if top > 0 else p
        return np.clip(p, 0.0, 1.0)

    def add_engine_b(self, tl: RunTimeline, ev: list[Evidence], a: int, b: int, empty: list[tuple[int, int]]) -> None:
        from .boundary_tcn import peaks
        val = self.engine_b_evidence(tl, a, b)
        raw = tl.boundary_prob[:tl.n_frames]
        w = int(round(self.s["engine_b_window_s"] * tl.fps))
        weight = self.s["engine_b_weight"]

        def near(lo: int, hi: int) -> float:
            lo, hi = max(a, lo - w), min(b, hi + w + 1)
            return float(val[lo:hi].max()) if hi > lo else 0.0
        if self.s["engine_b_candidates"]:
            from .boundary_tcn import pool
            e = self.cfg["engine_b"]
            pooled = pool(np.nan_to_num(raw), int(round(e["pool_s"] * tl.fps)))
            gap = int(round(max(e["peak_min_gap_s"], e["pool_s"]) * tl.fps))
            taken = [(x.start, x.end) for x in ev]
            for f in peaks(pooled[a:b], e["peak_threshold"], gap) + a:
                f = int(f)
                if not a < f < b or any(p - w <= f <= q + w for p, q in taken + empty):
                    continue
                ev.append(Evidence(f, f, "engine_b", 0.0))
        e_cfg, minutes = self.cfg["engine_b"], max((b - a) / tl.fps / 60.0, 1e-9)
        g_raw = int(round(e_cfg["peak_min_gap_s"] * tl.fps))
        g_pool = int(round(max(e_cfg["peak_min_gap_s"], e_cfg["pool_s"]) * tl.fps))
        self._b_rate = {"peaks_per_min_raw": round(len(peaks(np.nan_to_num(raw[a:b]), e_cfg["peak_threshold"], g_raw))
                                                   / minutes, 2),
                        "peaks_per_min_pooled": round(len(peaks(val[a:b], e_cfg["peak_threshold"], g_pool)) / minutes, 2)}
        for e in ev:
            if e.kind == "no_person":
                continue
            e.engine_b = round(near(e.start, e.end), 4)
            e.strength = e.strength + weight * e.engine_b

    def add_activity_change(self, tl: RunTimeline, ev: list[Evidence], a: int, b: int) -> None:
        win = max(1, int(round(self.s["ac_window_s"] * tl.fps)))
        norm = video_stats(tl.activity, a, b, win, int(round(HOP_S * tl.fps)))
        for e in ev:
            if e.kind == "no_person":
                continue
            e.activity_change = round(activity_change(tl.activity, e.start, e.end, a, b, win, norm), 4)
            e.strength = e.strength + self.s["ac_weight"] * e.activity_change

    # limits -----------------------------------------------------------------

    def limits(self, T_s: float, fps: float) -> list[dict]:
        K = len(self.step_defs)
        out = []
        for d in self.step_defs:
            lo = d.get("min_s", self.s["min_factor"] * T_s / K)
            hi = d.get("max_s", self.s["max_factor"] * T_s / K)
            out.append({"phase": d["name"], "min_s": round(lo, 3), "max_s": round(hi, 3),
                        "source": "steps.toml" if ("min_s" in d or "max_s" in d) else "derived",
                        "min_f": max(1, int(round(lo * fps))), "max_f": max(1, int(round(hi * fps)))})
        return out

    # solvers ----------------------------------------------------------------

    def solve_dp(self, ev: list[Evidence], a: int, b: int, lim: list[dict]) -> list[tuple] | None:
        """Exactly K - 1 boundaries, max total strength, every phase within its limits."""
        K = len(lim)
        if K == 1:
            return [(0, a, b, 1.0, 1.0)]
        M = len(ev)
        if M < K - 1:
            return None
        S = np.array([e.start for e in ev], np.int64)
        E = np.array([e.end for e in ev], np.int64)
        st = np.array([e.strength for e in ev], np.float64)
        target = (b - a) / K
        pen = 1e-3 / max(b - a, 1)          # tie break only: at most 1e-3 in total, far below strength steps

        def ok(dur, k):
            return (dur >= lim[k]["min_f"]) & (dur <= lim[k]["max_f"])
        NEG = -np.inf
        val = np.full((K - 1, M), NEG)
        back = np.full((K - 1, M), -1, np.int64)
        d0 = S - a
        val[0] = np.where(ok(d0, 0), st - pen * np.abs(d0 - target), NEG)
        for k in range(1, K - 1):
            for j in range(M):
                dur = S[j] - E                                   # phase k from E[i] to S[j]
                cand = np.where((np.arange(M) < j) & ok(dur, k) & np.isfinite(val[k - 1]),
                                val[k - 1] - pen * np.abs(dur - target), NEG)
                i = int(np.argmax(cand))
                if np.isfinite(cand[i]):
                    val[k, j] = cand[i] + st[j]
                    back[k, j] = i
        last = b - E
        fin = np.where(ok(last, K - 1) & np.isfinite(val[K - 2]), val[K - 2] - pen * np.abs(last - target), NEG)
        j = int(np.argmax(fin))
        if not np.isfinite(fin[j]):
            return None
        chosen = [j]
        for k in range(K - 2, 0, -1):
            j = int(back[k, j])
            chosen.append(j)
        chosen.reverse()
        phases, open_f, open_s = [], a, 1.0
        for k, j in enumerate(chosen):
            ev[j].used = True
            phases.append((k, open_f, int(S[j]), open_s, float(st[j])))
            open_f, open_s = int(E[j]), float(st[j])
        phases.append((K - 1, open_f, b, open_s, 1.0))
        return phases

    def solve_forward(self, ev: list[Evidence], a: int, b: int, lim: list[dict]) -> list[tuple]:
        K = len(lim)
        phases, open_f, open_s, k = [], a, 1.0, 0
        while k < K - 1:
            lo, hi = open_f + lim[k]["min_f"], open_f + lim[k]["max_f"]
            if lo >= b:
                break
            after = [e for e in ev if e.start >= lo and e.start > open_f]
            first_np = next((e for e in after if e.kind == "no_person"), None)
            if first_np is not None and first_np.start <= hi:
                pick = first_np
            else:
                window = [e for e in after if e.start <= hi]
                pick = max(reversed(window), key=lambda e: e.strength) if window else None
                if pick is None:
                    if hi >= b:
                        break
                    pick = Evidence(hi, hi, "forced", 0.0)
                    ev.append(pick)
            if pick.start >= b:
                break
            pick.used = True
            phases.append((k, open_f, pick.start, open_s, pick.strength))
            open_f, open_s, k = pick.end, pick.strength, k + 1
        phases.append((k, open_f, b, open_s, 1.0))
        return phases

    # steps --------------------------------------------------------------------

    def _assemble(self, tl: RunTimeline, pieces: list[tuple[str, int, int, float | None]]) -> list[Step]:
        """pieces: (label, start frame, end frame, score) in order; holes become other,
        short other stretches join the step before (or after, at the start)."""
        min_other = int(round(self.s["other_min_s"] * tl.fps))
        full, pos = [], 0
        for label, p, q, sc in pieces:
            if p > pos:
                full.append([OTHER, pos, p, None])
            if q > p:
                full.append([label, p, q, sc])
            pos = max(pos, q)
        if pos < tl.n_frames:
            full.append([OTHER, pos, tl.n_frames, None])
        merged: list[list] = []
        for piece in full:
            if piece[0] == OTHER and piece[2] - piece[1] < min_other and len(full) > 1:
                if merged:
                    merged[-1][2] = piece[2]
                    continue
                piece[0], piece[3] = "__join_next__", None
            if merged and merged[-1][0] == "__join_next__":
                piece[1] = merged.pop()[1]
            if merged and merged[-1][0] == piece[0] == OTHER:
                merged[-1][2] = piece[2]
                continue
            merged.append(piece)
        return [Step(lab, tl.t(p), tl.t(q), SOURCE, None if sc is None else round(float(sc), 3))
                for lab, p, q, sc in merged]

    def segment_run(self, tl: RunTimeline) -> list[Step]:
        idx = np.flatnonzero(tl.person)
        rep = {"engine": "A", "vocabulary": self.vocabulary, "fps": tl.fps, "duration_s": round(tl.duration_s, 3)}
        self.report = rep
        self._b_rate = {}
        if len(idx) == 0:
            rep.update(mode="phases" if self.vocabulary else "fallback", phases_expected=len(self.vocabulary),
                       phases_found=0, note="no person in the whole run", evidence=[])
            steps = [Step(OTHER, tl.t(0), tl.t(tl.n_frames), SOURCE, None)]
            check_contract(steps, tl, self.vocabulary)
            return steps
        a, b = int(idx[0]), int(idx[-1]) + 1
        if self.s["start_detect"]:
            a, rep["start_of_work"] = self.start_of_work(tl, a, b)
        T_s = (b - a) / tl.fps
        rep["active"] = {"start_s": tl.t(a), "end_s": tl.t(b), "T_s": round(T_s, 3)}
        ev = self.evidence(tl, a, b)
        if not self.vocabulary:
            if self.ctx is not None:
                self.ctx.notice(AvError("AV-SEG-003"), "segmentation")
            steps = self._fallback(tl, a, b, ev)
            rep.update(mode="fallback", segments=sum(s.label != OTHER for s in steps))
        else:
            lim = self.limits(T_s, tl.fps)
            solver = self.s["phase_solver"]
            phases = self.solve_dp(ev, a, b, lim) if solver == "dp" else None
            used = solver
            if phases is None:
                if solver == "dp":
                    used = "forward (dp found no choice inside the limits)"
                    for e in ev:
                        e.used = False
                phases = self.solve_forward(ev, a, b, lim)
            pieces = [(self.vocabulary[k], p, q, min(so, sc)) for k, p, q, so, sc in phases]
            steps = self._assemble(tl, pieces)
            rep.update(mode="phases", solver=solver, solver_used=used, K=len(lim),
                       limits=[{k: v for k, v in d.items() if k not in ("min_f", "max_f")} for d in lim],
                       phases_expected=len(lim), phases_found=len(phases),
                       note=f"found {len(phases)} of {len(lim)} phases")
        ev.sort(key=lambda e: (e.start, e.end))
        rep["evidence"] = [{"start_s": tl.t(e.start), "end_s": tl.t(e.end), "kind": e.kind,
                            "strength": round(e.strength, 4), "used": e.used,
                            **({} if e.activity_change is None else {"activity_change": e.activity_change}),
                            **({} if e.engine_b is None else {"engine_b": e.engine_b}),
                            **({} if e.embed_change is None else {"embed_change": e.embed_change})}
                           for e in ev]
        if self.s["activity_change"]:
            rep["activity_change"] = {"on": bool(tl.activity), "ac_weight": self.s["ac_weight"],
                                      "ac_window_s": self.s["ac_window_s"]}
        if self.s["engine_b_weight"] > 0:
            rep["engine_b"] = {"on": tl.boundary_prob is not None, "engine_b_weight": self.s["engine_b_weight"],
                               "pool_s": self.cfg["engine_b"]["pool_s"], "norm": self.s["engine_b_norm"],
                               "window_s": self.s["engine_b_window_s"], "candidates": self.s["engine_b_candidates"],
                               **getattr(self, "_b_rate", {}), **tl.boundary_info}
        if self.s["embed_weight"] > 0:
            rep["embed_change"] = {"on": tl.embedding is not None, "embed_weight": self.s["embed_weight"],
                                   "embed_window_s": self.s["embed_window_s"], "window_s": self.s["engine_b_window_s"]}
        rep["evidence_counts"] = {k: sum(e.kind == k for e in ev)
                                  for k in ("no_person", "cut_pause", "pause", "cut",
                                            *(("engine_b",) if self.s["engine_b_weight"] > 0 else ()), "forced")}
        check_contract(steps, tl, self.vocabulary)
        return steps

    def _fallback(self, tl: RunTimeline, a: int, b: int, ev: list[Evidence]) -> list[Step]:
        empty = [(e.start, e.end) for e in ev if e.kind == "no_person"]
        cuts = []
        for p, q in self.pauses(tl, self.s["fallback_pause_min_s"]):
            m = (p + q) // 2
            if a < p and q < b and not any(s <= m < t for s, t in empty):
                cuts.append(m)
        bounds = sorted([(p, q) for p, q in empty] + [(m, m) for m in cuts])
        pieces, pos, n = [], a, 0
        for p, q in bounds:
            if p > pos:
                n += 1
                pieces.append((f"segment_{n}", pos, p, None))
            pos = max(pos, q)
        if b > pos:
            n += 1
            pieces.append((f"segment_{n}", pos, b, None))
        return self._assemble(tl, pieces)
