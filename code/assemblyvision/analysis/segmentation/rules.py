"""RuleBasedEngine, Engine A (LLD 3.6): ordered process phases from boundary evidence.

Evidence on the run timeline (strengths are seg_a.strength_*):
  no_person  a stretch of at least seg_a.no_person_min_s with no stored track
             (title cards, but also close ups of parts; on screen text is never read)
  cut_pause  a shot cut with a subject pause of at least seg_a.long_pause_s nearby
  pause      such a long pause away from cuts (boundary at its middle)
  cut        any other shot cut

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
from .base import OTHER, SOURCE, RunTimeline, SegmentationEngine, check_contract


@dataclass
class Evidence:
    start: int        # previous phase ends here (frame)
    end: int          # next phase starts here (frame); equal to start for point evidence
    kind: str
    strength: float
    used: bool = False


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
        return ev

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
        pen = 1e-3 / max(b - a, 1)          # tie break only: strengths differ by >= 0.2 in practice

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
        if len(idx) == 0:
            rep.update(mode="phases" if self.vocabulary else "fallback", phases_expected=len(self.vocabulary),
                       phases_found=0, note="no person in the whole run", evidence=[])
            steps = [Step(OTHER, tl.t(0), tl.t(tl.n_frames), SOURCE, None)]
            check_contract(steps, tl, self.vocabulary)
            return steps
        a, b = int(idx[0]), int(idx[-1]) + 1
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
                            "strength": e.strength, "used": e.used} for e in ev]
        rep["evidence_counts"] = {k: sum(e.kind == k for e in ev)
                                  for k in ("no_person", "cut_pause", "pause", "cut", "forced")}
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
