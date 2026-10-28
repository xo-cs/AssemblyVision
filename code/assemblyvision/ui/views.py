"""HTML building blocks of the interface (page headers, cards, KPI cards,
tables, pills, notes) and the pure helpers behind them: notice grouping,
readable metric names, the neutral QiSens video name, the trunk band colours,
which runs the page lists and the GPU and live state of the sidebar.

Everything returns HTML strings for gr.HTML; the look lives in style.css
(classes av-*). Text from runs is always escaped here.
"""
from __future__ import annotations

import base64
import html
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from ..analysis.motion import BAND_COLORS

# readable names of metrics.csv keys (motion.MotionAnalyzer); the key itself is
# shown beside the name in small mono text
METRIC_NAMES = {
    "duration_s": "Duration",
    "duration_dev": "Deviation from the cycle median",
    "movement_path": "Walking path",
    "bend_mean_deg": "Mean trunk angle",
    "repetition_share": "Share of repetitive motion",
    "repetition_period_s": "Repetition period",
    "repetition_count": "Repetitions",
    "hand_travel_l": "Left hand travel",
    "hand_travel_r": "Right hand travel",
    "hand_travel_l_torso": "Left hand travel (torso lengths)",
    "hand_travel_r_torso": "Right hand travel (torso lengths)",
    "two_hand_simultaneous": "Both hands active together",
    "two_hand_use_ratio": "Hand use ratio (weaker to stronger)",
    "two_hand_speed_corr": "Hand speed correlation",
    "idle_time_s": "Idle time",
    "idle_share": "Idle share",
    "gaze_down_share": "Gaze down share",
}
GAZE_NAMES = {"front": "front", "left": "left", "right": "right", "away": "away", "side": "to the side"}

# short titles for notice groups whose texts differ per item
NOTICE_TITLES = {
    "AV-MOT-001": "metrics not reported, too few frames with the needed keypoints",
    "AV-MOT-002": "metrics not reported, the step is too short",
    "AV-MOT-004": "a metric failed for a step",
    "AV-TRK-001": "the tracked worker was lost for a while",
    "AV-POS-003": "pose estimation failed on some frames",
    "AV-EVL-004": "no labelled steps in a part of the video",
    "AV-RUN-006": "a setting is not supported, the default is used",
}
_SHOT_REF = re.compile(r"\s*\b(?:of|in|for)?\s*shot\s+run_\d{8}_\d{6}_[0-9a-f]{4}-(s\d{3,})")
_SHOT_ID = re.compile(r"run_\d{8}_\d{6}_[0-9a-f]{4}-(s\d{3,})")

STATUS_WORDS = {"queued": "Queued", "running": "Running", "done": "Done", "partial": "Partial", "failed": "Failed",
                "pending": "Waiting", "skipped": "Not run", "lost": "Stopped"}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


# --- names -----------------------------------------------------------------------------


def metric_label(key: str, bands: list[float] | None = None) -> str:
    """Readable name of a metric key, for example bend_b1_s -> 'Time in B1 (20 to 45°)'."""
    m = re.fullmatch(r"bend_b([0-3])_s", key)
    if m:
        b = int(m.group(1))
        lo, mid, hi = (bands or [20, 45, 60])[:3]
        rng = [f"under {lo:g}°", f"{lo:g} to {mid:g}°", f"{mid:g} to {hi:g}°", f"{hi:g}° or more"][b]
        return f"Time in B{b} ({rng})"
    m = re.fullmatch(r"gaze_([a-z]+)_share", key)
    if m and key not in METRIC_NAMES:
        return f"Gaze {GAZE_NAMES.get(m.group(1), m.group(1))} share"
    return METRIC_NAMES.get(key, key.replace("_", " ").capitalize())


def metric_value(value, unit: str) -> str:
    """Value with its unit in reading form: ratios as percent, angles with °."""
    if value is None:
        return "n/a"
    v = float(value)
    if unit == "ratio":
        return f"{100 * v:.0f} %"
    if unit == "deg":
        return f"{v:.0f}°"
    if unit == "count":
        return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"
    num = f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}" if abs(v) >= 1 else f"{v:.2f}".rstrip("0").rstrip(".")
    unit = {"torso": "torso lengths"}.get(unit, unit)
    return f"{num} {unit}".strip()


def qisens_alias(duration_s: float | None) -> str:
    """Neutral display name of a QiSens video: the real file name stays on disk only."""
    if duration_s is None or duration_s <= 0:
        return "QiSens video"
    if duration_s < 60:
        return f"QiSens video ({duration_s:.0f} s)"
    return f"QiSens video ({duration_s / 60:.1f} min)"


def video_label(source: str, name: str, duration_s: float | None = None) -> str:
    return qisens_alias(duration_s) if source == "qisens" else name


def unique_labels(labels: list[str]) -> list[str]:
    """Number repeated labels: 'QiSens video (9.9 min)', 'QiSens video 2 (9.9 min)'."""
    seen: dict[str, int] = {}
    out = []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        n = seen[lab]
        if n > 1:
            out.append(lab.replace("QiSens video", f"QiSens video {n}", 1) if lab.startswith("QiSens video")
                       else f"{lab} ({n})")
        else:
            out.append(lab)
    return out


def clock(s: float | None) -> str:
    """Seconds as m:ss, or h:mm:ss from one hour."""
    if s is None:
        return ""
    s = int(round(s))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def span(s: float | None) -> str:
    """Duration in words: '45 s', '3 min 12 s', '1 h 05 min'."""
    if s is None:
        return ""
    s = int(round(s))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def utc_date(text: str | None) -> str:
    try:
        t = datetime.strptime(text or "", "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return ""
    return t.strftime("%d %b %Y, %H:%M")


def utc_short(text: str | None) -> str:
    """'2026-10-26T12:38:31Z' -> '26 Oct, 12:38'."""
    try:
        t = datetime.strptime(text or "", "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return ""
    return f"{t.day} {t:%b}, {t:%H:%M}"


def run_option(label: str, created_utc: str | None, variant: str | None, status: str = "done") -> str:
    """Text of one run in the Results picker: 'process_480p30.mp4 · 26 Oct, 12:38 · m'."""
    bits = [label, utc_short(created_utc), variant or ""]
    if status in ("queued", "running"):
        bits.append(STATUS_WORDS[status].lower())
    return " · ".join(b for b in bits if b)


def listed_runs(runs: list[dict]) -> list[dict]:
    """Runs the page lists: those with steps (steps.csv), and those still queued
    or running. Early runs from before Engine A have no steps and are left out
    of the list only; their folders stay as they are."""
    return [r for r in runs if r.get("has_steps") or r.get("status") in ("queued", "running")]


def utc_seconds(a: str | None, b: str | None) -> float | None:
    try:
        ta = datetime.strptime(a or "", "%Y-%m-%dT%H:%M:%SZ")
        tb = datetime.strptime(b or "", "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return (tb - ta).total_seconds()


# --- notices -----------------------------------------------------------------------------


def _notice_title(text: str) -> str:
    t = _SHOT_REF.sub("", text or "").strip()
    t = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0].rstrip(".")
    return t[:1].lower() + t[1:] if t else ""


def group_notices(notices: list[dict]) -> list[dict]:
    """Notices grouped by code in order of first appearance:
    [{code, title, count, shots: ['s001', ...], texts: [...]}]."""
    groups: dict[str, dict] = {}
    for n in notices:
        code = n.get("code") or "notice"
        g = groups.setdefault(code, {"code": code, "titles": [], "count": 0, "shots": [], "texts": []})
        text = n.get("text", "")
        g["count"] += 1
        g["texts"].append(text)
        g["titles"].append(_notice_title(text))
        m = _SHOT_ID.search(text)
        if m and m.group(1) not in g["shots"]:
            g["shots"].append(m.group(1))
    out = []
    for g in groups.values():
        titles = set(g.pop("titles"))
        title = titles.pop() if len(titles) == 1 else NOTICE_TITLES.get(g["code"]) or _notice_title(g["texts"][0])
        out.append({**g, "title": title})
    return out


def notice_line(g: dict, show: int = 3) -> str:
    """'AV-POS-004, no worker was found in the first 5 s, 8 shots: s001, s005, s007...'"""
    head = f"{g['code']}, {g['title']}" if g["title"] else g["code"]
    if g["shots"] and len(g["shots"]) == g["count"]:
        more = "..." if len(g["shots"]) > show else ""
        noun = "shot" if g["count"] == 1 else "shots"
        return f"{head}, {g['count']} {noun}: {', '.join(g['shots'][:show])}{more}"
    return f"{head}, {g['count']} {'notice' if g['count'] == 1 else 'notices'}"


def notices_html(notices: list[dict]) -> str:
    if not notices:
        return note("No notices for this run.")
    out = ["<div class='av-notices'>"]
    for g in group_notices(notices):
        items = "".join(f"<li>{esc(_SHOT_ID.sub(lambda m: m.group(1), t))}</li>" for t in g["texts"])
        out.append(f"<details class='av-notice'><summary><span class='av-pill av-mono'>{esc(g['code'])}</span>"
                   f"<span class='av-notice-text'>{esc(notice_line(g)[len(g['code']) + 2:])}</span>"
                   f"<span class='av-count'>{g['count']}</span></summary><ul>{items}</ul></details>")
    out.append("</div>")
    return "".join(out)


# --- building blocks --------------------------------------------------------------------


def pill(text: str, kind: str = "", dot: bool = False, mono: bool = False) -> str:
    cls = "av-pill" + (f" av-pill-{kind}" if kind else "") + (" av-mono" if mono else "")
    return f"<span class='{cls}'>{'<i></i>' if dot else ''}{esc(text)}</span>"


def status_pill(status: str) -> str:
    kind = {"done": "", "running": "live", "queued": "muted", "pending": "muted", "skipped": "muted",
            "failed": "strong", "partial": "outline", "lost": "outline"}.get(status, "")
    return pill(STATUS_WORDS.get(status, status.capitalize()), kind, dot=status == "running")


def flag_pills(flags: str) -> str:
    return " ".join(pill(f.replace("_", " "), "muted") for f in (flags or "").split(",") if f.strip())


def page_header(title: str, subtitle: str) -> str:
    return f"<div class='av-page-head'><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>"


def note(text: str) -> str:
    return f"<div class='av-note'>{esc(text)}</div>"


def banner(text: str) -> str:
    """Grey full width line at the top of a page (public test clips)."""
    return f"<div class='av-note av-banner'>{esc(text)}</div>"


def section_title(title: str, sub: str = "") -> str:
    return (f"<div class='av-card-head'><h2>{esc(title)}</h2>"
            + (f"<p>{esc(sub)}</p>" if sub else "") + "</div>")


def kpi_cards(items: list[tuple[str, str] | tuple[str, str, str]]) -> str:
    """Row of KPI cards: (label, value) or (label, value, small note under the value)."""
    out = ["<div class='av-kpis'>"]
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else ""
        out.append(f"<div class='av-kpi'><div class='av-kpi-label'>{esc(label)}</div>"
                   f"<div class='av-kpi-value'>{esc(value)}</div>"
                   + (f"<div class='av-kpi-sub'>{esc(sub)}</div>" if sub else "") + "</div>")
    out.append("</div>")
    return "".join(out)


def table(head: list[str], rows: list[list], num: set[int] | None = None, raw: set[int] | None = None,
          empty: str = "Nothing to show.", scroll: bool = False) -> str:
    """Table with thin row dividers. Columns in num are right aligned with
    tabular figures; cells of columns in raw are HTML already. A scroll table
    keeps its height (about 460 px) and scrolls under a sticky header row."""
    if not rows:
        return note(empty)
    num, raw = num or set(), raw or set()
    th = "".join(f"<th class='{'av-num' if i in num else ''}'>{esc(h)}</th>" for i, h in enumerate(head))
    body = []
    for r in rows:
        tds = "".join(f"<td class='{'av-num' if i in num else ''}'>{c if i in raw else esc(c)}</td>"
                      for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    wrap = "av-table-wrap av-scroll" if scroll else "av-table-wrap"
    return f"<div class='{wrap}'><table class='av-table'><thead><tr>{th}</tr></thead>" \
           f"<tbody>{''.join(body)}</tbody></table></div>"


def progress_bar(frac: float) -> str:
    pct = max(0.0, min(1.0, frac)) * 100
    return f"<div class='av-progress'><div style='width:{pct:.1f}%'></div></div>"


def img_html(rgb: np.ndarray | None, alt: str) -> str:
    """A chart (RGB array) as an inline PNG, so it is never written to a cache file."""
    if rgb is None:
        return ""
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    return (f"<img class='av-chart' alt='{esc(alt)}' "
            f"src='data:image/png;base64,{base64.b64encode(buf.tobytes()).decode()}'>")


# --- trunk band colours ------------------------------------------------------------------


def band_color(band: int) -> str:
    """Traffic light colour of trunk band B0 to B3 (green, yellow, orange, red)."""
    return BAND_COLORS[int(band)]


def band_dot(band: int) -> str:
    return f"<span class='av-band-dot' style='background:{band_color(band)}'></span>"


def metric_band(key: str) -> int | None:
    """Band of a bend_bN_s metric key, else None."""
    m = re.fullmatch(r"bend_b([0-3])_s", key)
    return int(m.group(1)) if m else None


# --- GPU state for the sidebar ------------------------------------------------------------


def gpu_busy(lock_dir: Path, gpus=(0, 1), proc_locks: str | None = None) -> dict[int, bool]:
    """Whether .locks/gpu<i>.lock is held by some process, read from /proc/locks
    (no lock is taken, so a starting job is never disturbed)."""
    if proc_locks is None:
        try:
            proc_locks = Path("/proc/locks").read_text()
        except OSError:
            proc_locks = ""
    held = set()
    for line in proc_locks.splitlines():
        parts = line.split()
        dev = next((p for p in parts if p.count(":") == 2), None)
        if dev:
            mj, mn, ino = dev.split(":")
            try:
                held.add((int(mj, 16), int(mn, 16), int(ino)))
            except ValueError:
                continue
    out = {}
    for g in gpus:
        f = Path(lock_dir) / f"gpu{g}.lock"
        try:
            st = f.stat()
        except OSError:
            out[g] = False
            continue
        out[g] = (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino) in held
    return out


def server_dot(busy: dict[int, bool]) -> str:
    """'free' (green dot) when no GPU lock is held, else 'busy' (amber)."""
    return "busy" if any(busy.values()) else "free"


def server_pill(busy: dict[int, bool], live_on: bool = False) -> str:
    """Sidebar server pill; it also carries the live flag that the Live nav dot
    reads through CSS (#av-shell:has(.av-live-flag.on))."""
    parts = ", ".join(f"GPU{g} {'busy' if b else 'free'}" for g, b in sorted(busy.items()))
    return (f"<div class='av-server'><span class='av-dot av-dot-{server_dot(busy)}'></span>"
            f"<span>QiSens server, {esc(parts)}</span>"
            f"<span class='av-live-flag{' on' if live_on else ''}' hidden></span></div>")
