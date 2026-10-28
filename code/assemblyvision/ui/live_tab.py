"""Live page (LLD 3.12, HLD Figure 6): one combined live view, no station
switch. Every frame gets the trunk band timers, time bent, idle time, both
hands and lifts (only while the knees are seen); cycle counting with the
workbench zones is switched on in its own section. Controls in one row: Start
live, Stop, Save recording, Model for analysis, Analyse recording and the
planned Improvement suggestions. The overlay is the one big picture; the
webcam input stays as a small Camera source tile, since Gradio needs a visible
webcam component to capture frames. Start live also starts the webcam stream
and Stop ends it (app.js, window.avCam); Clear empties the page again.

No flicker: the panel and log are pushed only when their HTML changed
(panel_update), at most every PANEL_S, and every periodic event hides
Gradio's progress animation.

The camera is the browser webcam through Gradio 5 streaming: the webcam
component lists every camera the browser sees (a USB camera or an iPhone by
Continuity Camera) and sends about 10 JPEG frames per second through the SSH
forwarded page. The stream handler only pushes the frame to LiveManager and
returns the latest overlay at once. The overlay goes back as an in memory
JPEG data URL (live_image), so no camera frame is written to disk unless Save
recording is on (NFR-SEC-07). A timer refreshes the live panel from the
session every 0.5 s, so a reloaded page shows the running session at once.
Styling is in style.css (classes lv-* and av-*).
"""
from __future__ import annotations

import base64
import types

import cv2
import numpy as np

from ..config import VARIANTS
from ..errors import AvError
from ..live.stations import COMBINED
from ..live.templates import list_templates, load_template
from . import views as v

PANEL_S = 0.5
BAND_NAMES = ["B0 under 20°", "B1 20 to 45°", "B2 45 to 60°", "B3 60° or more"]
NO_SIDE_VIEW = "Lifts need a side view with the whole body."
CAMERA_TIPS = [("Workbench", "camera in front", "cycles and hand use."),
               ("Floor pick", "camera at the side, whole body in view", "stoop versus squat lifts."),
               ("Side work", "camera at the side", "time spent bent.")]


def live_image(gr, **kwargs):
    """gr.Image whose output is sent as an in memory JPEG data URL, not a cached
    file. The method is bound to this one instance: a gr.Image subclass would
    make Gradio write a .pyi copy of this module into the source folder."""
    from gradio.data_classes import Base64ImageData

    def postprocess(self, value):
        if value is None:
            return None
        img = np.asarray(value)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return Base64ImageData(url="data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode())
    img = gr.Image(**kwargs)
    img.postprocess = types.MethodType(postprocess, img)
    return img


def station_choices() -> list[tuple[str, str]]:
    out = []
    for name in list_templates():
        try:
            out.append((load_template(name).title, name))
        except AvError:
            continue
    return sorted(out)




def station_label(title: str) -> str:
    """'A. Workbench' -> 'A Workbench' for the station control."""
    return title.replace(". ", " ", 1)


def station_info(name: str | None) -> str:
    try:
        t = load_template(name)
    except AvError as e:
        return v.note(f"{e.code}: {e.text}")
    return (f"<div class='lv-info'><p class='av-hint' style='margin:0;color:#3f3f3f;font-size:14px'>"
            f"{v.esc(t.description)}</p><p class='av-hint'><b style='color:#111;font-weight:600'>Camera</b> "
            f"&nbsp;{v.esc(t.camera)}</p></div>")


def camera_tips() -> str:
    rows = "".join(f"<div class='lv-tip'><b>{a}</b><span>{b}: {c}</span></div>" for a, b, c in CAMERA_TIPS)
    return "<div class='av-card'>" + v.section_title("Camera tips") + f"<div class='lv-tips'>{rows}</div></div>"


def band_tiles(st: dict, running: bool) -> str:
    """Seconds per trunk band, each tile with a 4 px top bar in its band colour;
    the band of the trunk now has a dark outline."""
    cells = "".join(f"<div class='av-kpi lv-band{' on' if st['band'] == i and running else ''}' "
                    f"style='--band:{v.band_color(i)}'><div class='av-kpi-label'>{b}</div>"
                    f"<div class='av-kpi-value'>{st['band_s'][i]:.0f} s</div></div>"
                    for i, b in enumerate(BAND_NAMES))
    trunk = "n/a" if st["trunk_deg"] is None else f"{st['trunk_deg']}°"
    return (f"<p class='lv-sub'>Trunk angle now <b style='color:#111'>{trunk}</b>. Time per band:</p>"
            f"<div class='av-kpis lv-bands'>{cells}</div>")


def _combined(st: dict, running: bool) -> list[str]:
    ps = st["posture"]
    lifts = st.get("lifts", [])
    n_st = sum(1 for x in lifts if x["kind"] == "stoop")
    out = []
    if running:
        out.append(_big("LIFTING") if st.get("in_lift") else
                   _big("BENT", f"{ps['current_s']:.1f} s") if ps["current_s"] > 0 else _big("UPRIGHT"))
    tiles = [("Time bent", _v(ps["total_s"], "s"), "B1 or worse"),
             ("Longest bent stretch", _v(ps["longest_s"], "s")), ("Idle time", _v(st.get("idle_time_s"), "s")),
             ("Both hands together", _pct(st.get("two_hand_simultaneous"))),
             ("Hand use ratio", _pct(st.get("two_hand_use_ratio")))]
    if st.get("lifts_ready") or lifts:
        tiles.append(("Lifts", str(len(lifts)), f"stoop {n_st}, squat {len(lifts) - n_st}"))
        if running:
            tiles.append(("Knee now", _v(st.get("knee_deg"), "°", "{:.0f}"), "180° = straight leg"))
    else:
        tiles.append(("Lifts", "n/a", "needs a side view"))
    if st.get("cycles_on"):
        tiles += [("Cycles", str(st.get("cycles", 0))), ("Last cycle", _v(st.get("last_cycle_s"), "s")),
                  ("Mean cycle", _v(st.get("mean_cycle_s"), "s"))]
    out.append(v.kpi_cards(tiles))
    if st.get("cycles_on") and running and st.get("step"):
        out.append(f"<p class='lv-sub'>Cycle step now: <b style='color:#111'>{v.esc(st['step'])}</b>"
                   + ("" if st.get("step_s") is None else f", {st['step_s']:.1f} s") + "</p>")
    if not st.get("lifts_ready") and running:
        out.append(f"<p class='lv-sub'>{NO_SIDE_VIEW}</p>")
    if lifts:
        x = lifts[-1]
        out.append(v.kpi_cards([(f"Last lift ({x['n']})", x["kind"].upper()),
                                ("Peak trunk", _v(x["peak_trunk_deg"], "°", "{:.0f}")),
                                ("Knee", _v(x["knee_deg"], "°", "{:.0f}")),
                                ("B2 + B3", _v(x["b2_s"] + x["b3_s"], "s"))]))
    if st.get("comparison"):
        out.append(comparison_card(st["comparison"]))
    elif lifts:
        out.append(f"<p class='lv-sub'>The stoop and squat comparison appears after one lift of each type "
                   f"(now {n_st} stoop, {len(lifts) - n_st} squat).</p>")
    return out


def _v(x, unit="", fmt="{:.1f}") -> str:
    return "n/a" if x is None else (fmt.format(x) + (f" {unit}" if unit else ""))


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f} %"


def comparison_card(cmp_: dict) -> str:
    rows = [("Mean peak trunk", "peak_trunk_deg", "°", "{:.0f}"), ("Knee at lowest point", "knee_deg", "°", "{:.0f}"),
            ("Time in B2 + B3", "b2_b3_s", " s", "{:.1f}"), ("Lifts", "n", "", "{:d}")]
    out = ["<div class='lv-card'><div class='h'></div><div class='h'>STOOP</div><div class='h'>SQUAT</div>"]
    for label, key, unit, fmt in rows:
        out.append(f"<div class='lbl'>{label}</div>")
        for kind in ("stoop", "squat"):
            x = cmp_[kind][key]
            out.append(f"<div class='n'>{'n/a' if x is None else fmt.format(x) + unit}</div>")
    out.append("</div>")
    return "".join(out)


def status_pills(p: dict) -> str:
    """Session, elapsed, rate, fps, delay, frames and dropped as one row of pills;
    the state pill always matches the session status. data-live-rate tells the
    page how many frames per second to send (app.js)."""
    running = p["status"] == "running"
    pills = [v.pill(p["session_id"], mono=True), v.pill(f"elapsed {v.clock(p['elapsed_s'])}")]
    if running:
        pills.append(v.pill("camera paused", "outline") if p["paused"] else v.pill("live", "live", dot=True))
        if p.get("rate_fps"):
            pills.append(f"<span class='av-pill av-pill-outline' data-live-rate='{int(p['rate_fps'])}'>"
                         f"rate {int(p['rate_fps'])} fps</span>")
        pills.append(v.pill(_v(p["fps"], "fps")))
    else:
        pills.append(v.pill("stopped", "strong"))
    pills += [v.pill(f"delay p50 / p95 {_v(p['delay_p50_ms'], fmt='{:.0f}')} / {_v(p['delay_p95_ms'], 'ms', '{:.0f}')}"),
              v.pill(f"frames {p['frames_in']}"), v.pill(f"dropped {p['dropped']}")]
    if p["recording"]:
        pills.append(v.pill("REC", "strong", dot=running))
    return "<div class='lv-status av-pills'>" + "".join(pills) + "</div>"


def _big(word: str, small: str = "") -> str:
    return f"<div class='lv-big'>{v.esc(word)}" + (f"<small>{v.esc(small)}</small>" if small else "") + "</div>"


def panel_html(p: dict | None, port: int = 7860) -> str:
    if p is None:
        return ("<div class='av-card'>" + v.section_title("Live panel")
                + v.note("No live session. Press Start live, then press the record button under the Camera "
                         "source picture.") + "</div>")
    st = p["station"]
    running = p["status"] == "running"
    body = [f"<div class='av-card' data-live-status='{v.esc(p['status'])}'>", status_pills(p), f"<div class='lv-title'>{v.esc(st['title'])}</div>"]
    if not running:
        body.append(_big("STOPPED", f"after {v.span(p['elapsed_s'])}"))
    if st["kind"] == COMBINED:
        body += _combined(st, running)
    elif st["kind"] == "workbench":
        if running:
            body.append(_big((st.get("step") or "waiting").upper(),
                             "" if st.get("step_s") is None else f"{st['step_s']:.1f} s"))
        body.append(v.kpi_cards([("Cycles", str(st.get("cycles", 0))),
                                 ("Last cycle", _v(st.get("last_cycle_s"), "s")),
                                 ("Mean cycle", _v(st.get("mean_cycle_s"), "s")),
                                 ("Both hands together", _pct(st.get("two_hand_simultaneous"))),
                                 ("Hand use ratio", _pct(st.get("two_hand_use_ratio")))]))
    elif st["kind"] == "lift":
        lifts = st.get("lifts", [])
        if running:
            body.append(_big("LIFTING" if st.get("in_lift") else f"{len(lifts)} lift{'s' if len(lifts) != 1 else ''}"))
        if lifts:
            x = lifts[-1]
            body.append(v.kpi_cards([(f"Lift {x['n']}", x["kind"].upper()),
                                     ("Peak trunk", _v(x["peak_trunk_deg"], "°", "{:.0f}")),
                                     ("Knee", _v(x["knee_deg"], "°", "{:.0f}")),
                                     ("B2", _v(x["b2_s"], "s")), ("B3", _v(x["b3_s"], "s"))]))
        if st.get("comparison"):
            body.append(comparison_card(st["comparison"]))
        else:
            n_st = sum(1 for x in lifts if x["kind"] == "stoop")
            body.append(f"<p class='lv-sub'>The stoop and squat comparison appears after one lift of each type "
                        f"(now {n_st} stoop, {len(lifts) - n_st} squat).</p>")
    else:
        ps = st["posture"]
        g = st.get("gaze", {})
        if running:
            body.append(_big("BENT", f"{ps['current_s']:.1f} s") if ps["current_s"] > 0 else _big("UPRIGHT"))
        body.append(v.kpi_cards([("Longest bent stretch", _v(ps["longest_s"], "s")),
                                 ("Total bent", _v(ps["total_s"], "s")), ("Idle time", _v(st.get("idle_time_s"), "s")),
                                 ("Gaze front", _pct(g.get("front"))), ("Gaze side", _pct(g.get("side"))),
                                 ("Gaze away", _pct(g.get("away")))])
                    + "<p class='lv-sub'>Gaze is rough and relative to the camera.</p>")
    body.append(band_tiles(st, running))
    if p["notices"]:
        body.append("<div class='lv-notes av-note-list'>"
                    + "".join(v.note(n) for n in p["notices"][-4:]) + "</div>")
    body.append("</div>")
    return "".join(body)


def _lift_rows(log: list[dict]) -> list[list]:
    return [[x["n"], v.pill(x["kind"]), f"{x['peak_trunk_deg']:.0f}°", _v(x["knee_deg"], "°", "{:.0f}"),
             f"{x['b2_s']:.1f} s", f"{x['b3_s']:.1f} s", f"{x['duration_s']:.1f} s"] for x in log]


def log_html(p: dict | None) -> str:
    if not p:
        return ""
    st = p["station"]
    if st["kind"] == COMBINED:
        out = []
        if p["log"]:
            out.append(v.section_title("Lifts") + _table(["Lift", "Type", "Peak trunk", "Knee", "B2", "B3",
                                                          "Duration"], _lift_rows(p["log"]), {0, 2, 3, 4, 5, 6}))
        steps = p.get("steps") or []
        if steps:
            rows = [[v.esc(x["step"]), f"{x['start_s']:.1f} s", f"{x['duration_s']:.1f} s",
                     v.pill(x["closed_by"], "muted")] for x in steps[-12:]]
            out.append(v.section_title("Cycle steps, the last 12") + _table(["Step", "Start", "Duration",
                                                                             "Closed by"], rows, {1, 2}))
        return "<div class='av-card'>" + "".join(out) + "</div>" if out else ""
    if not p["log"]:
        return ""
    if st["kind"] == "lift":
        head = ["Lift", "Type", "Peak trunk", "Knee", "B2", "B3", "Duration"]
        rows = _lift_rows(p["log"])
        cap, num = "Lifts", {0, 2, 3, 4, 5, 6}
    else:
        head = ["Step", "Start", "Duration", "Closed by"]
        rows = [[v.esc(x["step"]), f"{x['start_s']:.1f} s", f"{x['duration_s']:.1f} s", v.pill(x["closed_by"], "muted")]
                for x in p["log"][-12:]]
        cap, num = "Steps, the last 12", {1, 2}
    return "<div class='av-card'>" + v.section_title(cap) + _table(head, rows, num) + "</div>"


def _table(head: list, rows: list, num: set | None = None) -> str:
    raw = set(range(len(head)))
    return v.table(head, [[c if isinstance(c, str) and c.startswith("<") else v.esc(c) for c in r] for r in rows],
                   num=num, raw=raw)


def summary_html(s: dict | None) -> str:
    if not s:
        return ""
    rec = "yes" if s["recording"] else "no"
    if s.get("recording_error"):
        rec += f" ({s['recording_error']})"
    out = ["<div class='av-card'>", v.section_title("Session summary", f"{s['session_id']}"),
           v.kpi_cards([("Duration", v.span(s["duration_s"])), ("Frames processed", str(s["processed"])),
                        ("Mean fps", _v(s["fps_mean"])),
                        ("Delay p50 / p95", f"{_v(s['delay_p50_ms'], fmt='{:.0f}')} / "
                                            f"{_v(s['delay_p95_ms'], 'ms', '{:.0f}')}"),
                        ("Recording", rec)])]
    for st in s["stations"]:
        title = "Measures" if st.get("kind") == COMBINED else f"Station {st['number']}: {v.esc(st['title'])}"
        out.append(f"<h3 class='lv-title' style='margin-top:22px'>{title}</h3>")
        out.append(_table(["Measure", "Value"], [[k, val] for k, val in st["rows"]], {1}))
    out.append("</div>")
    return "".join(out)


def panel_update(p: dict | None, shown: dict, port: int = 7860) -> tuple[str | None, str | None, dict]:
    """New panel and log HTML for the page, None for each one that equals what
    the page shows already (shown, kept per browser in a gr.State), and the new
    shown dict. The page then gets gr.skip() and nothing is redrawn."""
    panel, log = panel_html(p, port), log_html(p)
    new = {"panel": panel, "log": log}
    return (None if shown.get("panel") == panel else panel, None if shown.get("log") == log else log, new)


def save_label(on: bool) -> str:
    return f"Save recording: {'on' if on else 'off'}"


def clear_values(live) -> dict:
    """What Clear puts on the page (no session running): empty overlay, the
    'No live session' panel, no log or summary, the camera tile reset."""
    if not live.clear():
        return {}
    return {"overlay": None, "panel": panel_html(None), "log": "", "summary": "", "camera": None}


def build_live_tab(app, gr, job, job_timer, server=None):
    """The Live page inside the Blocks of GradioApp.build(). job and job_timer
    belong to New analysis: Analyse recording hands its batch run over there.
    server is the sidebar pill, refreshed at Start and Stop for the Live dot."""
    live = app.live
    live_running = live.session is not None and live.session.status == "running"
    gr.HTML(v.page_header("Live", "Pose analysis of a camera on your computer, about 10 frames per second. "
                          "The pose model runs on GPU1."))
    with gr.Row(elem_classes=["lv-top"], equal_height=False):
        with gr.Column(elem_classes=["av-card", "lv-controls"], scale=1):
            gr.HTML(v.section_title("Controls", "Start live, then press the record button under the Camera source "
                                    "picture. Click a person on the overlay to follow them."))
            with gr.Row(elem_classes=["av-actions", "lv-actions"]):
                start = gr.Button("Start live", variant="primary", elem_classes=["av-fit"])
                stop = gr.Button("Stop", variant="secondary", elem_classes=["av-fit"])
                save = gr.Checkbox(False, label=save_label(False), container=False,
                                   elem_classes=["av-switch", "av-fit"])
                clear = gr.Button("Clear", variant="secondary", elem_id="lv-clear", elem_classes=["av-fit"],
                                  interactive=not live_running)
                gr.HTML("<div class='av-field-label lv-inline-label'>Model for analysis</div>",
                        elem_classes=["av-fit"])
                variant = gr.Radio(list(VARIANTS), value=app.cfg["pose"]["variant"], label="Model for analysis",
                                   show_label=False, container=False, elem_classes=["av-seg", "av-fit"])
                has_rec = bool(live.last_summary and live.last_summary.get("recording"))
                analyse = gr.Button("Analyse recording", variant="secondary", interactive=has_rec,
                                    elem_classes=["av-fit"])
                gr.Button("Improvement suggestions", variant="secondary", interactive=False,
                          elem_classes=["av-fit"])
            gr.HTML("<div class='av-hint' style='margin-top:0'>Analyse recording needs a session stopped with "
                    "Save recording on. Improvement suggestions: local VLM, planned Week 11.</div>")
            msg = gr.HTML()
        with gr.Column(elem_classes=["av-card", "lv-cam"], scale=0, min_width=240):
            gr.HTML(v.section_title("Camera source"))
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera source",
                           show_label=False, elem_id="lv-camera", webcam_options=gr.WebcamOptions(mirror=False, constraints={
                               "video": {"width": {"ideal": 640}, "height": {"ideal": 480}}}))
    with gr.Column(elem_classes=["av-card", "lv-overlay"]):
        gr.HTML(v.section_title("Live overlay", "Click a person to follow them."))
        overlay = live_image(gr, label="Live overlay", show_label=False, interactive=False, type="numpy",
                             show_download_button=False, show_fullscreen_button=True)
    panel = gr.HTML(panel_html(None))
    with gr.Row(equal_height=False, elem_classes=["lv-extra"]):
        with gr.Column(scale=1):
            with gr.Accordion("Cycle counting (camera in front of a workbench)", open=False,
                              elem_classes=["av-accordion"]):
                cycles = gr.Checkbox(False, label="Count cycles", container=False, elem_classes=["av-switch"])
                gr.HTML("<div class='av-hint'>Pick on the left, assemble in the centre, place on the right. "
                        "Each place ends one cycle. Move the two borders to the edges of the zones.</div>")
                b1 = gr.Slider(0.1, 0.9, value=1 / 3, step=0.01, label="Border left | centre (share of image width)")
                b2 = gr.Slider(0.1, 0.9, value=2 / 3, step=0.01, label="Border centre | right")
        with gr.Column(scale=1):
            gr.HTML(camera_tips())
    log = gr.HTML()
    summary = gr.HTML()
    timer = gr.Timer(PANEL_S, active=False)
    shown = gr.State({})                       # panel and log HTML the page shows now
    srv = [server] if server is not None else []

    def srv_val():
        return [app.server_html()] if server is not None else []

    def note(text: str) -> str:
        return f"<span class='av-hint'>{v.esc(text)}</span>"

    def on_cycles(on, x1, x2):
        s = live.session
        if s is not None and s.status == "running" and bool(on) != s.cycles:
            s.set_cycles(bool(on), (x1, x2))
        return panel_html(live.panel(), app.port)

    def on_borders(x1, x2):
        s = live.session
        if s is not None and s.status == "running":
            s.set_borders(x1, x2)

    def on_start(rec, cyc, x1, x2):
        try:
            sid = live.start(COMBINED, rec, (x1, x2), bool(cyc))
        except AvError as e:
            return v.note(f"{e.code}: {e.text}"), gr.Timer(active=False), panel_html(live.panel()), gr.skip(), \
                gr.skip(), gr.skip(), {}, *[gr.skip()] * len(srv)
        rec_txt = " Recording to disk." if rec else ""
        return (note(f"Session {sid} started.{rec_txt} The camera starts sending frames (allow camera access if "
                     "the browser asks)."), gr.Timer(active=True), panel_html(live.panel()), "",
                gr.Button(interactive=False), gr.Button(interactive=False), {}, *srv_val())

    def on_stop():
        s = live.stop()
        p = live.panel()
        last = live.stopped_overlay()
        return (note("Stopped." if s else "No session was running."), gr.Timer(active=False), panel_html(p),
                log_html(p), summary_html(s), gr.Button(interactive=bool(s and s.get("recording"))),
                gr.Button(interactive=True), gr.skip() if last is None else last, {}, *srv_val())

    def on_tick(seen):
        p = live.panel()
        running = p is not None and p["status"] == "running"
        panel_, log_, seen = panel_update(p, seen or {}, app.port)
        out = [gr.skip() if panel_ is None else panel_, gr.skip() if log_ is None else log_, seen]
        if not running:                       # stopped by itself (idle or max time)
            s = live.last_summary
            last = live.stopped_overlay()
            out += [summary_html(s), gr.Timer(active=False), gr.Button(interactive=bool(s and s.get("recording"))),
                    gr.Button(interactive=True), gr.skip() if last is None else last, *srv_val()]
        else:
            out += [gr.skip()] * (5 + len(srv))
        return out

    def on_save(on):
        s = live.session
        txt = note("Save recording applies from the next session.") \
            if s is not None and s.status == "running" else gr.skip()
        return gr.Checkbox(label=save_label(bool(on))), txt

    def on_clear():
        vals = clear_values(live)
        if not vals:
            return [note("Stop the session first.")] + [gr.skip()] * 6
        return ["", vals["overlay"], vals["panel"], vals["log"], vals["summary"], vals["camera"], {}]

    def on_frame(frame):
        img = live.push(frame)
        return gr.skip() if img is None else img

    def on_click(evt):
        s = live.session
        if s is None or s.status != "running":
            return gr.skip()
        try:
            s.lock_subject(*evt.index[:2])
            return note("Following the selected person.")
        except AvError as e:
            return v.note(f"{e.code}: {e.text}")

    on_click.__annotations__["evt"] = gr.SelectData     # a real type, so Gradio passes the click

    def on_analyse(var):
        s = live.last_summary
        if not s:
            return v.note("Stop a session with Save recording on first."), gr.skip(), gr.skip()
        try:
            run_id = live.to_batch(s["session_id"], var, app.runs)
        except AvError as e:
            return v.note(f"{e.code}: {e.text}" + (f" ({e.detail})" if e.detail else "")), gr.skip(), gr.skip()
        return (note(f"Batch run {run_id} started on the recording (pose on GPU {app.cfg['gpu']['pose']}). "
                     "Its progress is on New analysis."), run_id, gr.Timer(active=True))

    def on_load():
        p = live.panel()
        running = p is not None and p["status"] == "running"
        s = live.last_summary
        cyc = bool(running and live.session.cycles)
        return panel_html(p, app.port), log_html(p), summary_html(None if running or live.cleared else s), \
            gr.Timer(active=running), gr.Button(interactive=bool(not running and s and s.get("recording"))), cyc, \
            gr.Button(interactive=not running), {}

    cycles.change(on_cycles, [cycles, b1, b2], panel, show_progress="hidden")
    b1.release(on_borders, [b1, b2], None)
    b2.release(on_borders, [b1, b2], None)
    start.click(on_start, [save, cycles, b1, b2], [msg, timer, panel, summary, analyse, clear, shown, *srv],
                js="(...a) => { window.avCam && window.avCam.access(); return a; }", show_progress="hidden") \
        .then(None, None, None, js="() => { window.avCam && window.avCam.recordIfLive(); }")
    stop.click(on_stop, None, [msg, timer, panel, log, summary, analyse, clear, overlay, shown, *srv],
               js="() => { window.avCam && window.avCam.stop(); }", show_progress="hidden")
    timer.tick(on_tick, shown, [panel, log, shown, summary, timer, analyse, clear, overlay, *srv],
               show_progress="hidden")
    save.change(on_save, save, [save, msg], show_progress="hidden")
    clear.click(on_clear, None, [msg, overlay, panel, log, summary, cam, shown],
                js="() => { window.avCam && window.avCam.stop(); }", show_progress="hidden")
    cam.stream(on_frame, cam, overlay, stream_every=app.cfg["live"]["stream_every_s"], time_limit=None,
               concurrency_limit=1, show_progress="hidden", trigger_mode="always_last")
    overlay.select(on_click, None, msg, show_progress="hidden")
    analyse.click(on_analyse, variant, [msg, job, job_timer])
    return on_load, [panel, log, summary, timer, analyse, cycles, clear, shown]
