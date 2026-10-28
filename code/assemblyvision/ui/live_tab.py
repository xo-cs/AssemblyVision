"""Live tab (LLD 3.12, HLD Figure 6): station, Save recording, Start live,
Stop, camera, overlay, live panel, log, summary and Analyse recording.

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
from ..live.templates import list_templates, load_template
from . import views as v

PANEL_S = 0.5


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
    """Session, elapsed, fps, delay, frames and dropped as one row of pills; the
    state pill always matches the session status."""
    running = p["status"] == "running"
    pills = [v.pill(p["session_id"], mono=True), v.pill(f"elapsed {v.clock(p['elapsed_s'])}")]
    if running:
        pills.append(v.pill("camera paused", "outline") if p["paused"] else v.pill("live", "live", dot=True))
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
                + v.note("No live session. Choose a station and press Start live, then press the record button "
                         "under the camera picture.") + "</div>")
    st = p["station"]
    running = p["status"] == "running"
    body = ["<div class='av-card'>", status_pills(p), f"<div class='lv-title'>{v.esc(st['title'])}</div>"]
    if not running:
        body.append(_big("STOPPED", f"after {v.span(p['elapsed_s'])}"))
    if st["kind"] == "workbench":
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
    trunk = "n/a" if st["trunk_deg"] is None else f"{st['trunk_deg']}°"
    bands = ["B0 under 20°", "B1 20 to 45°", "B2 45 to 60°", "B3 60° or more"]
    cells = "".join(f"<div class='av-kpi{' on' if st['band'] == i and running else ''}'><div class='av-kpi-label'>"
                    f"{b}</div><div class='av-kpi-value'>{st['band_s'][i]:.0f} s</div></div>"
                    for i, b in enumerate(bands))
    body.append(f"<p class='lv-sub'>Trunk angle now <b style='color:#111'>{trunk}</b>. Time per band:</p>"
                f"<div class='av-kpis lv-bands'>{cells}</div>")
    if p["notices"]:
        body.append("<div class='lv-notes av-note-list'>"
                    + "".join(v.note(n) for n in p["notices"][-4:]) + "</div>")
    body.append("</div>")
    return "".join(body)


def log_html(p: dict | None) -> str:
    if not p or not p["log"]:
        return ""
    st = p["station"]
    if st["kind"] == "lift":
        head = ["Lift", "Type", "Peak trunk", "Knee", "B2", "B3", "Duration"]
        rows = [[x["n"], v.pill(x["kind"]), f"{x['peak_trunk_deg']:.0f}°", _v(x["knee_deg"], "°", "{:.0f}"),
                 f"{x['b2_s']:.1f} s", f"{x['b3_s']:.1f} s", f"{x['duration_s']:.1f} s"] for x in p["log"]]
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
        out.append(f"<h3 class='lv-title' style='margin-top:22px'>Station {st['number']}: {v.esc(st['title'])}</h3>")
        out.append(_table(["Measure", "Value"], [[k, val] for k, val in st["rows"]], {1}))
    out.append("</div>")
    return "".join(out)


def build_live_tab(app, gr, job, job_timer):
    """The Live page inside the Blocks of GradioApp.build(). job and job_timer
    belong to New analysis: Analyse recording hands its batch run over there."""
    live = app.live
    choices = station_choices()
    default = app.cfg["live"]["default_template"]
    default = default if default in [c[1] for c in choices] else (choices[0][1] if choices else None)
    gr.HTML(v.page_header("Live", "Pose analysis of a camera on your computer, about 10 frames per second. "
                          "The pose model runs on GPU1."))
    with gr.Column(elem_classes=["av-card"]):
        gr.HTML(v.section_title("Station"))
        station = gr.Radio([(station_label(t), n) for t, n in choices], value=default, show_label=False,
                           container=False, elem_classes=["av-seg"])
        info = gr.HTML(station_info(default))
        with gr.Row(visible=default == "workbench") as zone_row:
            b1 = gr.Slider(0.1, 0.9, value=1 / 3, step=0.01, label="Border left | centre (share of image width)")
            b2 = gr.Slider(0.1, 0.9, value=2 / 3, step=0.01, label="Border centre | right")
    with gr.Row(elem_classes=["av-actions"]):
        start = gr.Button("Start live", variant="primary", elem_classes=["av-fit"])
        stop = gr.Button("Stop", variant="secondary", elem_classes=["av-fit"])
        save = gr.Checkbox(False, label="Save recording", container=False, elem_classes=["av-switch", "av-fit"])
        gr.HTML("<div style='flex:1'></div>")
        variant = gr.Radio(list(VARIANTS), value=app.cfg["pose"]["variant"], label="Model for analysis",
                           show_label=False, container=False, elem_classes=["av-seg", "av-fit"])
        has_rec = bool(live.last_summary and live.last_summary.get("recording"))
        analyse = gr.Button("Analyse recording", variant="secondary", interactive=has_rec, elem_classes=["av-fit"])
    msg = gr.HTML()
    with gr.Row(elem_classes=["lv-media"], equal_height=True):
        with gr.Column(elem_classes=["av-card"]):
            gr.HTML(v.section_title("Camera", "Choose the camera under the picture, then press record."))
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera", show_label=False,
                           webcam_options=gr.WebcamOptions(mirror=False, constraints={
                               "video": {"width": {"ideal": 640}, "height": {"ideal": 480}}}))
        with gr.Column(elem_classes=["av-card"]):
            gr.HTML(v.section_title("Live overlay", "Click a person to follow them."))
            overlay = live_image(gr, label="Live overlay", show_label=False, interactive=False, type="numpy",
                                 show_download_button=False, show_fullscreen_button=True)
    panel = gr.HTML(panel_html(None))
    log = gr.HTML()
    summary = gr.HTML()
    timer = gr.Timer(PANEL_S, active=False)

    def note(text: str) -> str:
        return f"<span class='av-hint'>{v.esc(text)}</span>"

    def on_station(name, x1, x2):
        s = live.session
        if s is not None and s.status == "running":
            s.switch_station(name, (x1, x2))
        return station_info(name), gr.Row(visible=name == "workbench")

    def on_borders(x1, x2):
        s = live.session
        if s is not None and s.status == "running":
            s.set_borders(x1, x2)

    def on_start(name, rec, x1, x2):
        try:
            sid = live.start(name, rec, (x1, x2))
        except AvError as e:
            return v.note(f"{e.code}: {e.text}"), gr.Timer(active=False), panel_html(live.panel()), gr.skip(), \
                gr.skip()
        return (note(f"Session {sid} started. Press the record button under the camera picture to send frames."),
                gr.Timer(active=True), panel_html(live.panel()), "", gr.Button(interactive=False))

    def on_stop():
        s = live.stop()
        p = live.panel()
        return (note("Stopped." if s else "No session was running."), gr.Timer(active=False), panel_html(p),
                log_html(p), summary_html(s), gr.Button(interactive=bool(s and s.get("recording"))))

    def on_tick():
        p = live.panel()
        running = p is not None and p["status"] == "running"
        out = [panel_html(p, app.port), log_html(p)]
        if not running:                       # stopped by itself (idle or max time)
            s = live.last_summary
            out += [summary_html(s), gr.Timer(active=False), gr.Button(interactive=bool(s and s.get("recording")))]
        else:
            out += [gr.skip(), gr.skip(), gr.skip()]
        return out

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
        return panel_html(p, app.port), log_html(p), summary_html(None if running else s), \
            gr.Timer(active=running), gr.Button(interactive=bool(not running and s and s.get("recording")))

    station.change(on_station, [station, b1, b2], [info, zone_row])
    b1.release(on_borders, [b1, b2], None)
    b2.release(on_borders, [b1, b2], None)
    start.click(on_start, [station, save, b1, b2], [msg, timer, panel, summary, analyse])
    stop.click(on_stop, None, [msg, timer, panel, log, summary, analyse])
    timer.tick(on_tick, None, [panel, log, summary, timer, analyse], show_progress="hidden")
    cam.stream(on_frame, cam, overlay, stream_every=app.cfg["live"]["stream_every_s"], time_limit=None,
               concurrency_limit=1, show_progress="hidden", trigger_mode="always_last")
    overlay.select(on_click, None, msg)
    analyse.click(on_analyse, variant, [msg, job, job_timer])
    return on_load, [panel, log, summary, timer, analyse]
