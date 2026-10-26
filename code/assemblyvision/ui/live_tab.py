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
"""
from __future__ import annotations

import base64
import html
import types

import cv2
import numpy as np

from ..config import VARIANTS
from ..errors import AvError
from ..live.templates import list_templates, load_template

PANEL_S = 0.5

LIVE_CSS = """
.lv-status { font-size: 1.05em; color: #333; margin-bottom: 6px; }
.lv-status b { color: #000; }
.lv-big { font-size: 2.6em; font-weight: bold; line-height: 1.1; margin: 4px 0 10px 0; }
.lv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin: 6px 0; }
.lv-cell { border: 2px solid #000; padding: 6px 10px; background: #fff; }
.lv-cell .k { font-size: 0.9em; color: #444; }
.lv-cell .v { font-size: 2.1em; font-weight: bold; }
.lv-bands { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin: 8px 0; }
.lv-bands div { border: 2px solid #000; padding: 4px; text-align: center; font-size: 1.1em; }
.lv-bands div.on { background: #000; color: #fff; font-weight: bold; }
.lv-card { display: grid; grid-template-columns: 1.1fr 1fr 1fr; border: 3px solid #000; margin: 10px 0; }
.lv-card div { padding: 8px 12px; border-bottom: 1px solid #999; font-size: 1.4em; }
.lv-card .h { background: #000; color: #fff; font-size: 2.2em; font-weight: bold; text-align: center; }
.lv-card .n { font-size: 2.6em; font-weight: bold; text-align: center; }
.lv-card .lbl { font-size: 1.2em; color: #333; display: flex; align-items: center; }
.lv-notes { border-left: 4px solid #000; padding: 2px 10px; color: #222; font-size: 0.95em; }
.lv-table { border-collapse: collapse; width: 100%; font-size: 1.1em; }
.lv-table th { background: #000; color: #fff; text-align: left; padding: 5px 8px; }
.lv-table td { border-bottom: 1px solid #999; padding: 5px 8px; }
.lv-table caption { text-align: left; font-weight: bold; font-size: 1.3em; padding: 8px 0 4px 0; }
"""


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


def station_info(name: str | None) -> str:
    try:
        t = load_template(name)
    except AvError as e:
        return f"{e.code}: {e.text}"
    return f"**{t.title}**: {t.description}\n\n**Camera:** {t.camera}"


def _v(x, unit="", fmt="{:.1f}") -> str:
    return "n/a" if x is None else (fmt.format(x) + (f" {unit}" if unit else ""))


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f} %"


def _cell(k: str, v: str) -> str:
    return f"<div class='lv-cell'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(v)}</div></div>"


def comparison_card(cmp_: dict) -> str:
    rows = [("Mean peak trunk", "peak_trunk_deg", "°", "{:.0f}"), ("Knee at lowest point", "knee_deg", "°", "{:.0f}"),
            ("Time in B2 + B3", "b2_b3_s", "s", "{:.1f}"), ("Lifts", "n", "", "{:d}")]
    out = ["<div class='lv-card'><div class='h'></div><div class='h'>STOOP</div><div class='h'>SQUAT</div>"]
    for label, key, unit, fmt in rows:
        out.append(f"<div class='lbl'>{label}</div>")
        for kind in ("stoop", "squat"):
            v = cmp_[kind][key]
            out.append(f"<div class='n'>{'n/a' if v is None else fmt.format(v) + unit}</div>")
    out.append("</div>")
    return "".join(out)


def panel_html(p: dict | None, port: int = 7860) -> str:
    if p is None:
        return ("<div class='lv-status'>No live session. Choose a station and press <b>Start live</b>, then press "
                "the record button under the camera.</div>")
    st = p["station"]
    head = (f"<div class='lv-status'><b>{html.escape(p['session_id'])}</b> &nbsp; {p['elapsed_s']} s &nbsp;|&nbsp; "
            f"<b>{_v(p['fps'], 'fps')}</b> &nbsp;|&nbsp; delay p50 {_v(p['delay_p50_ms'], 'ms', '{:.0f}')}, "
            f"p95 {_v(p['delay_p95_ms'], 'ms', '{:.0f}')} &nbsp;|&nbsp; frames {p['frames_in']}, "
            f"dropped {p['dropped']}" + (" &nbsp;|&nbsp; <b>REC</b>" if p["recording"] else "")
            + (" &nbsp;|&nbsp; <b>CAMERA PAUSED</b>" if p["paused"] else "")
            + (f" &nbsp;|&nbsp; <b>{p['status'].upper()}</b>" if p["status"] != "running" else "") + "</div>")
    body = [head, f"<div class='lv-status'>{html.escape(st['title'])}</div>"]
    if st["kind"] == "workbench":
        step = (st.get("step") or "waiting").upper()
        step_s = "" if st.get("step_s") is None else f" &nbsp;{st['step_s']:.1f} s"
        body.append(f"<div class='lv-big'>{html.escape(step)}{step_s}</div>")
        body.append("<div class='lv-grid'>" + _cell("Cycles", str(st.get("cycles", 0)))
                    + _cell("Last cycle", _v(st.get("last_cycle_s"), "s"))
                    + _cell("Mean cycle", _v(st.get("mean_cycle_s"), "s"))
                    + _cell("Both hands together", _pct(st.get("two_hand_simultaneous")))
                    + _cell("Hand use ratio", _pct(st.get("two_hand_use_ratio"))) + "</div>")
    elif st["kind"] == "lift":
        lifts = st.get("lifts", [])
        state = "LIFTING" if st.get("in_lift") else f"{len(lifts)} lift{'s' if len(lifts) != 1 else ''}"
        body.append(f"<div class='lv-big'>{state}</div>")
        if lifts:
            x = lifts[-1]
            body.append("<div class='lv-grid'>" + _cell(f"Lift {x['n']}", x["kind"].upper())
                        + _cell("Peak trunk", _v(x["peak_trunk_deg"], "°", "{:.0f}"))
                        + _cell("Knee", _v(x["knee_deg"], "°", "{:.0f}"))
                        + _cell("B2", _v(x["b2_s"], "s")) + _cell("B3", _v(x["b3_s"], "s")) + "</div>")
        if st.get("comparison"):
            body.append(comparison_card(st["comparison"]))
        else:
            n_st = sum(1 for x in lifts if x["kind"] == "stoop")
            n_sq = len(lifts) - n_st
            body.append(f"<div class='lv-status'>Comparison after one stoop and one squat lift "
                        f"(now {n_st} stoop, {n_sq} squat).</div>")
    else:
        ps = st["posture"]
        g = st.get("gaze", {})
        body.append(f"<div class='lv-big'>BENT {ps['current_s']:.1f} s</div>")
        body.append("<div class='lv-grid'>" + _cell("Longest bent stretch", _v(ps["longest_s"], "s"))
                    + _cell("Total bent", _v(ps["total_s"], "s")) + _cell("Idle time", _v(st.get("idle_time_s"), "s"))
                    + _cell("Gaze front", _pct(g.get("front"))) + _cell("Gaze side", _pct(g.get("side")))
                    + _cell("Gaze away", _pct(g.get("away"))) + "</div>"
                    + "<div class='lv-status'>Gaze is rough and relative to the camera.</div>")
    trunk = "n/a" if st["trunk_deg"] is None else f"{st['trunk_deg']}°"
    bands = ["B0 &lt;20°", "B1 20 to 45°", "B2 45 to 60°", "B3 ≥60°"]
    body.append(f"<div class='lv-status'>Trunk now: <b>{trunk}</b></div><div class='lv-bands'>"
                + "".join(f"<div class='{'on' if st['band'] == i else ''}'>{b}<br>{st['band_s'][i]:.0f} s</div>"
                          for i, b in enumerate(bands)) + "</div>")
    if p["notices"]:
        body.append("<div class='lv-notes'>" + "<br>".join(html.escape(n) for n in p["notices"][-4:]) + "</div>")
    return "".join(body)


def log_html(p: dict | None) -> str:
    if not p or not p["log"]:
        return ""
    st = p["station"]
    if st["kind"] == "lift":
        head = ["Lift", "Type", "Peak trunk", "Knee", "B2", "B3", "Duration"]
        rows = [[x["n"], x["kind"], f"{x['peak_trunk_deg']:.0f}°", _v(x["knee_deg"], "°", "{:.0f}"),
                 f"{x['b2_s']:.1f} s", f"{x['b3_s']:.1f} s", f"{x['duration_s']:.1f} s"] for x in p["log"]]
        cap = "Lifts"
    else:
        head = ["Step", "Start", "Duration", "Closed by"]
        rows = [[x["step"], f"{x['start_s']:.1f} s", f"{x['duration_s']:.1f} s", x["closed_by"]]
                for x in p["log"][-12:]]
        cap = "Steps (last 12)"
    return _table(cap, head, rows)


def _table(caption: str, head: list, rows: list) -> str:
    return (f"<table class='lv-table'><caption>{html.escape(caption)}</caption><tr>"
            + "".join(f"<th>{html.escape(str(h))}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
            + "</table>")


def summary_html(s: dict | None) -> str:
    if not s:
        return ""
    out = [f"<div class='lv-status'>Session <b>{html.escape(s['session_id'])}</b>: {s['duration_s']:.0f} s, "
           f"{s['processed']} frames processed ({_v(s['fps_mean'], 'fps')}), delay p50 "
           f"{_v(s['delay_p50_ms'], 'ms', '{:.0f}')}, p95 {_v(s['delay_p95_ms'], 'ms', '{:.0f}')}, "
           f"recording: {'yes' if s['recording'] else 'no'}"
           + (f" ({html.escape(s['recording_error'])})" if s.get("recording_error") else "") + "</div>"]
    for st in s["stations"]:
        out.append(_table(f"Station {st['number']}: {st['title']}", ["Measure", "Value"], st["rows"]))
    return "".join(out)


def build_live_tab(app, gr, run_dd):
    """The Live tab inside the Blocks of GradioApp.build()."""
    live = app.live
    choices = station_choices()
    default = app.cfg["live"]["default_template"]
    default = default if default in [c[1] for c in choices] else (choices[0][1] if choices else None)
    with gr.Tab("Live"):
        with gr.Row():
            with gr.Column(scale=3):
                station = gr.Dropdown(choices, value=default, label="Station")
                info = gr.Markdown(station_info(default))
                with gr.Row(visible=default == "workbench") as zone_row:
                    b1 = gr.Slider(0.1, 0.9, value=1 / 3, step=0.01, label="Border left | centre (image width)")
                    b2 = gr.Slider(0.1, 0.9, value=2 / 3, step=0.01, label="Border centre | right")
            with gr.Column(scale=2):
                save = gr.Checkbox(False, label="Save recording (raw camera frames to data/public/live/)")
                with gr.Row():
                    start = gr.Button("Start live", variant="primary")
                    stop = gr.Button("Stop")
                with gr.Row():
                    variant = gr.Radio(list(VARIANTS), value=app.cfg["pose"]["variant"], label="Model for analysis")
                    analyse = gr.Button("Analyse recording")
        msg = gr.Markdown()
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera (choose it below the picture)",
                           webcam_options=gr.WebcamOptions(mirror=False, constraints={
                               "video": {"width": {"ideal": 640}, "height": {"ideal": 480}}}))
            overlay = live_image(gr, label="Live overlay (click a person to follow them)", interactive=False,
                                 type="numpy")
        panel = gr.HTML(panel_html(None))
        log = gr.HTML()
        summary = gr.HTML()
        timer = gr.Timer(PANEL_S, active=False)

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
                return f"**{e.code}:** {e.text}", gr.Timer(active=False), panel_html(live.panel())
            return (f"Session **{sid}** started. Press the record button under the camera picture to send frames.",
                    gr.Timer(active=True), panel_html(live.panel()))

        def on_stop():
            s = live.stop()
            p = live.panel()
            return ("Stopped." if s else "No session was running."), gr.Timer(active=False), panel_html(p), \
                log_html(p), summary_html(s)

        def on_tick():
            p = live.panel()
            running = p is not None and p["status"] == "running"
            out = [panel_html(p, app.port), log_html(p)]
            if not running:                       # stopped by itself (idle or max time)
                out += [summary_html(live.last_summary), gr.Timer(active=False)]
            else:
                out += [gr.skip(), gr.skip()]
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
                return "Following the selected person."
            except AvError as e:
                return f"**{e.code}:** {e.text}"

        on_click.__annotations__["evt"] = gr.SelectData     # a real type, so Gradio passes the click

        def on_analyse(var):
            s = live.last_summary
            if not s:
                return "Stop a session with Save recording on first.", gr.skip()
            try:
                run_id = live.to_batch(s["session_id"], var, app.runs)
            except AvError as e:
                return f"**{e.code}:** {e.text}" + (f" ({e.detail})" if e.detail else ""), gr.skip()
            return (f"Batch run **{run_id}** started on the recording (pose on GPU {app.cfg['gpu']['pose']}); "
                    "follow it in the Run tab.", gr.update(choices=app.run_choices(), value=run_id))

        def on_load():
            p = live.panel()
            running = p is not None and p["status"] == "running"
            return panel_html(p, app.port), log_html(p), summary_html(None if running else live.last_summary), \
                gr.Timer(active=running)

        station.change(on_station, [station, b1, b2], [info, zone_row])
        b1.release(on_borders, [b1, b2], None)
        b2.release(on_borders, [b1, b2], None)
        start.click(on_start, [station, save, b1, b2], [msg, timer, panel])
        stop.click(on_stop, None, [msg, timer, panel, log, summary])
        timer.tick(on_tick, None, [panel, log, summary, timer])
        cam.stream(on_frame, cam, overlay, stream_every=app.cfg["live"]["stream_every_s"], time_limit=None,
                   concurrency_limit=1, show_progress="hidden", trigger_mode="always_last")
        overlay.select(on_click, None, msg)
        analyse.click(on_analyse, variant, [msg, run_dd])
        return on_load, [panel, log, summary, timer]
