"""GradioApp (LLD 3.10): the Run, Results and Live tabs. Suggestions, Compare
and Reports follow.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.ui.app [--sources qisens,public] [--port 7860] [--config FILE]

then, on the laptop: ssh -L 7860:127.0.0.1:7860 qisens@<server> and open
http://127.0.0.1:7860. The page binds 127.0.0.1 only, on a port from 7860 to
7869 (default 7860), and never makes a share link (fixed in code, AV-APP-005
otherwise).

Live mode (LLD 3.12) runs inside this process on GPU live.gpu (the A4000):
CUDA_VISIBLE_DEVICES is set to that GPU before torch is imported, so this
process never touches the batch pose GPU; batch runs are separate workers.

Data areas: only the sources named in --sources (default public) are ever
read; with qisens left out nothing under data/qisens/ is opened or listed.
Gradio's own temporary files (uploads, pictures sent to the page) go to
data/qisens/tmp/gradio when qisens is open, else to data/public/tmp/gradio, and
run files (the overlay video) are served in place from their run folder.
Analytics and Hugging Face calls are switched off and no web fonts are loaded.

The page never runs analysis (NFR-PERF-04): Start asks RunManager for a
worker process, and a timer reads the stored status every ui.poll_s. The page
keeps only the selected run ID; a reload restores the view from the run
folders (FR-APP-08), choosing ?run=<run_id>, else the run in progress, else the
newest run.
"""
from __future__ import annotations

import argparse
import html
import ipaddress
import os
import sys
from pathlib import Path

from ..config import DEFAULT_LAYERS, VARIANTS, Config
from ..errors import AvError
from ..storage.files import REPO_ROOT, SOURCES, source_root
from ..storage.runstore import FINISHED, RunStore

HOST = "127.0.0.1"
PORT = 7860
PORTS = range(7860, 7870)
ALL_STEPS = "All steps"
STAGE_NAMES = {"ingest": "Ingest", "shots": "Shots", "reference": "Reference frames", "pose_body": "Pose",
               "tracking": "Tracking", "segmentation": "Steps (Engine A)", "metrics": "Motion metrics",
               "overlay": "Overlay video", "eval": "Evaluation"}

CSS = """
.gradio-container { max-width: 1400px !important; }
#av-header { border-bottom: 3px solid #000; padding: 6px 0 10px 0; margin-bottom: 6px; }
#av-header h1 { font-size: 2.1em; margin: 0; letter-spacing: 0.02em; }
#av-header p { margin: 2px 0 0 0; color: #444; font-size: 1.05em; }
.av-bar { border: 2px solid #000; height: 26px; background: #fff; }
.av-bar > div { height: 100%; background: #000; }
.av-status table { border-collapse: collapse; width: 100%; margin-top: 8px; }
.av-status td, .av-status th { border-bottom: 1px solid #bbb; padding: 4px 8px; text-align: left; }
.av-status .running td { font-weight: bold; }
.av-msg { border-left: 4px solid #000; padding: 4px 10px; }
"""
# keep the page light even when the browser prefers dark mode
JS_LIGHT = "() => { document.body.classList.remove('dark'); }"


def check_bind(host: str, share: bool, port: int = PORT) -> None:
    """AV-APP-005 unless host is loopback or a private address, the port is 7860
    to 7869 and share is off (NFR-SEC-03)."""
    if share:
        raise AvError("AV-APP-005", detail="share=True")
    if port not in PORTS:
        raise AvError("AV-APP-005", detail=f"port {port} is not in 7860 to 7869")
    try:
        ip = ipaddress.ip_address("127.0.0.1" if host == "localhost" else host)
    except ValueError as e:
        raise AvError("AV-APP-005", detail=f"host {host!r}") from e
    if ip.is_unspecified or not (ip.is_loopback or ip.is_private):
        raise AvError("AV-APP-005", detail=f"host {host}")


def prepare_env(sources: list[str], cfg: Config, repo_root: Path = REPO_ROOT) -> Path:
    """Environment for Gradio, set before it is imported; returns its temp folder."""
    area = "qisens" if "qisens" in sources else "public"
    tmp = source_root(area, cfg["paths"], repo_root) / "tmp" / "gradio"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(tmp)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["MPLCONFIGDIR"] = str(repo_root / ".cache" / "matplotlib")
    os.environ["XDG_CONFIG_HOME"] = str(repo_root / ".config")
    os.environ.setdefault("XDG_CACHE_HOME", str(repo_root / ".cache"))
    os.environ["TMPDIR"] = str(tmp)
    # live pose sees only the live GPU (PCI bus order, CLAUDE.md); batch workers set their own
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["live"]["gpu"])
    os.environ["YOLO_OFFLINE"] = "1"
    import tempfile
    tempfile.tempdir = None
    return tmp


def _fmt(v: float | None, unit: str = "") -> str:
    if v is None:
        return ""
    txt = f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}" if abs(v) >= 1 else f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{txt} {unit}".strip()


def _err_text(e: AvError) -> str:
    return f"{e.code}: {e.text}" + (f" ({e.detail})" if e.detail else "")


class GradioApp:
    def __init__(self, cfg: Config, sources: list[str], repo_root: Path = REPO_ROOT, worker_cmd=None,
                 port: int = PORT, live=None):
        from ..live.manager import LiveManager
        from ..runs.manager import RunManager
        self.cfg = cfg
        self.store = RunStore(sources, cfg, repo_root)
        self.runs = RunManager(self.store, cfg, worker_cmd)
        self.sources = self.store.sources
        self.port = port
        # live sessions are filmed with consent and kept in the public area (data/public/live/)
        self.live = live or LiveManager(cfg, source_root("public", cfg["paths"], repo_root) / "live", port=port)

    # data for the page (plain functions, tested without a browser) -----------------

    def run_choices(self) -> list[tuple[str, str]]:
        return [(f"{r['run_id']} | {r['source']} | {r['video'] or '?'} | {r['status']}", r["run_id"])
                for r in self.store.list_runs()]

    def video_choices(self) -> list[tuple[str, str]]:
        return [(label, path) for label, path in self.store.list_videos()]

    def initial_run(self, query_run: str | None) -> str | None:
        if query_run:
            try:
                self.store.paths(query_run)
                return query_run
            except AvError:
                pass
        active = self.store.active_run()
        if active:
            return active
        runs = self.store.list_runs()
        return runs[0]["run_id"] if runs else None

    def status_html(self, run_id: str | None) -> tuple[str, bool]:
        """Progress bar, stage list, summary and error; and whether to keep polling."""
        if not run_id:
            return "<div class='av-status'>No run selected. Choose a video and press Start.</div>", False
        try:
            st = self.store.status(run_id)
        except AvError as e:
            return f"<div class='av-msg'>{html.escape(_err_text(e))}</div>", False
        pct = round(100 * st.progress)
        head = {"queued": "Queued", "running": "Running", "done": "Finished", "partial": "Finished in part",
                "failed": "Failed"}.get(st.status, st.status)
        now = f", now: {STAGE_NAMES.get(st.stage, st.stage)}" if st.stage else ""
        out = [f"<div class='av-status'><p><b>{html.escape(run_id)}</b> ({st.source}): <b>{head}</b>{now}, "
               f"{pct} %</p><div class='av-bar'><div style='width:{pct}%'></div></div>"]
        if st.waiting:
            out.append(f"<p class='av-msg'>{html.escape(st.waiting)}</p>")
        if st.stages:
            out.append("<table><tr><th>Stage</th><th>Status</th><th>Time</th></tr>")
            for name, s, sec, frac in st.stages:
                extra = f" {round(100 * frac)} %" if s == "running" else ""
                out.append(f"<tr class='{s}'><td>{STAGE_NAMES.get(name, name)}</td><td>{s}{extra}</td>"
                           f"<td>{'' if sec is None else f'{sec:.1f} s'}</td></tr>")
            out.append("</table>")
        if st.summary:
            out.append(f"<p><b>Summary</b><br>{html.escape(st.summary).replace(chr(10), '<br>')}</p>")
        if st.error:
            out.append(f"<p class='av-msg'><b>Error</b> {html.escape(st.error)}</p>")
        if st.status not in FINISHED and not st.active:
            out.append("<p class='av-msg'>No worker process is running for this run.</p>")
        out.append("</div>")
        return "".join(out), st.active

    def subject_gallery(self, run_id: str | None) -> list:
        from .charts import mark_subject
        if not run_id:
            return []
        try:
            items = self.store.reference_items(run_id)
        except AvError:
            return []
        out = []
        for it in items:
            short = it["shot_id"].rsplit("-", 1)[-1]
            if it["subject"]:
                tid = it["subject"].rsplit("-", 1)[-1]
                cap = f"{short} ({it['shot_type']}): subject {tid}, {it['subject_source'] or 'auto'}"
                if it["subject_box"] is None:
                    cap += ", never seen"
                elif it["box_frame"] != it["frame_idx"]:
                    cap += f", not seen here: box from frame {it['box_frame']}"
            else:
                cap = f"{short} ({it['shot_type']}): no subject"
            out.append((mark_subject(it["png"], it["subject_box"]), cap))
        return out

    def availability(self, run_id: str) -> str:
        """Which results exist; a failed or skipped stage says why (TC-APP-09)."""
        m = self.store.get_run(run_id)
        lines = []
        for name in ("segmentation", "metrics", "overlay", "eval"):
            rec = m.get("stages", {}).get(name)
            if rec is None:
                continue
            s = rec.get("status")
            if s == "failed":
                err = rec.get("error", {})
                lines.append(f"- **{STAGE_NAMES[name]}**: not available. {err.get('code', '')} {err.get('text', '')}")
            elif s in ("skipped", "pending", "running"):
                lines.append(f"- **{STAGE_NAMES[name]}**: not available ({s}).")
        if m.get("metrics", {}).get("stale"):
            lines.append("- **Motion metrics** belong to an earlier step result (stale).")
        return "\n".join(lines)

    def steps_table(self, run_id: str) -> list[list]:
        return [[s["step"], s["label"], _fmt(s["start_s"]), _fmt(s["end_s"]), _fmt(s["duration_s"]), s["source"]]
                for s in self.store.get_segments(run_id) if s["label"] != "other"]

    def _step_names(self, run_id: str) -> dict[int, str]:
        """segment index (metrics.csv step column) -> 'N label' as in steps.csv."""
        return {s["seg"]: f"{s['step']} {s['label']}" for s in self.store.get_segments(run_id)
                if s["label"] != "other"}

    def step_choices(self, run_id: str) -> list:
        return [ALL_STEPS] + [(name, str(seg)) for seg, name in self._step_names(run_id).items()]

    def metrics_table(self, run_id: str, step: str | None) -> list[list]:
        rows = self.store.get_metrics(run_id)
        if step and step != ALL_STEPS:
            rows = [r for r in rows if r["step"] == int(step)]
        names = self._step_names(run_id)
        out = []
        for r in rows:
            value = _fmt(r["value"], r["unit"]) if r["value"] is not None else "n/a"
            out.append([names.get(r["step"], r["label"]), r["metric"], value,
                        "" if r["coverage"] is None else f"{100 * r['coverage']:.0f} %", r["flags"]])
        return out

    def bending_rows(self, run_id: str) -> list[tuple[str, list]]:
        rows = self.store.get_metrics(run_id)
        out = []
        for s in self.store.get_segments(run_id):
            if s["label"] == "other":
                continue
            vals = {r["metric"]: r["value"] for r in rows if r["step"] == s["seg"]}
            if not any(f"bend_b{b}_s" in vals for b in range(4)):
                continue
            out.append((f"{s['step']} {s['label']}", [vals.get(f"bend_b{b}_s") for b in range(4)]))
        return out

    def baseline_table(self, run_id: str) -> list[list]:
        return [[b["label"], b["n"], _fmt(b["median_s"], "s"), _fmt(b["q1_s"], "s"), _fmt(b["q3_s"], "s"),
                 b.get("flags", "")] for b in self.store.get_baseline(run_id)]

    def eval_text(self, run_id: str) -> str:
        ev = self.store.get_eval(run_id)
        if not ev:
            return "No evaluation for this run (it needs hand labels, run_pose --gt)."
        m = ev.get("metrics", {})
        rows = [("Frame accuracy", _fmt(m.get("frame_accuracy"), "%")), ("Edit score", _fmt(m.get("edit"))),
                ("F1@10", _fmt(m.get("f1_10"))), ("F1@25", _fmt(m.get("f1_25"))), ("F1@50", _fmt(m.get("f1_50"))),
                ("Boundary error (mean)", _fmt(m.get("boundary_mae_s"), "s"))]
        txt = (f"Engine {ev.get('engine', 'A')} against `{ev.get('gt_file', '?')}`, "
               f"{ev.get('gt_steps', '?')} labelled and {ev.get('pred_steps', '?')} found steps.\n\n"
               "| Measure | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows))
        return txt

    def notices_text(self, run_id: str) -> str:
        notes = self.store.get_run(run_id).get("notices", [])
        if not notes:
            return "No notices."
        return "\n".join(f"- {n.get('stage', '')}: {n.get('code', '')} {n.get('text', '')}" for n in notes)

    def results(self, run_id: str | None) -> dict:
        """Everything of the Results tab for one run (TC-APP-10: reads only)."""
        from .charts import bending_bars, timeline
        if not run_id:
            return {"note": "No run selected.", "overlay": None, "timeline": None, "steps": [], "steps_choices":
                    [ALL_STEPS], "metrics": [], "bending": None, "baseline": [], "eval": "", "notices": ""}
        m = self.store.get_run(run_id)
        segs = self.store.get_segments(run_id)
        duration = float(m.get("ingest", {}).get("cfr", {}).get("duration_s") or (segs[-1]["end_s"] if segs else 0))
        bands = m.get("metrics", {}).get("params", {}).get("motion", {}).get("angle_bands") \
            or self.cfg["motion"]["angle_bands"]
        bend = self.bending_rows(run_id)
        ov = self.store.overlay_path(run_id)
        return {"note": self.availability(run_id), "overlay": str(ov) if ov else None,
                "timeline": timeline(segs, duration) if segs else None, "steps": self.steps_table(run_id),
                "steps_choices": self.step_choices(run_id), "metrics": self.metrics_table(run_id, ALL_STEPS),
                "bending": bending_bars(bend, bands) if bend else None, "baseline": self.baseline_table(run_id),
                "eval": self.eval_text(run_id), "notices": self.notices_text(run_id)}

    # actions --------------------------------------------------------------------

    def start(self, video: str | None, variant: str, source_hint: str | None = None) -> tuple[str | None, str]:
        """Create and start a run; (run_id or None, message)."""
        if not video:
            return None, _err_text(AvError("AV-APP-002"))
        source = next((s for s in self.sources if Path(video).resolve().is_relative_to(self.store.roots[s])), None)
        if source is None:
            return None, _err_text(AvError("AV-RUN-001", name=Path(video).name))
        try:
            run_id, notices = self.runs.create_run(video, source, variant)
            self.runs.start(run_id)
        except AvError as e:
            return None, _err_text(e)
        msg = f"Run {run_id} started on {Path(video).name} (model {variant}, Engine A)."
        msg += "".join(f"\n\n{_err_text(n)}" for n in notices)
        return run_id, msg

    def upload(self, file: str | None, source: str) -> tuple[str | None, str]:
        if not file:
            return None, ""
        try:
            dest = self.store.save_upload(file, Path(file).name, source)
        except AvError as e:
            return None, _err_text(e)
        return str(dest), f"Uploaded to {source}/input/{dest.name}."

    # page ----------------------------------------------------------------------------

    def build(self):
        import gradio as gr
        font = [gr.themes.Font("DejaVu Sans"), gr.themes.Font("Arial"), "sans-serif"]
        theme = gr.themes.Base(primary_hue=gr.themes.colors.neutral, neutral_hue=gr.themes.colors.neutral,
                               text_size=gr.themes.sizes.text_lg, radius_size=gr.themes.sizes.radius_none,
                               font=font, font_mono=[gr.themes.Font("DejaVu Sans Mono"), "monospace"]).set(
            body_background_fill="white", body_text_color="black", block_border_color="#999999",
            button_primary_background_fill="black", button_primary_text_color="white",
            button_primary_background_fill_hover="#333333", button_secondary_background_fill="white",
            button_secondary_border_color="black", button_secondary_text_color="black")
        poll_s = self.cfg["ui"]["poll_s"]
        variant0 = self.cfg["pose"]["variant"]

        # delete_cache: every hour, Gradio's temp files older than a day are removed
        from .live_tab import LIVE_CSS, build_live_tab
        with gr.Blocks(title="AssemblyVision", theme=theme, css=CSS + LIVE_CSS, js=JS_LIGHT, analytics_enabled=False,
                       delete_cache=(3600, 86400)) as demo:
            gr.HTML("<div id='av-header'><h1>AssemblyVision</h1><p>Team 4, QiSens capstone: motion analysis of "
                    "train assembly work</p></div>")
            with gr.Row():
                run_dd = gr.Dropdown(label="Run", choices=[], value=None, scale=6, allow_custom_value=False)
                refresh = gr.Button("Refresh list", scale=1)
            timer = gr.Timer(poll_s, active=False)
            with gr.Tabs():
                with gr.Tab("Run"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            video_dd = gr.Dropdown(label="Video", choices=self.video_choices(), value=None)
                            with gr.Row():
                                upload = gr.File(label="Upload a video (mp4 or avi)", file_types=[".mp4", ".avi"],
                                                 type="filepath", scale=3)
                                up_src = gr.Radio(self.sources, value=self.sources[0], label="Upload to",
                                                  visible=len(self.sources) > 1, scale=1)
                        with gr.Column(scale=2):
                            variant = gr.Radio(list(VARIANTS), value=variant0, label="Pose model variant")
                            gr.Radio(["A"], value="A", label="Step engine", interactive=False,
                                     info="Engine B is not trained yet")
                            start_btn = gr.Button("Start", variant="primary")
                    msg = gr.Markdown()
                    status = gr.HTML()
                    gallery = gr.Gallery(label="Subject per shot (S = the worker the analysis follows)",
                                         columns=3, height="auto", object_fit="contain", allow_preview=True)
                with gr.Tab("Results"):
                    note = gr.Markdown()
                    overlay = gr.Video(label="Overlay video", interactive=False, height=520)
                    timeline_img = gr.Image(label="Process timeline (Engine A)", interactive=False,
                                            type="numpy", format="png")
                    gr.Markdown("### Steps")
                    steps_df = gr.Dataframe(headers=["Step", "Label", "Start s", "End s", "Duration s", "Source"],
                                            interactive=False, wrap=True)
                    gr.Markdown("### Motion metrics per step")
                    step_dd = gr.Dropdown(label="Step", choices=[ALL_STEPS], value=ALL_STEPS)
                    metrics_df = gr.Dataframe(headers=["Step", "Metric", "Value", "Coverage", "Flags"],
                                              interactive=False, wrap=True, max_height=520)
                    bend_img = gr.Image(label="Back bending: time per band and step", interactive=False,
                                        type="numpy", format="png")
                    gr.Markdown("### Cycle baseline")
                    base_df = gr.Dataframe(headers=["Label", "n", "Median", "Q1", "Q3", "Flags"],
                                           interactive=False)
                    gr.Markdown("### Evaluation")
                    eval_md = gr.Markdown()
                    with gr.Accordion("Notices of this run", open=False):
                        notices_md = gr.Markdown()
                live_load, live_out = build_live_tab(self, gr, run_dd)

            results_out = [note, overlay, timeline_img, steps_df, step_dd, metrics_df, bend_img, base_df, eval_md,
                           notices_md]

            def fill_results(run_id):
                r = self.results(run_id)
                return [r["note"], r["overlay"], r["timeline"], r["steps"],
                        gr.update(choices=r["steps_choices"], value=ALL_STEPS), r["metrics"], r["bending"],
                        r["baseline"], r["eval"], r["notices"]]

            def on_select(run_id):
                try:
                    html_, active = self.status_html(run_id)
                    return [html_, self.subject_gallery(run_id), gr.Timer(active=active), *fill_results(run_id)]
                except AvError as e:
                    return [f"<div class='av-msg'>{html.escape(_err_text(e))}</div>", [], gr.Timer(active=False),
                            *[gr.skip()] * len(results_out)]

            def on_poll(run_id):
                self.runs.reap()
                html_, active = self.status_html(run_id)
                if active or not run_id:
                    return [html_, gr.skip(), gr.Timer(active=active), *[gr.skip()] * len(results_out)]
                # finished: fill everything once and stop the timer (TC-APP-07)
                return [html_, self.subject_gallery(run_id), gr.Timer(active=False), *fill_results(run_id)]

            def on_load(request):
                q = dict(request.query_params) if request is not None else {}
                run_id = self.initial_run(q.get("run"))
                return gr.update(choices=self.run_choices(), value=run_id)
            on_load.__annotations__["request"] = gr.Request     # a real type, so Gradio passes the request

            def on_refresh(run_id):
                return gr.update(choices=self.run_choices(), value=run_id), \
                    gr.update(choices=self.video_choices())

            def on_start(video, var):
                run_id, text = self.start(video, var)
                if run_id is None:
                    return gr.skip(), f"<div class='av-msg'>{html.escape(text)}</div>"
                return gr.update(choices=self.run_choices(), value=run_id), text

            def on_upload(file, src):
                dest, text = self.upload(file, src)
                if dest is None:
                    return gr.skip(), text, None
                return gr.update(choices=self.video_choices(), value=dest), text, None

            sel_out = [status, gallery, timer, *results_out]
            demo.load(on_load, None, run_dd)
            demo.load(live_load, None, live_out)
            run_dd.change(on_select, run_dd, sel_out)
            timer.tick(on_poll, run_dd, sel_out)
            refresh.click(on_refresh, run_dd, [run_dd, video_dd])
            start_btn.click(on_start, [video_dd, variant], [run_dd, msg])
            upload.upload(on_upload, [upload, up_src], [video_dd, msg, upload])
            step_dd.change(lambda r, s: self.metrics_table(r, s) if r else [], [run_dd, step_dd], metrics_df)
        self.demo = demo
        return demo

    def launch(self, host: str = HOST, port: int | None = None, share: bool = False) -> None:
        port = getattr(self, "port", PORT) if port is None else port
        check_bind(host, share, port)
        import gradio as gr
        gr.set_static_paths([str(root / "runs") for root in self.store.roots.values()])
        demo = self.build()
        demo.queue(default_concurrency_limit=4)
        demo.launch(server_name=host, server_port=port, share=False, inbrowser=False, show_api=False,
                    allowed_paths=[str(r) for r in self.store.roots.values()],
                    max_file_size=f"{self.cfg['ui']['max_upload_mb']}mb")


def parse_sources(text: str) -> list[str]:
    names = [s.strip() for s in text.split(",") if s.strip()]
    if not names or any(s not in SOURCES for s in names):
        raise argparse.ArgumentTypeError(f"--sources takes a comma list of {', '.join(SOURCES)}")
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.ui.app", description="Start the AssemblyVision page.")
    p.add_argument("--sources", type=parse_sources, default=["public"],
                   help="data areas the page may read, comma list of qisens,public (default public)")
    p.add_argument("--port", type=int, default=PORT, choices=list(PORTS), metavar="7860..7869",
                   help="port on 127.0.0.1 (7860 to 7869, default 7860)")
    p.add_argument("--config", action="append", type=Path, default=[], help="extra TOML layer")
    args = p.parse_args(argv)
    try:
        cfg = Config.load([*DEFAULT_LAYERS, *args.config])
        tmp = prepare_env(args.sources, cfg)
        check_bind(HOST, False, args.port)
        app = GradioApp(cfg, args.sources, port=args.port)
        print(f"AssemblyVision page on http://{HOST}:{args.port} (sources {','.join(app.sources)}, temp {tmp})",
              flush=True)
        app.launch()
    except AvError as e:
        print(_err_text(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
