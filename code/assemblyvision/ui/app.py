"""GradioApp (LLD 3.10): a left sidebar and four pages, New analysis, Results,
Runs and Live. Suggestions, Compare and Reports follow.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.ui.app [--sources qisens,public] [--port 7860] [--config FILE]

then, on the laptop: ssh -L 7860:127.0.0.1:7860 qisens@<server> and open
http://127.0.0.1:7860. The page binds 127.0.0.1 only, on a port from 7860 to
7869 (default 7860), and never makes a share link (fixed in code, AV-APP-005
otherwise). ?run=<run_id> opens that run on Results, ?page=new|results|runs|live
opens a page.

Live mode (LLD 3.12) runs inside this process on GPU live.gpu (the A4000):
CUDA_VISIBLE_DEVICES is set to that GPU before torch is imported, so this
process never touches the batch pose GPU; batch runs are separate workers.

Data areas: only the sources named in --sources (default public) are ever
read; with qisens left out nothing under data/qisens/ is opened or listed.
Gradio's own temporary files (uploads, pictures sent to the page) go to
data/qisens/tmp/gradio when qisens is open, else to data/public/tmp/gradio, and
run files (the overlay video) are served in place from their run folder.
Charts are sent as inline PNG. Analytics and Hugging Face calls are switched
off and no web fonts are loaded. A QiSens video is shown by a neutral name
("QiSens video (9.9 min)"); its file name stays on disk.

The page never runs analysis (NFR-PERF-04): Start asks RunManager for a
worker process, and a timer reads the stored status every ui.poll_s. The page
keeps only run IDs; a reload restores the view from the run folders
(FR-APP-08), choosing ?run=<run_id>, else the run in progress, else the
newest run.

Look: views.py builds the HTML parts, style.css holds all styling.
"""
from __future__ import annotations

import argparse
import ipaddress
import os
import statistics
import sys
from pathlib import Path

from ..config import DEFAULT_LAYERS, VARIANTS, Config
from ..errors import AvError
from ..phases import PUBLIC_NOTE
from ..storage.files import REPO_ROOT, SOURCES, source_root
from ..storage.runstore import FINISHED, RunStore
from . import views as v

HOST = "127.0.0.1"
PORT = 7860
PORTS = range(7860, 7870)
ALL_STEPS = "All steps"
ALL_PHASES = "All phases"
REACH_CAPTION = ("Hand positions relative to the worker's own body, so camera movement does not matter. "
                 "Units: torso lengths.")
STAGE_NAMES = {"ingest": "Ingest", "shots": "Shots", "reference": "Reference frames", "pose_body": "Pose",
               "tracking": "Tracking", "segmentation": "Steps (Engine A)", "metrics": "Motion metrics",
               "overlay": "Overlay video", "eval": "Evaluation"}
PAGES = (("new", "New analysis"), ("results", "Results"), ("runs", "Runs"), ("live", "Live"))
UI_DIR = Path(__file__).resolve().parent
NO_LABELS = "No hand labels for this video, so accuracy cannot be scored."


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


def _theme(gr):
    sans = ["-apple-system", "BlinkMacSystemFont", gr.themes.Font("SF Pro Text"), gr.themes.Font("Inter"),
            gr.themes.Font("Helvetica Neue"), "Arial", "sans-serif"]
    mono = [gr.themes.Font("SF Mono"), "ui-monospace", "Menlo", gr.themes.Font("DejaVu Sans Mono"), "monospace"]
    return gr.themes.Base(primary_hue=gr.themes.colors.neutral, neutral_hue=gr.themes.colors.neutral,
                          text_size=gr.themes.sizes.text_md, radius_size=gr.themes.sizes.radius_lg,
                          spacing_size=gr.themes.sizes.spacing_md, font=sans, font_mono=mono).set(
        body_background_fill="#f7f7f5", body_text_color="#111111", body_text_color_subdued="#6b6b6b",
        background_fill_primary="#ffffff", background_fill_secondary="#f7f7f5",
        border_color_primary="#e5e5e5", border_color_accent="#111111", color_accent="#111111",
        color_accent_soft="#f0f0ee", block_background_fill="transparent", block_border_width="0px",
        block_shadow="none", block_label_background_fill="transparent", block_label_border_width="0px",
        block_label_text_color="#6b6b6b", block_title_text_color="#6b6b6b", block_title_text_weight="600",
        block_padding="0px", layout_gap="14px", panel_background_fill="transparent", panel_border_width="0px",
        input_background_fill="#ffffff", input_border_color="#141414", input_border_width="1px", input_border_color_focus="#111111",
        input_shadow="none", input_shadow_focus="0 0 0 3px rgba(0,0,0,0.08)", input_radius="10px",
        checkbox_background_color_selected="#111111", checkbox_border_color_selected="#111111",
        checkbox_label_background_fill="transparent", checkbox_label_background_fill_selected="transparent",
        checkbox_label_border_width="0px", slider_color="#111111", loader_color="#111111",
        button_primary_background_fill="#000000", button_primary_background_fill_hover="#2a2a2a",
        button_primary_text_color="#ffffff", button_primary_border_color="#000000",
        button_secondary_background_fill="#ffffff", button_secondary_background_fill_hover="#f0f0ee",
        button_secondary_text_color="#111111", button_secondary_border_color="#d4d4d2",
        button_large_radius="10px", button_small_radius="10px",
        table_border_color="#e5e5e5", table_even_background_fill="#ffffff", table_odd_background_fill="#ffffff")


class GradioApp:
    def __init__(self, cfg: Config, sources: list[str], repo_root: Path = REPO_ROOT, worker_cmd=None,
                 port: int = PORT, live=None):
        from ..live.manager import LiveManager
        from ..runs.manager import RunManager
        self.cfg = cfg
        self.repo_root = repo_root
        self.store = RunStore(sources, cfg, repo_root)
        self.runs = RunManager(self.store, cfg, worker_cmd)
        self.sources = self.store.sources
        self.port = port
        self._durations: dict[tuple[str, float], float | None] = {}
        self._reach: dict[tuple, tuple] = {}         # (run_id, phase, manifest and steps mtime) -> (png, coverage)
        # live sessions are filmed with consent and kept in the public area (data/public/live/)
        self.live = live or LiveManager(cfg, source_root("public", cfg["paths"], repo_root) / "live", port=port)

    # names shown on the page (a QiSens file name never is) ----------------------------

    def _video_seconds(self, path: str, runs: list[dict]) -> float | None:
        """Duration of a video: from the manifest of a run on it, else read once from the file header."""
        for r in runs:
            if r.get("video_path") == path and r.get("video_s"):
                return float(r["video_s"])
        try:
            key = (path, Path(path).stat().st_mtime)
        except OSError:
            return None
        if key not in self._durations:
            import cv2
            cap = cv2.VideoCapture(path)
            fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            self._durations[key] = n / fps if fps and n and fps > 0 else None
        return self._durations[key]

    def run_label(self, r: dict) -> str:
        return v.video_label(r["source"], r["video"] or "video", r.get("video_s"))

    def video_name(self, path: str) -> str:
        """Display name of one video path."""
        p = Path(path)
        source = next((s for s, root in self.store.roots.items() if p.resolve().is_relative_to(root)), "public")
        if source != "qisens":
            return p.name
        return v.qisens_alias(self._video_seconds(str(p), self.store.list_runs()))

    # data for the page (plain functions, tested without a browser) -----------------

    def listed_runs(self) -> list[dict]:
        """Runs shown on the page: with steps, or still queued or running (views.listed_runs)."""
        return v.listed_runs(self.store.list_runs())

    def run_choices(self) -> list[tuple[str, str]]:
        """Results picker: 'process_480p30.mp4 · 26 Oct, 12:38 · m', the run ID as value."""
        return [(v.run_option(self.run_label(r), r["created_utc"], r.get("variant"), r["status"]), r["run_id"])
                for r in self.listed_runs()]

    def video_choices(self) -> list[tuple[str, str]]:
        runs = None
        labels, paths = [], []
        for label, path in self.store.list_videos():
            if label.startswith("qisens/"):
                runs = self.store.list_runs() if runs is None else runs
                label = v.qisens_alias(self._video_seconds(path, runs))
            labels.append(label)
            paths.append(path)
        return list(zip(v.unique_labels(labels), paths))

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
        runs = self.listed_runs()
        return runs[0]["run_id"] if runs else None

    def status_html(self, run_id: str | None) -> tuple[str, bool]:
        """Progress bar, stage list and error of one run; and whether to keep polling."""
        if not run_id:
            return v.note("No analysis is running. Choose a video and press Start analysis."), False
        try:
            st = self.store.status(run_id)
            m = self.store.get_run(run_id)
        except AvError as e:
            return v.note(_err_text(e)), False
        pct = round(100 * st.progress)
        head = {"queued": "Queued", "running": "Running", "done": "Finished", "partial": "Finished in part",
                "failed": "Failed"}.get(st.status, st.status)
        now = f", now {STAGE_NAMES.get(st.stage, st.stage).lower()}" if st.stage else ""
        name = v.video_label(st.source, Path(m.get("video", "")).name, None if st.source != "qisens" else
                             self._video_seconds(m.get("video", ""), self.store.list_runs()))
        out = [f"<div class='av-job-head'><div><b>{v.esc(name)}</b> &nbsp;{v.status_pill(st.status)}</div>"
               f"<span>{v.esc(head)}{v.esc(now)}, {pct} %</span></div>", v.progress_bar(st.progress)]
        if st.waiting:
            out.append(f"<div class='av-job-msg'>{v.note(st.waiting)}</div>")
        if st.stages:
            rows = []
            for name_, s, sec, frac in st.stages:
                extra = f" {round(100 * frac)} %" if s == "running" else ""
                rows.append([STAGE_NAMES.get(name_, name_), v.status_pill(s) + v.esc(extra),
                             "" if sec is None else v.span(sec) if sec >= 60 else f"{sec:.1f} s"])
            out.append("<div style='margin-top:18px'>" + v.table(["Stage", "Status", "Time"], rows, num={2},
                                                                   raw={1}) + "</div>")
        if st.error:
            out.append(f"<div class='av-job-msg'>{v.note(st.error)}</div>")
        if st.status not in FINISHED and not st.active:
            out.append(f"<div class='av-job-msg'>{v.note('No worker process is running for this run.')}</div>")
        out.append(f"<div class='av-job-msg'><span class='av-pill av-mono'>{v.esc(run_id)}</span></div>")
        return "".join(out), st.active

    def job_kpis(self, run_id: str | None) -> str:
        """KPI cards of a finished run (New analysis page)."""
        if not run_id:
            return ""
        try:
            st = self.store.status(run_id)
            m = self.store.get_run(run_id)
        except AvError:
            return ""
        if st.status not in FINISHED:
            return ""
        pb, mc = m.get("pose_body", {}), m.get("metrics", {}).get("counts", {})
        n_rec = mc.get("records")
        items = [("Frames", f"{pb['frames']:,}" if pb.get("frames") else "n/a"),
                 ("Pose speed", f"{pb['pose_fps']:.0f} fps" if pb.get("pose_fps") else "n/a"),
                 ("Peak GPU memory", f"{pb['gpu_peak_mb']:.0f} MB" if pb.get("gpu_peak_mb") else "n/a"),
                 ("Shots", str(m.get("shots", {}).get("count", "n/a"))),
                 ("Tracks", str(m.get("tracking", {}).get("totals", {}).get("tracks_kept", "n/a"))),
                 ("Phases found", self._phases_found(run_id, m)),
                 ("Metrics reported", f"{n_rec - mc.get('null', 0)} of {n_rec}" if n_rec else "n/a")]
        return v.kpi_cards(items)

    def _phases_found(self, run_id: str, m: dict) -> str:
        seg = m.get("segmentation", {})
        if seg.get("phases_expected"):
            return f"{seg.get('phases_found', 0)} of {seg['phases_expected']}"
        labels = {s["label"] for s in self.store.get_segments(run_id) if s["label"] != "other"}
        return str(len(labels)) if labels or seg else "n/a"

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

    def availability_html(self, run_id: str) -> str:
        """The same as notes; evaluation has its own card and is left out here."""
        m = self.store.get_run(run_id)
        names = {"segmentation": "Steps", "metrics": "Motion metrics", "overlay": "The overlay video"}
        notes = []
        for name, title in names.items():
            rec = m.get("stages", {}).get(name)
            s = None if rec is None else rec.get("status")
            if s == "failed":
                err = rec.get("error", {})
                notes.append(f"<b>{title}</b> could not be made. {v.esc(err.get('code', ''))} "
                             f"{v.esc(err.get('text', ''))}")
            elif s == "skipped":
                notes.append(f"<b>{title}</b> were not computed in this run." if name != "overlay"
                             else f"<b>{title}</b> was not made in this run.")
            elif s in ("pending", "running"):
                notes.append(f"<b>{title}</b> are still being computed." if name != "overlay"
                             else f"<b>{title}</b> is still being made.")
        if m.get("metrics", {}).get("stale"):
            notes.append("<b>Motion metrics</b> belong to an earlier step result.")
        if not notes:
            return ""
        return "<div class='av-note-list'>" + "".join(f"<div class='av-note'>{n}</div>" for n in notes) + "</div>"

    def steps_table(self, run_id: str) -> list[list]:
        return [[s["step"], s["label"], _fmt(s["start_s"]), _fmt(s["end_s"]), _fmt(s["duration_s"]), s["source"]]
                for s in self.store.get_segments(run_id) if s["label"] != "other"]

    def steps_html(self, run_id: str) -> str:
        """Steps with the cycle count of their phase; median and range (the cycle
        baseline, FR-SEG-03) once some phase repeats."""
        base = {b["label"]: b for b in self.store.get_baseline(run_id)}
        spread = any((b.get("n") or 0) >= 2 for b in base.values())
        rows = []
        for s in self.store.get_segments(run_id):
            if s["label"] == "other":
                continue
            b = base.get(s["label"], {})
            row = [s["step"], v.esc(s["label"]), v.clock(s["start_s"]), v.clock(s["end_s"]),
                   f"{s['duration_s']:.1f} s", b.get("n", ""),
                   v.pill(s["source"].replace("engine_", "Engine ").replace("_", " "), "muted")]
            if spread:
                rng = f"{_fmt(b['q1_s'], 's')} to {_fmt(b['q3_s'], 's')}" \
                    if (b.get("n") or 0) >= 2 and b.get("q1_s") is not None else ""
                row[6:6] = [_fmt(b.get("median_s"), "s") if (b.get("n") or 0) >= 2 else "", rng]
            rows.append(row)
        head = ["Step", "Phase", "Start", "End", "Duration", "Cycles"] + \
            (["Median", "Range (Q1 to Q3)"] if spread else []) + ["Source"]
        out = v.section_title("Steps", "Phases found by Engine A, in video time (min:s).") + \
            v.table(head, rows, num={0, 2, 3, 4, 5, 6, 7} if spread else {0, 2, 3, 4, 5},
                    raw={1, len(head) - 1}, empty="No steps were stored for this run.")
        if rows and not spread:
            out += ("<p class='av-hint'>Each phase occurs once in this video, so there is no spread to show yet. "
                    "The median and range appear when a phase repeats.</p>")
        return out

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

    def _bands(self, run_id: str) -> list[float]:
        m = self.store.get_run(run_id)
        return m.get("metrics", {}).get("params", {}).get("motion", {}).get("angle_bands") \
            or self.cfg["motion"]["angle_bands"]

    def metrics_html(self, run_id: str | None, step: str | None) -> str:
        if not run_id:
            return ""
        try:
            rows = self.store.get_metrics(run_id)
        except AvError as e:
            return v.note(_err_text(e))
        if step and step != ALL_STEPS:
            rows = [r for r in rows if r["step"] == int(step)]
        names, bands = self._step_names(run_id), self._bands(run_id)
        out = [[v.esc(names.get(r["step"], r["label"])),
                ("" if v.metric_band(r["metric"]) is None else v.band_dot(v.metric_band(r["metric"])))
                + f"{v.esc(v.metric_label(r['metric'], bands))}<code class='av-key av-mono'>{v.esc(r['metric'])}</code>",
                v.metric_value(r["value"], r["unit"]),
                "" if r["coverage"] is None else f"{100 * r['coverage']:.0f} %", v.flag_pills(r["flags"])]
               for r in rows]
        return v.table(["Step", "Metric", "Value", "Coverage", "Flags"], out, num={2, 3}, raw={0, 1, 4},
                       empty="No motion metrics for this run.", scroll=True)

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

    def reach_choices(self, run_id: str) -> list[str]:
        labels = []
        for s in self.store.get_segments(run_id):
            if s["label"] != "other" and s["label"] not in labels:
                labels.append(s["label"])
        return [ALL_PHASES] + labels

    def reach(self, run_id: str, phase: str | None = ALL_PHASES) -> tuple:
        """Hand reach heatmap (RGB) and its coverage for one phase or all, computed
        once from the stored KTS files and kept per run and phase."""
        from ..analysis.motion import MotionAnalyzer
        from ..analysis.reach import run_reach
        from .charts import reach_heatmap
        p = self.store.paths(run_id)
        phase = phase or ALL_PHASES
        stamp = tuple(f.stat().st_mtime if f.is_file() else 0.0 for f in (p.manifest, p.steps_csv))
        key = (run_id, phase, stamp)
        if key not in self._reach:
            m = self.store.get_run(run_id)
            fps = float(m["ingest"]["cfr"]["fps"])
            subjects = MotionAnalyzer(self.cfg).load_subjects(m, p.root)
            r = run_reach(subjects, self.store.get_segments(run_id), fps, None if phase == ALL_PHASES else {phase})
            self._reach[key] = (reach_heatmap(r, "" if phase == ALL_PHASES else phase), r.coverage)
        return self._reach[key]

    def reach_html(self, run_id: str | None, phase: str | None = ALL_PHASES) -> str:
        if not run_id:
            return ""
        try:
            img, cov = self.reach(run_id, phase)
        except (AvError, OSError, KeyError, ValueError) as e:
            text = _err_text(e) if isinstance(e, AvError) else "the keypoint series of this run could not be read"
            return v.note(f"No hand reach heatmap: {text}.")
        out = f"<p class='av-hint'>Coverage {100 * cov:.0f} % (share of the frames used).</p>"
        if cov < 0.5:
            out += v.note(f"Coverage is low ({100 * cov:.0f} %): the wrists, shoulders or hips were often not "
                          "seen, so the map shows only part of the work.")
        return out + v.img_html(img, "Hand reach heatmap")

    def baseline_table(self, run_id: str) -> list[list]:
        return [[b["label"], b["n"], _fmt(b["median_s"], "s"), _fmt(b["q1_s"], "s"), _fmt(b["q3_s"], "s"),
                 b.get("flags", "")] for b in self.store.get_baseline(run_id)]

    def baseline_html(self, run_id: str) -> str:
        rows = [[v.esc(b["label"]), b["n"], _fmt(b["median_s"], "s"), _fmt(b["q1_s"], "s"), _fmt(b["q3_s"], "s"),
                 v.flag_pills(b.get("flags", ""))] for b in self.store.get_baseline(run_id)]
        return v.section_title("Cycle baseline", "Duration per phase over all its cycles: median and quartiles.") \
            + v.table(["Phase", "Cycles", "Median", "Q1", "Q3", "Flags"], rows, num={1, 2, 3, 4}, raw={0, 5},
                      empty="No cycle baseline for this run.")

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

    def eval_html(self, run_id: str) -> str:
        ev = self.store.get_eval(run_id)
        head = v.section_title("Evaluation", "Steps found by the engine against hand labels.")
        if not ev:
            return head + v.note(NO_LABELS)
        m = ev.get("metrics", {})
        items = [("Frame accuracy", _fmt(m.get("frame_accuracy"), "%") or "n/a"),
                 ("Edit score", _fmt(m.get("edit")) or "n/a"), ("F1@10", _fmt(m.get("f1_10")) or "n/a"),
                 ("F1@25", _fmt(m.get("f1_25")) or "n/a"), ("F1@50", _fmt(m.get("f1_50")) or "n/a"),
                 ("Mean boundary error", _fmt(m.get("boundary_mae_s"), "s") or "n/a")]
        sub = (f"Engine {ev.get('engine', 'A')} against {ev.get('gt_file', '?')}: {ev.get('gt_steps', '?')} "
               f"labelled and {ev.get('pred_steps', '?')} found steps.")
        return head + v.kpi_cards(items) + f"<p class='av-hint'>{v.esc(sub)}</p>"

    def notices_text(self, run_id: str) -> str:
        notes = self.store.get_run(run_id).get("notices", [])
        if not notes:
            return "No notices."
        return "\n".join(f"- {n.get('stage', '')}: {n.get('code', '')} {n.get('text', '')}" for n in notes)

    def notices_html(self, run_id: str) -> str:
        notes = self.store.get_run(run_id).get("notices", [])
        sub = f"{len(notes)} notices in {len(v.group_notices(notes))} groups. Open a group for the full list." \
            if notes else ""
        return v.section_title("Notices", sub) + "<div class='av-scroll'>" + v.notices_html(notes) + "</div>"

    def results_kpis(self, run_id: str) -> str:
        m = self.store.get_run(run_id)
        segs = [s for s in self.store.get_segments(run_id) if s["label"] != "other"]
        dur = m.get("ingest", {}).get("cfr", {}).get("duration_s")
        pb, mc = m.get("pose_body", {}), m.get("metrics", {}).get("counts", {})
        n_rec = mc.get("records")
        items = [("Video length", v.clock(dur) if dur else "n/a", "min:s"),
                 ("Phases found", self._phases_found(run_id, m)),
                 ("Mean phase duration", f"{statistics.mean(s['duration_s'] for s in segs):.0f} s" if segs else "n/a"),
                 ("Frames with a person", f"{100 * pb['frames_with_person'] / pb['frames']:.0f} %"
                  if pb.get("frames") and pb.get("frames_with_person") is not None else "n/a"),
                 ("Metrics reported", f"{n_rec - mc.get('null', 0)} of {n_rec}" if n_rec else "n/a")]
        ev = self.store.get_eval(run_id)
        if ev:
            em = ev.get("metrics", {})
            items += [("Frame accuracy", _fmt(em.get("frame_accuracy"), "%") or "n/a"),
                      ("Mean boundary error", _fmt(em.get("boundary_mae_s"), "s") or "n/a")]
        return v.kpi_cards(items)

    def results_head(self, run_id: str | None) -> str:
        if not run_id:
            return v.page_header("Results", "No run yet. Start one on New analysis.")
        try:
            r = next(x for x in self.store.list_runs() if x["run_id"] == run_id)
        except StopIteration:
            return v.page_header("Results", run_id)
        bits = [self.run_label(r), f"model {r['variant']}" if r.get("variant") else "", v.utc_date(r["created_utc"])]
        head = v.page_header("Results", ", ".join(b for b in bits if b))
        return head + (v.banner(PUBLIC_NOTE) if r["source"] == "public" else "")

    def results(self, run_id: str | None) -> dict:
        """Everything of the Results page for one run (TC-APP-10: reads only)."""
        from .charts import bending_bars, timeline
        if not run_id:
            return {"note": "No run selected.", "overlay": None, "timeline": None, "steps": [], "steps_choices":
                    [ALL_STEPS], "metrics": [], "bending": None, "baseline": [], "eval": "", "notices": "",
                    "head": self.results_head(None), "kpis": "", "note_html": "", "steps_html": "",
                    "metrics_html": "", "baseline_html": "", "eval_html": "", "notices_html": ""}
        m = self.store.get_run(run_id)
        segs = self.store.get_segments(run_id)
        duration = float(m.get("ingest", {}).get("cfr", {}).get("duration_s") or (segs[-1]["end_s"] if segs else 0))
        bands = self._bands(run_id)
        bend = self.bending_rows(run_id)
        ov = self.store.overlay_path(run_id)
        return {"note": self.availability(run_id), "overlay": str(ov) if ov else None,
                "timeline": timeline(segs, duration) if segs else None, "steps": self.steps_table(run_id),
                "steps_choices": self.step_choices(run_id), "metrics": self.metrics_table(run_id, ALL_STEPS),
                "bending": bending_bars(bend, bands) if bend else None, "baseline": self.baseline_table(run_id),
                "eval": self.eval_text(run_id), "notices": self.notices_text(run_id),
                "head": self.results_head(run_id), "kpis": self.results_kpis(run_id),
                "note_html": self.availability_html(run_id), "steps_html": self.steps_html(run_id),
                "metrics_html": self.metrics_html(run_id, ALL_STEPS), "baseline_html": self.baseline_html(run_id),
                "eval_html": self.eval_html(run_id), "notices_html": self.notices_html(run_id)}

    def runs_html(self, current: str | None = None) -> str:
        """The Runs page table, newest first, with an Open button per run."""
        rows = []
        for r in self.listed_runs():
            took = v.utc_seconds(r["created_utc"], r.get("finished_utc"))
            btn = f"<button class='av-open' data-open-run='{v.esc(r['run_id'])}'>Open</button>"
            rows.append([f"<span class='av-mono'>{v.esc(r['run_id'])}</span>", v.pill(r["source"]),
                         v.esc(self.run_label(r)), v.pill(r["variant"], "outline", mono=True) if r.get("variant") else "",
                         v.esc(v.utc_date(r["created_utc"])),
                         v.span(took) if took is not None else "", v.status_pill(r["status"]), btn])
        return v.table(["Run ID", "Data", "Video", "Model", "Date (UTC)", "Duration", "Status", ""], rows,
                       num={5}, raw={0, 1, 2, 3, 6, 7}, empty="No runs yet. Start one on New analysis.", scroll=True)

    def live_running(self) -> bool:
        s = self.live.session
        return s is not None and s.status == "running"

    def server_html(self) -> str:
        return v.server_pill(v.gpu_busy(self.repo_root / ".locks"), self.live_running())

    def run_id_html(self, run_id: str | None) -> str:
        return f"<div class='av-run-id av-mono'>{v.esc(run_id or '')}</div>"

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
        msg = f"Run {run_id} started on {self.video_name(video)} (model {variant}, Engine A)."
        msg += "".join(f"\n\n{_err_text(n)}" for n in notices)
        return run_id, msg

    def upload(self, file: str | None, source: str) -> tuple[str | None, str]:
        if not file:
            return None, ""
        try:
            dest = self.store.save_upload(file, Path(file).name, source)
        except AvError as e:
            return None, _err_text(e)
        if source == "qisens":
            return str(dest), "Uploaded to the QiSens area. It is selected below."
        return str(dest), f"Uploaded to {source}/input/{dest.name}."

    # page ----------------------------------------------------------------------------

    def build(self):
        import gradio as gr
        poll_s = self.cfg["ui"]["poll_s"]
        variant0 = self.cfg["pose"]["variant"]
        css = (UI_DIR / "style.css").read_text()
        js = (UI_DIR / "app.js").read_text()
        keys = [k for k, _ in PAGES]

        # matplotlib and its font list load once here, not on the first Results view
        from .charts import _plt
        from .live_tab import build_live_tab
        _plt()
        # delete_cache: every hour, Gradio's temp files older than a day are removed
        with gr.Blocks(title="AssemblyVision", theme=_theme(gr), css=css, js=js, analytics_enabled=False,
                       delete_cache=(3600, 86400), fill_width=True) as demo:
            job = gr.State(None)                       # the run followed on New analysis
            job_timer = gr.Timer(poll_s, active=False)
            with gr.Row(elem_id="av-shell", equal_height=False):
                with gr.Column(elem_id="av-sidebar", scale=0, min_width=260):
                    gr.HTML("<div class='av-brand'><b>AssemblyVision</b><span>AI Based Motion Analysis and Process "
                            "Improvements in Train Assembly</span></div>")
                    navs = [gr.Button(title, elem_id=f"nav-{key}", size="md",
                                      elem_classes=["av-nav"] + (["av-active"] if key == "new" else []))
                            for key, title in PAGES]
                    with gr.Column(elem_classes=["av-side-foot"]):
                        server = gr.HTML(self.server_html())
                with gr.Column(elem_id="av-main", scale=1):
                    with gr.Tabs(elem_id="av-pages", selected="new") as pages:
                        # --- New analysis -------------------------------------------------
                        with gr.Tab("New analysis", id="new"), gr.Column(elem_classes=["av-page"]):
                            gr.HTML(v.page_header("New analysis", "Choose a video and a pose model, then start. "
                                                  "The analysis runs on the server, so this page may be closed."))
                            with gr.Column(elem_classes=["av-card"]):
                                gr.HTML(v.section_title("Video", "An mp4 or avi file from the data folders, "
                                                        "or a new upload."))
                                mode = gr.Radio(["Existing video", "Upload a video"], value="Existing video",
                                                show_label=False, container=False, elem_classes=["av-seg"])
                                video_dd = gr.Dropdown(label="Video", choices=self.video_choices(), value=None,
                                                       elem_classes=["av-plain"])
                                with gr.Column(visible=False) as up_col:
                                    up_src = gr.Radio(self.sources, value=self.sources[0], label="Save to",
                                                      visible=len(self.sources) > 1, elem_classes=["av-seg"])
                                    upload = gr.File(label="Video file (mp4 or avi)", file_types=[".mp4", ".avi"],
                                                     type="filepath", height=120)
                            with gr.Column(elem_classes=["av-card"]):
                                gr.HTML(v.section_title("Settings"))
                                variant = gr.Radio(list(VARIANTS), value=variant0, label="Pose model",
                                                   elem_classes=["av-seg"])
                                gr.HTML("<div class='av-hint' style='margin-top:-6px'>n, s, m, l: smaller is faster, "
                                        "larger is more accurate.</div>"
                                        "<div class='av-field-label' style='margin-top:18px'>Step engine</div>"
                                        "<div class='av-seg-static'><span class='on'>Engine A</span>"
                                        "<span>Engine B</span></div><div class='av-hint'>Engine B is not trained "
                                        "yet, so Engine A finds the steps.</div>")
                            with gr.Row(elem_classes=["av-actions"]):
                                start_btn = gr.Button("Start analysis", variant="primary", elem_classes=["av-fit"])
                                msg = gr.HTML()
                            with gr.Column(elem_classes=["av-card"]):
                                gr.HTML(v.section_title("Progress"))
                                status = gr.HTML(self.status_html(None)[0])
                            job_kpis = gr.HTML()
                            with gr.Row(elem_classes=["av-actions"]):
                                open_btn = gr.Button("Open results", variant="secondary", visible=False,
                                                     elem_classes=["av-fit"])
                            with gr.Accordion("Subject per shot", open=False, elem_classes=["av-accordion"]):
                                gr.HTML("<div class='av-hint' style='margin:0 0 12px 0'>S marks the worker the "
                                        "analysis follows in each shot.</div>")
                                gallery = gr.Gallery(show_label=False, columns=3, height="auto", object_fit="contain",
                                                     allow_preview=True, elem_classes=["av-gallery"],
                                                     show_share_button=False, show_download_button=False)

                        # --- Results ------------------------------------------------------
                        with gr.Tab("Results", id="results"), gr.Column(elem_classes=["av-page"]):
                            with gr.Row(elem_classes=["av-head-row"], equal_height=False):
                                res_head = gr.HTML(self.results_head(None))
                                with gr.Column(elem_classes=["av-picker-col"], scale=0, min_width=520):
                                    with gr.Row(elem_classes=["av-picker-row"], equal_height=True):
                                        run_dd = gr.Dropdown(label="Run", show_label=False, choices=[], value=None,
                                                             allow_custom_value=False, container=False,
                                                             min_width=420, elem_classes=["av-picker"])
                                        refresh = gr.Button("Refresh", variant="secondary",
                                                            elem_classes=["av-fit", "av-picker-btn"])
                                    run_id_box = gr.HTML(self.run_id_html(None))
                            res_kpis = gr.HTML()
                            note = gr.HTML()
                            with gr.Column(elem_classes=["av-card"]) as video_card:
                                gr.HTML(v.section_title("Overlay video", "Tracks and skeletons drawn on the video; "
                                                        "S is the worker the analysis follows."))
                                overlay = gr.Video(label="Overlay video", show_label=False, interactive=False,
                                                   elem_classes=["av-video"], show_download_button=True)
                            timeline_img = gr.HTML(label="Process timeline (Engine A)", elem_classes=["av-card"])
                            steps_html = gr.HTML(elem_classes=["av-card"])
                            with gr.Column(elem_classes=["av-card"]):
                                gr.HTML(v.section_title("Motion metrics per step", "Every value with its coverage, "
                                                        "the share of frames that had the keypoints it needs."))
                                with gr.Row():
                                    step_dd = gr.Dropdown(label="Step", choices=[ALL_STEPS], value=ALL_STEPS,
                                                          elem_classes=["av-plain", "av-field"], min_width=280,
                                                          scale=0)
                                metrics_html = gr.HTML()
                            bend_img = gr.HTML(elem_classes=["av-card"])
                            with gr.Column(elem_classes=["av-card"]):
                                gr.HTML(v.section_title("Hand reach heatmap", REACH_CAPTION))
                                with gr.Row():
                                    reach_dd = gr.Dropdown(label="Phase", choices=[ALL_PHASES], value=ALL_PHASES,
                                                           elem_classes=["av-plain", "av-field"], min_width=280,
                                                           scale=0)
                                reach_html = gr.HTML()
                            eval_html = gr.HTML(elem_classes=["av-card"])
                            notices_html = gr.HTML(elem_classes=["av-card"])

                        # --- Runs ---------------------------------------------------------
                        with gr.Tab("Runs", id="runs"), gr.Column(elem_classes=["av-page"]):
                            gr.HTML(v.page_header("Runs", "Every analysis on this server, newest first."))
                            runs_html = gr.HTML(elem_classes=["av-card"])
                            open_box = gr.Textbox(elem_id="av-open-run", elem_classes=["av-hidden"],
                                                  show_label=False, container=False)

                        # --- Live ---------------------------------------------------------
                        with gr.Tab("Live", id="live"), gr.Column(elem_classes=["av-page"]):
                            live_load, live_out = build_live_tab(self, gr, job, job_timer, server)
            server_timer = gr.Timer(5.0)

            def goto(key: str) -> list:
                return [gr.Tabs(selected=key)] + [gr.Button(elem_classes=["av-nav"] + (["av-active"] if k == key
                                                                                       else [])) for k in keys]
            nav_out = [pages, *navs]

            def results_out_values(run_id):
                r = self.results(run_id)
                return [r["head"], r["kpis"], r["note_html"], gr.Column(visible=bool(r["overlay"])), r["overlay"],
                        v.section_title("Process timeline", "When each phase happens in the video (Engine A).")
                        + (v.img_html(r["timeline"], "Process timeline") if r["timeline"] is not None
                           else v.note("No steps were stored for this run.")),
                        r["steps_html"], gr.update(choices=r["steps_choices"], value=ALL_STEPS), r["metrics_html"],
                        v.section_title("Back bending", "Seconds per trunk angle band and step: green B0, yellow B1, "
                                        "orange B2, red B3. The angle depends on the camera view.")
                        + (v.img_html(r["bending"], "Back bending per step") if r["bending"] is not None
                           else v.note("No back bending values for this run.")),
                        gr.update(choices=self.reach_choices(run_id) if run_id else [ALL_PHASES], value=ALL_PHASES),
                        self.reach_html(run_id),
                        r["eval_html"], r["notices_html"], self.run_id_html(run_id)]
            results_out = [res_head, res_kpis, note, video_card, overlay, timeline_img, steps_html, step_dd,
                           metrics_html, bend_img, reach_dd, reach_html, eval_html, notices_html, run_id_box]

            def fill_results(run_id):
                try:
                    return results_out_values(run_id)
                except AvError as e:
                    return [self.results_head(None), "", v.note(_err_text(e)), *[gr.skip()] * (len(results_out) - 3)]

            def job_values(run_id):
                html_, active = self.status_html(run_id)
                done = bool(run_id) and not active and self._finished(run_id)
                return [html_, self.job_kpis(run_id) if done else "", gr.Button(visible=done),
                        self.subject_gallery(run_id), gr.Timer(active=active)]
            job_out = [status, job_kpis, open_btn, gallery, job_timer]

            def on_load(request):
                q = dict(request.query_params) if request is not None else {}
                run_id = self.initial_run(q.get("run"))
                active = self.store.active_run()
                page = q.get("page") if q.get("page") in keys else ("results" if q.get("run") else "new")
                return [gr.update(choices=self.run_choices(), value=run_id), active, *job_values(active),
                        self.runs_html(run_id), *goto(page)]
            on_load.__annotations__["request"] = gr.Request     # a real type, so Gradio passes the request

            def on_poll(job_id, shown):
                self.runs.reap()
                vals = job_values(job_id)
                if vals[-1].active or not job_id:
                    return [*vals, gr.skip(), *[gr.skip()] * len(results_out)]
                # finished: fill the summary once, stop the timer (TC-APP-07), refresh Results if it shows this run
                same = shown == job_id
                return [*vals, gr.update(choices=self.run_choices(), value=shown),
                        *(fill_results(job_id) if same else [gr.skip()] * len(results_out))]

            def on_refresh(run_id):
                return gr.update(choices=self.run_choices(), value=run_id), \
                    gr.update(choices=self.video_choices()), self.runs_html(run_id)

            def on_start(video, var):
                run_id, text = self.start(video, var)
                if run_id is None:
                    return [gr.skip(), v.note(text), *[gr.skip()] * len(job_out)]
                return [run_id, f"<span class='av-hint'>{v.esc(text.split(chr(10))[0])}</span>", *job_values(run_id)]

            def on_mode(m):
                return gr.update(visible=m == "Existing video"), gr.Column(visible=m != "Existing video")

            def on_upload(file, src):
                dest, text = self.upload(file, src)
                if dest is None:
                    return gr.skip(), v.note(text), None, gr.skip(), gr.skip()
                return (gr.update(choices=self.video_choices(), value=dest, visible=True),
                        f"<span class='av-hint'>{v.esc(text)}</span>", None, "Existing video",
                        gr.Column(visible=False))

            def on_open(run_id):
                return [gr.update(choices=self.run_choices(), value=run_id), *goto("results")]

            def on_open_box(text):
                run_id = (text or "").split("|", 1)[0]
                if not run_id:
                    return [gr.skip()] * (1 + len(nav_out))
                return on_open(run_id)

            demo.load(on_load, None, [run_dd, job, *job_out, runs_html, *nav_out])
            demo.load(live_load, None, live_out)
            for key, btn in zip(keys, navs):
                if key == "runs":
                    btn.click(lambda r: [self.runs_html(r), *goto("runs")], run_dd, [runs_html, *nav_out],
                              show_progress="hidden")
                else:
                    btn.click(lambda k=key: goto(k), None, nav_out, show_progress="hidden")
            run_dd.change(fill_results, run_dd, results_out)
            job_timer.tick(on_poll, [job, run_dd], [*job_out, run_dd, *results_out], show_progress="hidden")
            server_shown = gr.State(self.server_html())

            def on_server(last):
                """The sidebar pill only when it changed, so it never redraws for nothing."""
                h = self.server_html()
                return (gr.skip(), last) if h == last else (h, h)
            server_timer.tick(on_server, server_shown, [server, server_shown], show_progress="hidden")
            refresh.click(on_refresh, run_dd, [run_dd, video_dd, runs_html])
            start_btn.click(on_start, [video_dd, variant], [job, msg, *job_out])
            mode.change(on_mode, mode, [video_dd, up_col])
            upload.upload(on_upload, [upload, up_src], [video_dd, msg, upload, mode, up_col])
            open_btn.click(on_open, job, [run_dd, *nav_out])
            open_box.input(on_open_box, open_box, [run_dd, *nav_out])
            step_dd.change(self.metrics_html, [run_dd, step_dd], metrics_html, show_progress="hidden")
            reach_dd.input(self.reach_html, [run_dd, reach_dd], reach_html, show_progress="hidden")
        self.demo = demo
        return demo

    def _finished(self, run_id: str) -> bool:
        try:
            return self.store.status(run_id).status in FINISHED
        except AvError:
            return False

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
