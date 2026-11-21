"""The Labels page (Week 12): hand labels for validation, for logged in accounts only.

A task is either the class of each public train domain clip, or a small task on a stored run (assemblyvision/validation_labels.py).
The frames and videos are served by this app behind the login; labels are saved per annotator in the data area.
"""
from __future__ import annotations

import html
from pathlib import Path

from .. import validation_labels as L
from ..errors import AvError
from ..tools import train_domain as TD
from . import views as v


def _name(request) -> str | None:
    return getattr(request, "username", None)


def clip_classes() -> list[str]:
    p = TD.FOLDER / "classes.csv"
    return [r["class"] for r in TD.read_csv(p) if r.get("class")] if p.is_file() else []


def clip_seconds(file: str) -> float:
    import cv2
    cap = cv2.VideoCapture(str(TD.FOLDER / file))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        return cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0.0
    finally:
        cap.release()


def video_html(url: str | None, a: float | None = None, b: float | None = None) -> str:
    if not url:
        return "<p class='av-quiet'>The video is not available.</p>"
    frag = f"#t={a:.2f},{b:.2f}" if a is not None else ""
    return (f"<div class='av-label-video'><video controls preload='metadata' style='width:100%;max-height:480px' "
            f"src='{html.escape(url + frag, quote=True)}'></video></div>")


def build_labels_tab(app, gr):
    gr.HTML(v.page_header("Labels", "Hand labels for validation. Each person labels alone, under their own account; "
                          "do not look at the system's result for the item first. No screenshots or downloads of frames "
                          "(docs_team/labelling_guide.md, section 1); clear the browser cache after a QiSens session."))
    choices = [("Train domain clips: class of each clip", L.CLIPS)] + [(t["title"], k) for k, t in L.TASKS.items()]
    with gr.Column(elem_classes=["av-card"]):
        with gr.Row():
            task = gr.Dropdown(choices, value=L.CLIPS, label="Task", scale=2)
            run = gr.Dropdown([], label="Run", visible=False, scale=2)
            open_b = gr.Button("Open", variant="primary", elem_classes=["av-fit"], scale=0)
        head = gr.HTML()
    with gr.Column(elem_classes=["av-card"], visible=False) as body:
        image = gr.Image(show_label=False, interactive=False, visible=False, type="numpy", format="jpeg")
        video = gr.HTML(visible=False)
        answer = gr.Radio([], label="Answer", visible=False)
        with gr.Row(visible=False) as second_row:
            second = gr.Dropdown([], label="Second class (only if the clip changes activity)", allow_custom_value=False)
            change = gr.Number(label="Change time (s)", precision=2)
        with gr.Row():
            prev_b = gr.Button("Previous", elem_classes=["av-fit"])
            save_b = gr.Button("Save and next", variant="primary", elem_classes=["av-fit"])
            next_b = gr.Button("Next", elem_classes=["av-fit"])
            mark_b = gr.Button("Mark", variant="primary", visible=False, elem_classes=["av-fit"])
            undo_b = gr.Button("Remove last mark", visible=False, elem_classes=["av-fit"])
        note = gr.HTML()
    state = gr.State(None)
    mark_t = gr.Number(visible=False)
    outs = [state, head, body, image, video, answer, second_row, second, change, prev_b, save_b, next_b, mark_b, undo_b, note]

    def path_of(st, user):
        return L.clip_path(TD.FOLDER, user) if st["task"] == L.CLIPS else L.label_path(st["area"], st["run"], st["task"], user)

    def render(st, user, msg=""):
        i = st["i"]
        done = L.read(path_of(st, user))
        task_key = st["task"]
        hide = gr.update(visible=False)
        if task_key == L.CLIPS:
            f = st["items"][i]
            old = next((r for r in done if r["file"] == f), {})
            top = f"<b>Clip {i + 1} of {len(st['items'])}</b> &middot; {len(done)} saved &middot; {v.esc(f)}"
            return [st, top, gr.update(visible=True), hide,
                    gr.update(value=video_html(app._file_url(TD.FOLDER / f)), visible=True),
                    gr.update(choices=st["classes"], value=old.get("class") or None, visible=True, label="Class of this clip"),
                    gr.update(visible=True), gr.update(choices=[""] + st["classes"], value=old.get("second_class", "")),
                    gr.update(value=float(old["change_s"]) if old.get("change_s") else None),
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), hide, hide, msg]
        t = L.TASKS[task_key]
        it = st["items"][i]
        if t["kind"] == "event":
            marks = ", ".join(f"{float(r['time_s']):.1f}" for r in done) or "none yet"
            top = (f"<b>{v.esc(t['title'])}</b><br>{v.esc(t['ask'])}<br>Window {it['time_s']:.1f} to {it['end_s']:.1f} s "
                   f"&middot; {len(done)} marks: {marks}")
            return [st, top, gr.update(visible=True), hide,
                    gr.update(value=video_html(app._file_url(st["video"]), it["time_s"], it["end_s"]), visible=True),
                    hide, hide, gr.skip(), gr.skip(), hide, hide, hide, gr.update(visible=True), gr.update(visible=True), msg]
        old = next((r for r in done if int(r["item"]) == it["item"]), {})
        top = f"<b>Item {i + 1} of {len(st['items'])}</b> &middot; {len(done)} saved<br>{v.esc(t['ask'])}"
        if t["kind"] == "clip":
            media = [hide, gr.update(value=video_html(app._file_url(st["video"]), it["time_s"], it["end_s"]), visible=True)]
        else:
            media = [gr.update(value=L.read_frame(st["video"], it["frame"]), visible=True), hide]
        radio = gr.update(choices=t["options"], value=old.get("value") or None, visible=True, label="Answer")
        return [st, top, gr.update(visible=True), *media, radio, hide, gr.skip(), gr.skip(), gr.update(visible=True),
                gr.update(visible=True), gr.update(visible=True), hide, hide, msg]

    def closed(msg):
        return [None, f"<p class='av-quiet'>{v.esc(msg)}</p>", gr.update(visible=False), *[gr.skip()] * 11, ""]

    def on_task(k):
        return gr.update(visible=k != L.CLIPS, choices=app.run_choices() if k != L.CLIPS else [])

    def on_open(k, run_id, request):
        user = _name(request)
        try:
            L.check_user(user)
            if k == L.CLIPS:
                classes = clip_classes()
                if not classes:
                    return closed("classes.csv is not in the train domain folder yet.")
                items = TD.videos()
                if not items:
                    return closed("There are no clips in the train domain folder.")
                return render({"task": k, "i": 0, "items": items, "classes": classes}, user)
            if not run_id:
                return closed("Choose a run.")
            paths = app.store.paths(run_id)
            items, fps = L.run_items(k, run_id, paths.root)
            if not items:
                return closed("This run has no frames with a tracked subject.")
            st = {"task": k, "i": 0, "items": items, "run": run_id, "area": str(paths.root.parent.parent),
                  "video": str(paths.cfr), "fps": fps}
            return render(st, user)
        except (ValueError, OSError, KeyError, AvError) as e:
            return closed(str(e))

    def step(st, d, request):
        if st is None:
            return [gr.skip()] * len(outs)
        st = {**st, "i": max(0, min(len(st["items"]) - 1, st["i"] + d))}
        return render(st, _name(request))

    def on_prev(st, request):
        return step(st, -1, request)

    def on_next(st, request):
        return step(st, 1, request)

    def on_save(st, a, s, c, request):
        if st is None:
            return [gr.skip()] * len(outs)
        user = _name(request)
        try:
            if not a:
                return render(st, user, "<span class='av-note'>Choose an answer first.</span>")
            if st["task"] == L.CLIPS:
                f = st["items"][st["i"]]
                L.save_clip(path_of(st, user), f, st["classes"], a, s or None, c, clip_seconds(f), user)
            else:
                L.save_item(path_of(st, user), st["items"][st["i"]], a, user)
        except (ValueError, OSError) as e:
            return render(st, user, f"<span class='av-note'>{v.esc(str(e))}</span>")
        return render({**st, "i": min(len(st["items"]) - 1, st["i"] + 1)}, user)

    def on_mark(st, t, request):
        if st is None:
            return [gr.skip()] * len(outs)
        user = _name(request)
        try:
            L.add_mark(path_of(st, user), float(t), st["items"][0], st["fps"], user)
        except (ValueError, TypeError) as e:
            return render(st, user, f"<span class='av-note'>{v.esc(str(e))}</span>")
        return render(st, user)

    def on_undo(st, request):
        if st is None:
            return [gr.skip()] * len(outs)
        L.remove_last_mark(path_of(st, _name(request)))
        return render(st, _name(request))

    for f in (on_open, on_prev, on_next, on_save, on_mark, on_undo):
        f.__annotations__["request"] = gr.Request
    task.change(on_task, task, run, show_progress="hidden")
    open_b.click(on_open, [task, run], outs)
    prev_b.click(on_prev, state, outs, show_progress="hidden")
    next_b.click(on_next, state, outs, show_progress="hidden")
    save_b.click(on_save, [state, answer, second, change], outs, show_progress="hidden")
    mark_b.click(lambda t: t, mark_t, mark_t, js="() => { const x = document.querySelector('.av-label-video video'); "
                 "return x ? x.currentTime : -1; }", show_progress="hidden").then(on_mark, [state, mark_t], outs,
                                                                                 show_progress="hidden")
    undo_b.click(on_undo, state, outs, show_progress="hidden")
