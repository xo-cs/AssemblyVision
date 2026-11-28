"""The Labels page (team accounts) and the Ratings page (mentor) of the measurement ground truth (Week 12, round 2).

Labels: frames (trunk band, both hands working, gaze), wrists (click left and right) and events (cycle starts, tool
pickups, object placements in one 2 minute clip). Ratings: the suggestions of a run, rated useful, partly useful or not
useful. Items and files: assemblyvision/validation_labels.py. Frames and clips are shown by this app behind the login
and saved nowhere outside the run's data area; the server refuses a save by the wrong kind of account, so hiding the
page in the sidebar is not the only guard.
"""
from __future__ import annotations

import numpy as np

from .. import validation_labels as L
from ..errors import AvError
from . import views as v
from .labels_tab import video_html

FLAGSHIP_HINT = "The QiSens flagship is the first run in the list."


def _user(request) -> str | None:
    return getattr(request, "username", None)


def _team(user: str | None) -> str:
    L.check_user(user)
    if L.is_mentor(user):
        raise ValueError("this labelling is for the team's accounts")
    return user


def _mentor(user: str | None) -> str:
    L.check_user(user)
    if not L.is_mentor(user):
        raise ValueError("this rating is for the mentor's account")
    return user


def marked(frame: np.ndarray | None, k: float, left, right) -> np.ndarray | None:
    """The frame with the clicked wrists drawn on it (L green, R orange); points in video pixels, k = video px per shown px."""
    if frame is None:
        return None
    import cv2
    img = frame.copy()
    for p, col, txt in ((left, (34, 197, 94), "L"), (right, (249, 115, 22), "R")):
        if p is not None:
            c = (int(round(p[0] / k)), int(round(p[1] / k)))
            cv2.circle(img, c, 7, col, 2)
            cv2.putText(img, txt, (c[0] + 9, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return img


def build_gt_tab(app, gr):
    gr.HTML(v.page_header("Labels", "Measurement ground truth. Label alone, under your own account, without looking at the "
                          "system's result. No screenshots or downloads of frames (docs_team/labelling_guide.md); clear "
                          "the browser cache after a QiSens session."))
    with gr.Column(elem_classes=["av-card"]):
        with gr.Row():
            task = gr.Dropdown([(t["title"], k) for k, t in L.GT_TASKS.items()], value="frames", label="Task", scale=2)
            run = gr.Dropdown([], label="Run", scale=2)
            open_b = gr.Button("Open", variant="primary", elem_classes=["av-fit"], scale=0)
        head = gr.HTML(f"<p class='av-hint'>{FLAGSHIP_HINT}</p>")
    with gr.Column(elem_classes=["av-card"], visible=False) as body:
        image = gr.Image(show_label=False, interactive=False, visible=False, type="numpy", format="jpeg")
        video = gr.HTML(visible=False)
        trunk = gr.Radio(L.TRUNK, label="Trunk band (angle of the trunk from upright)", visible=False)
        hands = gr.Radio(L.HANDS, label="Are both hands working?", visible=False)
        gaze = gr.Radio(L.GAZE2, label="Gaze direction (left and right as seen in the image)", visible=False)
        with gr.Row(visible=False) as wrist_row:
            no_left = gr.Button("Left wrist not visible", elem_classes=["av-fit"])
            no_right = gr.Button("Right wrist not visible", elem_classes=["av-fit"])
            clear = gr.Button("Clear the clicks", elem_classes=["av-fit"])
        with gr.Row(visible=False) as event_row:
            ev_btns = [gr.Button(k.capitalize(), variant="primary", elem_classes=["av-fit"]) for k in L.EVENT_KINDS]
            undo = gr.Button("Remove last mark", elem_classes=["av-fit"])
        with gr.Row() as nav_row:
            prev_b = gr.Button("Previous", elem_classes=["av-fit"])
            save_b = gr.Button("Save and next", variant="primary", elem_classes=["av-fit"])
            next_b = gr.Button("Next", elem_classes=["av-fit"])
        note = gr.HTML()
    state = gr.State(None)
    clicks = gr.State(None)                 # {"left": point or "none" or None, "right": ...} of the current wrist item
    t_box = gr.Number(visible=False)
    outs = [state, clicks, head, body, image, video, trunk, hands, gaze, wrist_row, event_row, nav_row, note]
    hide = lambda: gr.update(visible=False)

    def render(st, user, cl=None, msg=""):
        kind, i = st["task"], st["i"]
        path = L.gt_path(st["area"], st["run"], kind, user)
        done = L.read(path)
        if kind == "events":
            w = st["items"][0]
            marks = "; ".join(f"{float(r['time_s']):.1f} s {r['kind']}" for r in done) or "none yet"
            top = (f"<b>2 minute clip</b> from {v.esc(w['phase'])}, {w['time_s']:.0f} to {w['end_s']:.0f} s (chosen by a fixed "
                   f"rule: {v.esc(w['rule'])}).<br>Play the clip and press a button at the moment of each event: a "
                   f"<b>cycle start</b> when the same work cycle begins again, a <b>tool pickup</b> when a hand closes on a tool, "
                   f"an <b>object placement</b> when a hand lets go of a placed object.<br>{len(done)} marks: {v.esc(marks)}")
            return [st, None, top, gr.update(visible=True), hide(),
                    gr.update(value=video_html(app._file_url(st["video"]), w["time_s"], w["end_s"]), visible=True),
                    hide(), hide(), hide(), hide(), gr.update(visible=True), hide(), msg]
        it = st["items"][i]
        old = next((r for r in done if int(r["item"]) == it["item"]), {})
        top = (f"<b>Frame {i + 1} of {len(st['items'])}</b> &middot; {len(done)} saved &middot; {v.esc(it['phase'])} &middot; "
               f"items drawn with a fixed seed from frames with a tracked worker, evenly over the phases ({v.esc(st['src'])})")
        frame = L.read_frame(st["video"], it["frame"])
        k = st["width"] / frame.shape[1] if frame is not None else 1.0
        if kind == "frames":
            return [st, None, top, gr.update(visible=True), gr.update(value=frame, visible=True), hide(),
                    gr.update(value=old.get("trunk") or None, visible=True), gr.update(value=old.get("hands") or None, visible=True),
                    gr.update(value=old.get("gaze") or None, visible=True), hide(), hide(), gr.update(visible=True), msg]
        if cl is None:                                 # a saved item shows its saved clicks
            pt = lambda a, b: None if not old or old.get(a) in ("", None) else (float(old[a]), float(old[b]))
            cl = {"left": pt("left_x", "left_y") if old else None, "right": pt("right_x", "right_y") if old else None}
            if old:
                cl = {s: (cl[s] if cl[s] is not None else "none") for s in ("left", "right")}
        nxt = "left" if cl.get("left") is None else "right" if cl.get("right") is None else None
        say = {"left": "Click the worker's <b>LEFT</b> wrist (their own left).", "right": "Click the worker's <b>RIGHT</b> wrist.",
               None: "Both wrists are set: Save and next, or Clear the clicks."}[nxt]
        pts = {s: cl[s] if isinstance(cl.get(s), tuple) else None for s in ("left", "right")}
        return [st, cl, top + "<br>" + say, gr.update(visible=True), gr.update(value=marked(frame, k, pts["left"], pts["right"]), visible=True),
                hide(), hide(), hide(), hide(), gr.update(visible=True), hide(), gr.update(visible=True), msg]

    def closed(msg):
        return [None, None, f"<p class='av-quiet'>{v.esc(msg)}</p>", gr.update(visible=False), *[gr.skip()] * 8, ""]

    def on_open(task_key, run_id, request):
        try:
            user = _team(_user(request))
            if not run_id:
                return closed("Choose a run.")
            paths = app.store.paths(run_id)
            items, fps, src = L.gt_items(task_key, run_id, paths.root)
            if not items:
                return closed("This run has no frames with a tracked worker.")
            m = app.store.get_run(run_id)
            st = {"task": task_key, "i": 0, "items": items, "run": run_id, "area": str(paths.root.parent.parent),
                  "video": str(paths.cfr), "fps": fps, "src": src, "width": float(m["ingest"]["cfr"]["width"])}
            return render(st, user)
        except (ValueError, OSError, KeyError, AvError) as e:
            return closed(str(e))

    def move(st, d, request):
        if st is None or st["task"] == "events":
            return [gr.skip()] * len(outs)
        return render({**st, "i": max(0, min(len(st["items"]) - 1, st["i"] + d))}, _user(request))

    def on_prev(st, request):
        return move(st, -1, request)

    def on_next(st, request):
        return move(st, 1, request)

    def on_save(st, cl, a, b, c, request):
        if st is None:
            return [gr.skip()] * len(outs)
        try:
            user = _team(_user(request))
            path = L.gt_path(st["area"], st["run"], st["task"], user)
            it = st["items"][st["i"]]
            if st["task"] == "frames":
                if not (a and b and c):
                    return render(st, user, msg="<span class='av-note'>Answer all three questions first.</span>")
                L.save_frame(path, it, a, b, c, user)
            elif st["task"] == "wrists":
                if not cl or cl.get("left") is None or cl.get("right") is None:
                    return render(st, user, cl, "<span class='av-note'>Set both wrists first (click, or not visible).</span>")
                L.save_wrists(path, it, None if cl["left"] == "none" else cl["left"], None if cl["right"] == "none" else cl["right"], user)
            else:
                return [gr.skip()] * len(outs)
        except (ValueError, OSError) as e:
            return render(st, _user(request), cl, f"<span class='av-note'>{v.esc(str(e))}</span>")
        return render({**st, "i": min(len(st["items"]) - 1, st["i"] + 1)}, user)

    def on_click(st, cl, evt, request):
        if st is None or st["task"] != "wrists":
            return [gr.skip()] * len(outs)
        cl = dict(cl or {"left": None, "right": None})
        frame = L.read_frame(st["video"], st["items"][st["i"]]["frame"])
        k = st["width"] / frame.shape[1] if frame is not None else 1.0
        side = "left" if cl.get("left") is None else "right" if cl.get("right") is None else None
        if side:
            cl[side] = (evt.index[0] * k, evt.index[1] * k)
        return render(st, _user(request), cl)

    def not_visible(side):
        def f(st, cl, request):
            if st is None or st["task"] != "wrists":
                return [gr.skip()] * len(outs)
            return render(st, _user(request), {**(cl or {"left": None, "right": None}), side: "none"})
        f.__annotations__["request"] = gr.Request
        return f

    def on_clear(st, request):
        if st is None:
            return [gr.skip()] * len(outs)
        return render(st, _user(request), {"left": None, "right": None})

    def mark(kind):
        def f(st, t, request):
            if st is None or st["task"] != "events":
                return [gr.skip()] * len(outs)
            try:
                user = _team(_user(request))
                L.add_event(L.gt_path(st["area"], st["run"], "events", user), float(t), kind, st["items"][0], user)
            except (ValueError, TypeError) as e:
                return render(st, _user(request), msg=f"<span class='av-note'>{v.esc(str(e))}</span>")
            return render(st, user)
        f.__annotations__["request"] = gr.Request
        return f

    def on_undo(st, request):
        if st is None or st["task"] != "events":
            return [gr.skip()] * len(outs)
        user = _team(_user(request))
        L.remove_last_event(L.gt_path(st["area"], st["run"], "events", user))
        return render(st, user)

    for f in (on_open, on_prev, on_next, on_save, on_clear, on_undo):
        f.__annotations__["request"] = gr.Request
    on_click.__annotations__.update(evt=gr.SelectData, request=gr.Request)
    open_b.click(on_open, [task, run], outs)
    prev_b.click(on_prev, state, outs, show_progress="hidden")
    next_b.click(on_next, state, outs, show_progress="hidden")
    save_b.click(on_save, [state, clicks, trunk, hands, gaze], outs, show_progress="hidden")
    image.select(on_click, [state, clicks], outs, show_progress="hidden")
    no_left.click(not_visible("left"), [state, clicks], outs, show_progress="hidden")
    no_right.click(not_visible("right"), [state, clicks], outs, show_progress="hidden")
    clear.click(on_clear, state, outs, show_progress="hidden")
    js = "() => { const x = document.querySelector('.av-label-video video'); return x ? x.currentTime : -1; }"
    for b, k in zip(ev_btns, L.EVENT_KINDS):
        b.click(lambda t: t, t_box, t_box, js=js, show_progress="hidden").then(mark(k), [state, t_box], outs, show_progress="hidden")
    undo.click(on_undo, state, outs, show_progress="hidden")
    return run


def build_ratings_tab(app, gr):
    gr.HTML(v.page_header("Ratings", "Rate the improvement suggestions of a run. Each one is a rule based suggestion with its "
                          "evidence; please say whether it would be useful to a person improving this work."))
    with gr.Column(elem_classes=["av-card"]):
        with gr.Row():
            run = gr.Dropdown([], label="Run", scale=3)
            open_b = gr.Button("Open", variant="primary", elem_classes=["av-fit"], scale=0)
        head = gr.HTML(f"<p class='av-hint'>{FLAGSHIP_HINT}</p>")
    with gr.Column(elem_classes=["av-card"], visible=False) as body:
        card = gr.HTML()
        rating = gr.Radio(L.RATINGS, label="Your rating")
        comment = gr.Textbox(label="Comment (optional)", lines=2)
        with gr.Row():
            prev_b = gr.Button("Previous", elem_classes=["av-fit"])
            save_b = gr.Button("Save and next", variant="primary", elem_classes=["av-fit"])
            next_b = gr.Button("Next", elem_classes=["av-fit"])
        note = gr.HTML()
    state = gr.State(None)
    outs = [state, head, body, card, rating, comment, note]

    def render(st, user, msg=""):
        i, s = st["i"], st["items"][st["i"]]
        done = L.read(L.gt_path(st["area"], st["run"], "ratings", user))
        old = next((r for r in done if int(r["item"]) == i), {})
        ranges = ", ".join(f"{a:.0f} to {b:.0f} s" for a, b in s["time_ranges"])
        body_html = (f"<p><b>Suggestion {i + 1} of {len(st['items'])}</b> &middot; {len(done)} rated</p>"
                     f"<p><b>{v.esc(s['phase'])}</b>, cycle {s['cycle']}: {v.esc(s['suggestion'])}</p>"
                     f"<p class='av-hint'>Evidence: {v.esc(s['metric'])} = {s['value']} {v.esc(s['unit'])}; rule threshold "
                     f"{v.esc(s['threshold'])}; time ranges {v.esc(ranges)}. The suggestion is rule based and not validated.</p>")
        return [st, "", gr.update(visible=True), body_html, gr.update(value=old.get("rating") or None),
                gr.update(value=old.get("comment", "")), msg]

    def on_open(run_id, request):
        from ..output import extras as X
        try:
            user = _mentor(_user(request))
            if not run_id:
                raise ValueError("Choose a run.")
            root = app.store.paths(run_id).root
            items = app.memo("extras", run_id, lambda: X.compute(root))["suggestions"]
            if not items:
                raise ValueError("This run has no suggestions.")
            return render({"i": 0, "items": items, "run": run_id, "area": str(root.parent.parent)}, user)
        except (ValueError, OSError, KeyError, AvError) as e:
            return [None, f"<p class='av-quiet'>{v.esc(str(e))}</p>", gr.update(visible=False), "", gr.skip(), gr.skip(), ""]

    def move(st, d, request):
        if st is None:
            return [gr.skip()] * len(outs)
        return render({**st, "i": max(0, min(len(st["items"]) - 1, st["i"] + d))}, _user(request))

    def on_prev(st, request):
        return move(st, -1, request)

    def on_next(st, request):
        return move(st, 1, request)

    def on_save(st, r, c, request):
        if st is None:
            return [gr.skip()] * len(outs)
        try:
            user = _mentor(_user(request))
            L.save_rating(L.gt_path(st["area"], st["run"], "ratings", user), st["i"], st["items"][st["i"]], r, c, user)
        except (ValueError, OSError) as e:
            return render(st, _user(request), f"<span class='av-note'>{v.esc(str(e))}</span>")
        return render({**st, "i": min(len(st["items"]) - 1, st["i"] + 1)}, user)

    for f in (on_open, on_prev, on_next, on_save):
        f.__annotations__["request"] = gr.Request
    open_b.click(on_open, run, outs)
    prev_b.click(on_prev, state, outs, show_progress="hidden")
    next_b.click(on_next, state, outs, show_progress="hidden")
    save_b.click(on_save, [state, rating, comment], outs, show_progress="hidden")
    return run

