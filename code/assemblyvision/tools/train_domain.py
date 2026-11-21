"""The public train domain test (Week 12, rule experiments/w12-train-domain/rule.md).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.train_domain init     # the file templates (*.template) in the folder
    python -m assemblyvision.tools.train_domain check    # classes.csv, labels.csv, descriptions.md
    python -m assemblyvision.tools.train_domain run --gpu 0    # all clips and the joined video, then eval (GPU)
    python -m assemblyvision.tools.train_domain eval     # scores the stored runs again (CPU)

Public data only (data/public/videos/train_domain). One run per clip and one of the joined video with the
B zero shot setting, untuned, all classes as candidates. Clip test: the best class of a clip; a clip with
two labels gets two segments, no order given. Joined test: the clips in a seeded random order, segmented
without an order. The runs are hidden from the Results list. Results and the freeze record are written to
experiments/w12-train-domain; run refuses to start while a GPU lock is held or the method differs from the tag.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..storage.files import REPO_ROOT, utc_now

FOLDER = REPO_ROOT / "data/public/videos/train_domain"
OUT = REPO_ROOT / "experiments/w12-train-domain"
SEED = 20261121
JOIN_FPS = 30
TWO_MIN_S = 2.0                                  # clip with two labels: shortest segment
JOIN_MIN_S = 5.0                                 # joined video: shortest segment (the shortest clip is 7.4 s)


def videos(folder: Path = FOLDER) -> list[str]:
    return sorted(p.name for p in folder.glob("*.mp4"))


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]


def read_files(folder: Path = FOLDER) -> tuple[list[str], dict[str, dict]]:
    """(class names in file order, {clip: {"first", "second" or None, "change_s" or None}}); checks all three files."""
    from ..analysis import phase_text as pt
    for f in ("classes.csv", "labels.csv", "descriptions.md"):
        if not (folder / f).is_file():
            raise ValueError(f"{f} is not in {folder}")
    classes = [r["class"] for r in read_csv(folder / "classes.csv")]
    if len(classes) < 2 or len(set(classes)) != len(classes) or not all(classes):
        raise ValueError("classes.csv needs at least two different class names")
    parsed = pt.parse(folder / "descriptions.md").get(None, [])
    if [p["name"] for p in parsed] != classes:
        raise ValueError("the ## headings of descriptions.md must be the classes of classes.csv, in the same order")
    for p in parsed:
        if not p["short"] or not p["descriptions"]:
            raise ValueError(f"descriptions.md, {p['name']}: needs \"> \" lines and \"- \" lines")
    labels = {}
    for r in read_csv(folder / "labels.csv"):
        a, b, c = r.get("class", ""), r.get("second_class", ""), r.get("change_s", "")
        if a not in classes or (b and b not in classes):
            raise ValueError(f"labels.csv, {r.get('file')}: a class that is not in classes.csv")
        if bool(b) != bool(c) or (b and b == a):
            raise ValueError(f"labels.csv, {r.get('file')}: a second class needs a change time and must differ")
        labels[r["file"]] = {"first": a, "second": b or None, "change_s": float(c) if c else None}
    want = videos(folder)
    if sorted(labels) != want:
        raise ValueError(f"labels.csv must have one row per clip ({len(want)}); missing "
                         f"{sorted(set(want) - set(labels))}, unknown {sorted(set(labels) - set(want))}")
    return classes, labels


def init(folder: Path = FOLDER) -> list[str]:
    """Templates next to the clips; a real file is never overwritten."""
    t = {"classes.csv.template": "class,definition\nExample class,one sentence that says what is seen in this class\n",
         "labels.csv.template": "file,class,second_class,change_s\n" + "".join(f"{v},,,\n" for v in videos(folder)),
         "descriptions.md.template": "# Train domain class descriptions (written from the class table only, no video seen)\n\n"
         "## Example class\n\nDistinguishes: what separates this class from the others\n\n"
         "> first short line for the VLM prompt\n> second short line\n> third short line\n\n"
         "- one visual description\n- another visual description (20 per class, as in docs_team/phase_descriptions.md)\n"}
    for k, v in t.items():
        (folder / k).write_text(v)
    return sorted(t)


def run_toml(classes: list[str], folder: Path = FOLDER) -> str:
    rel = (folder / "descriptions.md").relative_to(REPO_ROOT).as_posix()
    steps = ",\n".join(f"    {{ name = {json.dumps(c)} }}" for c in classes)
    return f'[zero_shot]\nphases = "file:{rel}"\n\n[steps]\nuse = true\nstep = [\n{steps},\n]\n'


def join_order(files: list[str], seed: int = SEED) -> list[str]:
    out = sorted(files)
    random.Random(seed).shuffle(out)
    return out


def join_clips(order: list[str], folder: Path, out_dir: Path) -> tuple[Path, list[float]]:
    """All clips at 1920x1080 and JOIN_FPS, in this order; (joined video, length of each part in seconds)."""
    parts = out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    vf = (f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
          f"fps={JOIN_FPS},format=yuv420p")
    names, lens = [], []
    for i, v in enumerate(order):
        p = parts / f"{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(folder / v), "-an", "-vf", vf, "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "18", str(p)], check=True)
        n = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v", "-show_entries",
                            "stream=nb_read_packets", "-of", "csv=p=0", str(p)], capture_output=True, text=True, check=True)
        names.append(p)
        lens.append(int(n.stdout.strip().rstrip(",")) / JOIN_FPS)
    lst = parts / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in names))
    joined = out_dir / "joined.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)],
                   check=True)
    return joined, lens


def joined_truth(order: list[str], lens: list[float], labels: dict[str, dict]) -> list[tuple[str, float, float]]:
    """[(class, start s, end s)] of the joined video from the clip lengths; neighbours of one class are merged."""
    out: list[list] = []
    t = 0.0
    for v, n in zip(order, lens):
        lab = labels[v]
        pieces = [(lab["first"], t, t + n)]
        if lab["second"]:
            c = t + min(lab["change_s"], n)
            pieces = [(lab["first"], t, c), (lab["second"], c, t + n)]
        for name, a, b in pieces:
            if out and out[-1][0] == name:
                out[-1][2] = b
            else:
                out.append([name, a, b])
        t += n
    return [(a, round(b, 3), round(c, 3)) for a, b, c in out]


def score_items(items: list[tuple[str, str]], classes: list[str]) -> dict:
    """Accuracy, per class precision, recall and F1, and the confusion matrix (rows true, columns predicted)."""
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in items:
        cm[idx[t], idx[p]] += 1
    per = {}
    for c, i in idx.items():
        tp, fp, fn = int(cm[i, i]), int(cm[:, i].sum() - cm[i, i]), int(cm[i].sum() - cm[i, i])
        pr = tp / (tp + fp) if tp + fp else None
        rc = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        per[c] = {"support": int(cm[i].sum()), "precision": pr, "recall": rc, "f1": f1}
    return {"items": len(items), "accuracy": round(100 * float(np.trace(cm)) / max(len(items), 1), 2),
            "per_class": per, "confusion": cm.tolist(), "classes": classes}


def clip_items(ll: np.ndarray, fps: float, lab: dict, classes: list[str]) -> tuple[list[tuple[str, str]], dict]:
    """(true, predicted) per labelled segment of a clip. With two labels, the prediction of a true segment is
    the class that covers most of its frames in the two predicted segments."""
    from ..analysis import clip_classes as C
    if not lab["second"]:
        k, lead = C.best_class(ll)
        return [(lab["first"], classes[k])], {"pred": classes[k], "lead": round(lead, 4)}
    a, b, t = C.two_segments(ll, fps, TWO_MIN_S)
    N = ll.shape[1] - 1
    pred = np.r_[np.full(t, a), np.full(N - t, b)]
    c = min(N - 1, max(1, int(round(lab["change_s"] * fps))))
    out = []
    for name, x in ((lab["first"], pred[:c]), (lab["second"], pred[c:])):
        out.append((name, classes[int(np.bincount(x, minlength=len(classes)).argmax())]))
    return out, {"pred": [classes[a], classes[b]], "change_s": round(t / fps, 3)}


def likelihood(root: Path, n_classes: int) -> tuple[np.ndarray, float]:
    """The B zero shot class log likelihood of a stored run (the ordered decoder's own function)."""
    from ..analysis.segmentation import build_timeline, create_engine
    from ..analysis.siglip import run_phase_scores
    from ..config import Config
    from .run_pose import _MemoryCtx, stored_config
    manifest = json.loads((root / "manifest.json").read_text())
    cfg = Config.load(overrides=stored_config(manifest))
    engine = create_engine("A", cfg, _MemoryCtx("public"))
    tl = build_timeline(manifest, root, engine.cleaner, None)
    tl.phase_scores, missing = run_phase_scores(root, tl.n_frames, tl.fps, cfg["seg_a"], cfg["zero_shot"]["vlm_file"],
                                                cfg["zero_shot"]["siglip_file"])
    ll, rep = engine.phase_likelihood(tl, n_classes)
    if missing or ll is None or rep.get("skipped"):
        raise ValueError(f"{root.name}: scores not usable (missing {missing}, {rep.get('skipped')})")
    return ll, tl.fps


def joined_scores(ll: np.ndarray, fps: float, truth: list, classes: list[str], duration: float, cfg) -> dict:
    from ..analysis import clip_classes as C
    from ..evaluation.steps_eval import evaluate
    from ..types import Step
    pred = [Step(classes[k], a / fps, b / fps, "engine_b", None) for k, a, b in C.unordered(ll, fps, JOIN_MIN_S)]
    gt = [Step(c, a, b, "gt", None) for c, a, b in truth]
    res, rows = evaluate(pred, gt, duration, cfg["eval"]["seg_fps"], cfg["eval"]["overlaps"])
    return {"metrics": res, "boundaries": rows, "pred_segments": [(s.label, round(s.start_s, 2), round(s.end_s, 2)) for s in pred],
            "gt_segments": truth}


def _locks_free(gpus: list[int]) -> list[int]:
    busy = []
    for g in gpus:
        with open(REPO_ROOT / ".locks" / f"gpu{g}.lock", "a+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except OSError:
                busy.append(g)
    return busy


def run_one(video: Path, layer: Path, gpu: int, label: str) -> str:
    r = subprocess.run([sys.executable, "-m", "assemblyvision.tools.run_pose", "--source", "public", "--video", str(video),
                        "--variant", "x", "--gpu", str(gpu), "--no-overlay", "--config", str(layer),
                        "--set", 'seg_a.engine="B zero shot"'], capture_output=True, text=True, cwd=REPO_ROOT / "code")
    m = re.search(r"(run_\d{8}_\d{6}_[0-9a-f]{4}) done", r.stdout)
    if r.returncode != 0 or not m:
        raise RuntimeError(f"{video.name}: the run failed: {(r.stderr or r.stdout).strip()[-400:]}")
    root = REPO_ROOT / "data/public/runs" / m.group(1)
    if not (root / "siglip.npz").is_file() or not (root / "vlm.npz").is_file():
        raise RuntimeError(f"{video.name}: {root.name} has no zero shot scores (was GPU1 busy?)")
    man = json.loads((root / "manifest.json").read_text())
    man["milestone"] = {"label": label, "hidden": True, "copied_from": None}
    (root / "manifest.json").write_text(json.dumps(man, indent=2))
    return m.group(1)


def evaluate_all(folder: Path = FOLDER) -> dict:
    from ..config import Config
    from . import freeze
    classes, labels = read_files(folder)
    state = json.loads((folder / "generated" / "runs.json").read_text())
    items, per = [], {}
    for v in videos(folder):
        ll, fps = likelihood(REPO_ROOT / "data/public/runs" / state["clips"][v], len(classes))
        it, info = clip_items(ll, fps, labels[v], classes)
        items += it
        per[v] = {"run": state["clips"][v], "true": [t for t, _ in it], **info, "correct": all(t == p for t, p in it)}
    root = REPO_ROOT / "data/public/runs" / state["joined"]
    ll, fps = likelihood(root, len(classes))
    truth = joined_truth(state["order"], state["lengths"], labels)
    cfg = Config.load()
    duration = float(json.loads((root / "manifest.json").read_text())["ingest"]["cfr"]["duration_s"])
    return {"created_utc": utc_now(), "freeze": freeze.record(), "setting": "B zero shot, untuned", "seed": SEED,
            "min_segment_s": {"two_labels": TWO_MIN_S, "joined": JOIN_MIN_S}, "clip_test": score_items(items, classes),
            "clips": per, "joined_test": joined_scores(ll, fps, truth, classes, duration, cfg), "order": state["order"]}


def write_results(res: dict) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    c, j = res["clip_test"], res["joined_test"]["metrics"]
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    L = ["# w12-train-domain", "", f"Freeze: {res['freeze']['tag']} ({res['freeze']['tag_commit'][:7]}), ran on "
         f"{res['freeze']['ran_on'][:7]}, method unchanged: {res['freeze']['method_unchanged']}. {res['setting']}.", "",
         f"**Clip classification:** accuracy {c['accuracy']:.2f} % over {c['items']} labelled segments.", "",
         "| class | support | precision | recall | F1 |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {k} | {v['support']} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['f1'])} |" for k, v in c["per_class"].items()]
    L += ["", "Confusion matrix (rows true, columns predicted):", "", "| | " + " | ".join(c["classes"]) + " |",
          "| --- |" + " --- |" * len(c["classes"])]
    L += [f"| {n} | " + " | ".join(map(str, r)) + " |" for n, r in zip(c["classes"], c["confusion"])]
    L += ["", f"**Joined video (seed {res['seed']}):** frame accuracy {j['frame_accuracy']:.2f}, edit {j['edit']:.2f}, "
          f"F1@10 {j['f1_10']:.2f}, F1@25 {j['f1_25']:.2f}, F1@50 {j['f1_50']:.2f}, boundary MAE "
          + ("n/a" if j["boundary_mae_s"] is None else f"{j['boundary_mae_s']:.2f} s") + "."]
    (OUT / "table.md").write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.train_domain")
    p.add_argument("command", choices=("init", "check", "run", "eval"))
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args(argv)
    if a.command == "init":
        print("written:", ", ".join(init()))
        return 0
    try:
        classes, labels = read_files()
    except ValueError as e:
        print(f"train_domain: {e}", file=sys.stderr)
        return 2
    print(f"files ok: {len(classes)} classes, {len(labels)} clips, {sum(bool(x['second']) for x in labels.values())} with two labels")
    if a.command == "check":
        return 0
    if a.command == "run":
        from ..config import Config
        from . import freeze
        rec = freeze.record()
        if not rec["method_unchanged"]:
            print(f"refused: the method differs from {rec['tag']}: {rec['changed']}", file=sys.stderr)
            return 2
        busy = _locks_free([a.gpu, Config.load()["zero_shot"]["vlm_gpu"]])
        if busy:
            print(f"refused: GPU lock held: {busy}", file=sys.stderr)
            return 2
        gen = FOLDER / "generated"
        gen.mkdir(exist_ok=True)
        layer = gen / "run.toml"
        layer.write_text(run_toml(classes))
        path = gen / "runs.json"
        state = json.loads(path.read_text()) if path.is_file() else {"clips": {}}
        for i, v in enumerate(videos()):
            if v not in state["clips"]:
                state["clips"][v] = run_one(FOLDER / v, layer, a.gpu, f"train domain clip {v}")
                path.write_text(json.dumps(state, indent=1))
            print(f"{i + 1}/{len(labels)} {v}: {state['clips'][v]}", flush=True)
        if "joined" not in state:
            state["order"] = join_order(videos())
            joined, state["lengths"] = join_clips(state["order"], FOLDER, gen)
            state["joined"] = run_one(joined, layer, a.gpu, "train domain joined video")
            path.write_text(json.dumps(state, indent=1))
        print(f"joined: {state['joined']}", flush=True)
    res = evaluate_all()
    write_results(res)
    j = res["joined_test"]["metrics"]
    print(f"clip accuracy {res['clip_test']['accuracy']:.2f} % ({res['clip_test']['items']} segments); joined frame accuracy "
          f"{j['frame_accuracy']:.2f}, edit {j['edit']:.2f}; experiments/w12-train-domain/table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
