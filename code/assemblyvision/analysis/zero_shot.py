"""Zero shot phase scoring of one run (Week 11): the siglip_scores and vlm_scores stages.

Used by tools/siglip_scores.py, tools/vlm_scores.py and the full pipeline run
(tools/run_pose.py). The phase set comes from [zero_shot] phases: "qisens"
(docs_team/phase_descriptions.md) or "ikea_asm:<furniture>". Both stages write only
scores into the run folder (siglip.npz, vlm.npz); a QiSens run's files stay under
data/qisens. In a full run the VLM runs in a thread on its own GPU (zero_shot.vlm_gpu,
under .locks/gpu<N>.lock) during the pose pass; SigLIP runs on the pose GPU after
tracking, because its crop view needs the subject's boxes.
"""
from __future__ import annotations

import fcntl
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..storage.files import REPO_ROOT, utc_now

PROCESS = {"qisens": "a video of a worker maintaining a railway vehicle (bogie maintenance) in a workshop",
           "ikea": "a top view video of a person assembling an IKEA {name}"}
IKEA_NAMES = {"Kallax_Shelf_Drawer": "Kallax shelf drawer", "Lack_TV_Bench": "Lack TV bench",
              "Lack_Coffee_Table": "Lack coffee table", "Lack_Side_Table": "Lack side table"}


def phase_set(name: str) -> tuple[list[dict], str, str]:
    """(phases, process text of the VLM prompt, label) of "qisens" or "ikea_asm:<furniture>"."""
    from . import phase_text as pt
    if name == "qisens":
        return pt.phases("qisens"), PROCESS["qisens"], "qisens"
    ds, _, furn = name.partition(":")
    if ds != "ikea_asm" or furn not in IKEA_NAMES:
        raise ValueError(f"zero_shot.phases {name!r}: \"qisens\" or \"ikea_asm:<furniture>\"")
    return pt.phases("ikea_asm", furn), PROCESS["ikea"].format(name=IKEA_NAMES[furn]), furn


def siglip_run(root: Path, manifest: dict, phases: list[dict], label: str, zs: dict, device: str,
               loaded: tuple | None = None) -> dict:
    """Scores the run's CFR video with SigLIP 2 and writes root/siglip.npz. zs: the
    [zero_shot] settings; loaded: (model, processor) to reuse. Returns counts."""
    from . import siglip as S
    half = device == "cuda"
    model, proc = loaded if loaded is not None else S.load(REPO_ROOT / "models", device, half=half)
    bank = S.text_bank(model, proc, phases, device)
    cfr = manifest["ingest"]["cfr"]
    res = S.score_video(model, proc, bank, root / cfr["path"], S.subject_boxes(manifest, root), float(cfr["fps"]),
                        zs["siglip_fps"], zs["siglip_max_px"], zs["siglip_pad"], device=device, half=half)
    S.save(root / "siglip.npz", res, {"model": S.MODEL_DIR, "revision": S.REVISION, "fps": zs["siglip_fps"],
                                      "max_px": zs["siglip_max_px"], "pad": zs["siglip_pad"],
                                      "phases": [ph["name"] for ph in phases], "descriptions_sha": S.text_sha(phases),
                                      "set": label, "date_utc": utc_now()})
    return {"samples": len(res["time_s"]), "crops": int(res["crop_ok"].sum())}


def vlm_score_video(video: Path, out: Path, phases: list[dict], process: str, zs: dict, port: int,
                    stop: threading.Event | None = None) -> dict:
    """Every zs vlm_window_s seconds a clip of vlm_frames frames goes to the running
    llama-server on port (vlm_slots requests at a time); writes out (vlm.npz or another
    name). Returns counts."""
    from . import vlm as V
    names = [ph["name"] for ph in phases]
    text = V.prompt(phases, process, zs["vlm_window_s"])
    wins = V.windows(video, zs["vlm_window_s"], zs["vlm_frames"], 1280)

    def one(w):
        if stop is not None and stop.is_set():
            return np.full(len(names), 1.0 / len(names), np.float32), "stopped", False
        try:
            return V.parse(V.ask(port, text, w[1], zs["vlm_px"]), names)
        except Exception as e:                 # a failed request counts as unparsed
            return np.full(len(names), 1.0 / len(names), np.float32), f"request failed: {e}"[:200], False
    times, res = [], []
    with ThreadPoolExecutor(zs["vlm_slots"]) as ex:
        pending = []
        for w in wins:                         # bounded queue: the frames of at most 4 x slots windows in memory
            times.append(w[0])
            pending.append(ex.submit(one, w))
            if len(pending) >= 4 * zs["vlm_slots"]:
                res.append(pending.pop(0).result())
        res += [f.result() for f in pending]
    probs = np.stack([r[0] for r in res]) if res else np.zeros((0, len(names)), np.float32)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, time_s=np.array(times, np.float32), probs=probs,
                        reasons=np.array([r[1] for r in res]), parsed=np.array([r[2] for r in res]),
                        meta=json.dumps({"model": V.MODEL, "mmproj": V.MMPROJ, "llama_cpp": "b12005",
                                         "window_s": zs["vlm_window_s"], "n_frames": zs["vlm_frames"],
                                         "px": zs["vlm_px"], "slots": zs["vlm_slots"], "phases": names,
                                         "prompt": text, "date_utc": utc_now()}))
    tmp.replace(out)
    return {"windows": len(res), "unparsed": sum(not r[2] for r in res)}


def try_lock(gpu: int):
    """The open lock file of .locks/gpu<N>.lock, or None when another job holds it."""
    d = REPO_ROOT / ".locks"
    d.mkdir(exist_ok=True)
    f = open(d / f"gpu{gpu}.lock", "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


class VlmJob:
    """The vlm_scores stage of a full run, in a thread: takes the VLM GPU's lock without
    waiting (busy -> error "busy"), starts llama-server with its log in log_path (the run
    folder), scores the video into root/vlm.npz and stops the server. result() gives
    {"status": "done"|"busy"|"failed", "seconds", ...}."""

    def __init__(self, root: Path, video: Path, phases: list[dict], process: str, zs: dict, log_path: Path):
        self.root, self.video, self.phases, self.process, self.zs, self.log_path = \
            root, video, phases, process, zs, log_path
        self.stop = threading.Event()
        self._out: dict = {}
        self._thread = threading.Thread(target=self._run, name="vlm_scores", daemon=True)

    def start(self) -> "VlmJob":
        self._thread.start()
        return self

    def _run(self) -> None:
        from . import vlm as V
        t0 = time.monotonic()
        lock = try_lock(self.zs["vlm_gpu"])
        if lock is None:
            self._out = {"status": "busy", "seconds": 0.0, "detail": f"GPU {self.zs['vlm_gpu']} is in use"}
            return
        server = None
        try:
            server = V.start_server(self.zs["vlm_gpu"], self.zs["vlm_port"], self.log_path,
                                    ctx=V.context_size(self.zs), parallel=self.zs["vlm_slots"])
            ready = time.monotonic() - t0
            rec = vlm_score_video(self.video, self.root / "vlm.npz", self.phases, self.process, self.zs,
                                  self.zs["vlm_port"], self.stop)
            self._out = {"status": "done", "server_ready_s": round(ready, 1), **rec}
        except Exception as e:
            self._out = {"status": "failed", "detail": f"{type(e).__name__}: {e}"[:300]}
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=60)
                except Exception:
                    server.kill()
            lock.close()
            self._out["seconds"] = round(time.monotonic() - t0, 2)

    def busy(self) -> bool:
        """True once the job has found the VLM GPU in use (it does so at once)."""
        self._thread.join(timeout=5.0)
        return self._out.get("status") == "busy"

    def result(self) -> dict:
        self._thread.join()
        return self._out
