"""Zero shot phase scores with Qwen3 VL 8B (Week 11): the vlm_scores stage.

A local llama-server (third_party/llama.cpp, tag b12005, built inside ~/AssemblyVision)
serves models/Qwen3VL-8B-Instruct-Q4_K_M.gguf with models/mmproj-Qwen3VL-8B-Instruct-F16.gguf
on 127.0.0.1 only. Every window_s seconds a short clip of n_frames full frames over that
window (long side px, JPEG in memory, never written to disk) goes to the server with a
prompt that lists the phases in order with their short descriptions
(analysis/phase_text.py) and asks for JSON: a probability per phase and a one line
reason. Scores are renormalised to sum to 1; a reply that cannot be parsed counts as
uniform (and is counted). Only scores and reasons are stored.
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

from ..storage.files import REPO_ROOT

SERVER = REPO_ROOT / "third_party/llama.cpp/build/bin/llama-server"
MODEL = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
MMPROJ = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"


def prompt(phases: list[dict], process: str) -> str:
    lines = [f"These {len(phases)} frames come from {process}, in time order over about two seconds.",
             "The work goes through these phases, always in this order:", ""]
    for i, p in enumerate(phases):
        lines.append(f"{i + 1}. {p['name']}: " + " ".join(p["short"]))
    names = ", ".join(f'"{p["name"]}"' for p in phases)
    lines += ["", "Which phase is shown? Answer with JSON only, in this form:",
              '{"probabilities": {' + names.replace('", "', '": p, "') + '": p}, "reason": "one short line"}',
              "The probabilities are numbers from 0 to 1 that sum to 1."]
    return "\n".join(lines)


def start_server(gpu: int, port: int, log: Path, ctx: int = 8192, parallel: int = 2) -> subprocess.Popen:
    import os
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [str(SERVER), "-m", str(REPO_ROOT / "models" / MODEL), "--mmproj", str(REPO_ROOT / "models" / MMPROJ),
           "--host", "127.0.0.1", "--port", str(port), "-ngl", "99", "-c", str(ctx), "--parallel", str(parallel),
           "--no-webui", "--temp", "0"]
    f = open(log, "a")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited ({proc.returncode}); see {log}")
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server did not become ready in 10 minutes")


def jpeg(frame: np.ndarray, px: int) -> str:
    import cv2
    h, w = frame.shape[:2]
    s = px / max(h, w)
    small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def ask(port: int, text: str, frames: list[np.ndarray], px: int, timeout: float = 120.0) -> str:
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg(f, px)}} for f in frames]
    content.append({"type": "text", "text": text})
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 160,
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse(reply: str, names: list[str]) -> tuple[np.ndarray, str, bool]:
    """(probabilities [K] summing to 1, reason, parsed)."""
    K = len(names)
    try:
        m = re.search(r"\{.*\}", reply, re.S)
        d = json.loads(m.group(0)) if m else {}
        probs = d.get("probabilities", d)
        low = {str(k).strip().lower(): v for k, v in probs.items()}
        p = np.array([float(low.get(n.lower(), 0.0)) for n in names], np.float64)
        p = np.clip(np.nan_to_num(p), 0.0, None)
        if p.sum() <= 0:
            raise ValueError("no probability")
        return (p / p.sum()).astype(np.float32), str(d.get("reason", ""))[:200], True
    except Exception:
        return np.full(K, 1.0 / K, np.float32), reply.strip().replace("\n", " ")[:200], False


def windows(video: Path, window_s: float, n_frames: int, max_px: int):
    """Yields (window start s, [n_frames RGB frames]) over the video: the frames at
    start + k * window_s / n_frames, k = 0..n_frames-1 (ffmpeg, in memory)."""
    from .siglip import sample_frames
    fps = n_frames / window_s
    buf, t0 = [], 0.0
    for t, frame, _ in sample_frames(video, fps, max_px):
        if not buf:
            t0 = t
        buf.append(frame)
        if len(buf) == n_frames:
            yield t0, buf
            buf = []
    if buf:
        yield t0, buf
