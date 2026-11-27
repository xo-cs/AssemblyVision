"""Speed of the web app, measured in headless Chrome through the real app (Week 12).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision
    AV_PASSWORD='...' python scripts/ui_timing.py --base http://127.0.0.1:7860 --user NAME --run RUN_ID --area qisens

Logs in, then times: the first page and the interactive page; the switch to Results; opening the run; the Movement
tabs 2D, 3D and 4D; the start of the overlay video. Records every response the page loads (size on the wire, type,
compression, range support). Prints only timings, sizes and headers: no screenshot, no page text, no frame. Chrome's
profile and cache lie in data/<area>/tmp/ and are deleted at the end, so nothing of a QiSens run leaves data/qisens.
Writes data/<area>/validation/ui_timing_<run>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
BUSY = "document.querySelector('.pending, .eta-bar, .progress-text, .generating') !== null"


class Chrome:
    def __init__(self, profile: Path, port: int = 9344):
        self.proc = subprocess.Popen(["google-chrome", "--headless=new", "--no-sandbox", f"--remote-debugging-port={port}",
                                      "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
                                      "--autoplay-policy=no-user-gesture-required", "--window-size=1400,1000",
                                      f"--user-data-dir={profile}", f"--disk-cache-dir={profile / 'cache'}", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     env={"HOME": str(profile), "PATH": "/usr/bin:/bin"})
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.1)
        self.loop = asyncio.new_event_loop()
        self.ws = self.loop.run_until_complete(websockets.connect(url, max_size=2 ** 28))
        self.n, self.responses, self.inflight, self.sent, self.done = 0, {}, {}, {}, []
        for m in ("Runtime.enable", "Page.enable", "Network.enable"):
            self.call(m)

    def _event(self, m: dict) -> None:
        p = m.get("params", {})
        meth = m.get("method", "")
        if meth == "Network.requestWillBeSent":
            self.inflight[p["requestId"]] = p["request"]["url"]
            self.sent[p["requestId"]] = (time.monotonic(), p["request"]["url"], p["request"].get("postData", ""))
        elif meth == "Network.responseReceived":
            r = p["response"]
            h = {k.lower(): v for k, v in r.get("headers", {}).items()}
            self.responses[p["requestId"]] = {"url": r["url"], "status": r["status"], "mime": r.get("mimeType", ""),
                                              "encoding": h.get("content-encoding", ""), "accept_ranges": h.get("accept-ranges", ""),
                                              "cache": h.get("cache-control", ""), "bytes": 0, "t": time.monotonic()}
        elif meth in ("Network.loadingFinished", "Network.loadingFailed"):
            self.inflight.pop(p["requestId"], None)
            if p["requestId"] in self.sent:
                t, u, body = self.sent.pop(p["requestId"])
                fn = None
                if "queue/join" in u and body:
                    try:
                        fn = json.loads(body).get("fn_index")
                    except ValueError:
                        pass
                self.done.append({"url": u, "fn_index": fn, "start": t, "s": round(time.monotonic() - t, 3)})
            if meth == "Network.loadingFinished" and p["requestId"] in self.responses:
                self.responses[p["requestId"]]["bytes"] = int(p.get("encodedDataLength", 0))
        elif meth == "Network.dataReceived" and p["requestId"] in self.responses:
            self.responses[p["requestId"]]["bytes"] += int(p.get("encodedDataLength", 0))

    def call(self, method: str, **params):
        self.n += 1
        mid = self.n

        async def go():
            await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
            while True:
                m = json.loads(await self.ws.recv())
                if m.get("id") == mid:
                    return m
                self._event(m)
        return self.loop.run_until_complete(go())

    def pump(self, seconds: float) -> None:
        async def go():
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                try:
                    self._event(json.loads(await asyncio.wait_for(self.ws.recv(), max(0.01, end - time.monotonic()))))
                except asyncio.TimeoutError:
                    break
        self.loop.run_until_complete(go())

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("result", {}).get("value")

    def quiet(self) -> bool:
        """No Gradio request in flight and no 3D or 4D data loading (the heartbeat and the video stream do not count),
        and no progress shown."""
        busy = [u for u in self.inflight.values() if "/gradio_api/" in u and "heartbeat" not in u
                and ("file=" not in u or ".player." in u)]
        return not busy and not self.js(BUSY)

    def until(self, expr: str, timeout: float = 60.0, settle: float = 0.3) -> float | None:
        """Seconds from now until expr is true and the page has been quiet for settle seconds (None on timeout)."""
        t0 = time.monotonic()
        calm = None
        while time.monotonic() - t0 < timeout:
            self.pump(0.05)
            if self.js(expr) and self.quiet():
                calm = calm or time.monotonic()
                if time.monotonic() - calm >= settle:
                    return round(calm - t0, 3)
            else:
                calm = None
        return None

    def close(self) -> None:
        try:
            self.loop.run_until_complete(self.ws.close())
        except Exception:
            pass
        self.proc.terminate()
        self.proc.wait(10)


def kind(r: dict) -> str:
    u, m = r["url"], r["mime"]
    if "file=" in u:
        return "video" if m.startswith("video") or u.split("?")[0].endswith(".mp4") else \
            "3D/4D data" if (".player." in u) else "image" if m.startswith("image") else "file"
    if "/gradio_api/" in u:
        return "Gradio events and data"
    if m.startswith(("text/html",)):
        return "page"
    if m in ("application/javascript", "text/javascript", "text/css") or u.endswith((".js", ".css")):
        return "scripts and styles"
    return "other"


def measure(base: str, user: str, password: str, run_id: str, profile: Path) -> dict:
    c = Chrome(profile)
    res: dict = {"base": base, "run": run_id, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "times_s": {}}
    T = res["times_s"]
    try:
        c.call("Page.navigate", url=base + "/login")
        c.pump(1.0)
        ok = c.js(f"fetch('/login', {{method: 'POST', body: new URLSearchParams({{username: {json.dumps(user)}, "
                  f"password: {json.dumps(password)}}})}}).then(r => r.status)")
        if ok not in (200, 302, 303):
            raise SystemExit(f"login failed (HTTP {ok})")
        c.responses.clear()
        t0 = time.monotonic()
        c.call("Page.navigate", url=base + "/")
        first = c.until("document.querySelector('#av-main') !== null", settle=0.0)
        T["first page"] = first
        inter = c.until("document.querySelector('#nav-results') !== null")
        T["interactive page"] = None if inter is None else round(time.monotonic() - t0 - 0.3, 3)
        c.js("document.querySelector('#nav-results').click()")
        T["switch to Results"] = c.until("(() => { const p = document.querySelector('.av-picker'); return !!(p && p.offsetParent); })()")
        t0 = time.monotonic()
        c.call("Page.navigate", url=f"{base}/?view=results&run={run_id}")
        got = c.until(f"(() => document.querySelector('.av-run-id')?.innerText?.includes({json.dumps(run_id)}) "
                      "&& document.querySelectorAll('.av-kpi, .kpi, .av-card').length > 3)()", timeout=90)
        T["open a run (page load to filled Results)"] = None if got is None else round(time.monotonic() - t0 - 0.3, 3)
        names = {}
        try:
            cfg = json.loads(c.js("fetch('/config').then(r => r.text())"))
            comps = {x["id"]: x for x in cfg.get("components", [])}
            for i, d in enumerate(cfg.get("dependencies", [])):
                trig = d.get("targets") or []
                what = [f"{comps.get(t[0], {}).get('props', {}).get('elem_id') or comps.get(t[0], {}).get('type', '?')}.{t[1]}" for t in trig[:1]]
                outs = [comps.get(o, {}).get("type", "?") for o in d.get("outputs", [])][:3]
                names[d.get("id", i)] = f"{','.join(what)} -> {len(d.get('outputs', []))} outputs ({','.join(outs)})"
        except (ValueError, TypeError):
            pass
        res["open_run_events"] = sorted(({"fn_index": e["fn_index"], "what": names.get(e["fn_index"], "?"), "s": e["s"]}
                                         for e in c.done if e["fn_index"] is not None and e["start"] >= t0), key=lambda e: -e["s"])[:10]
        t1 = time.monotonic()
        c.call("Page.navigate", url=f"{base}/?view=results&run={run_id}")
        got = c.until(f"(() => document.querySelector('.av-run-id')?.innerText?.includes({json.dumps(run_id)}) "
                      "&& document.querySelectorAll('.av-kpi, .kpi, .av-card').length > 3)()", timeout=90)
        T["open the run again"] = None if got is None else round(time.monotonic() - t1 - 0.3, 3)
        c.pump(1.0)
        res["player_files_before_any_tab"] = sum(".player." in r["url"] for r in c.responses.values())
        for tab in ("3D", "4D", "2D"):
            c.js(f"""(() => {{ const t = Array.from(document.querySelectorAll('.av-mv-mode label')).find(x => x.innerText.trim() === {tab!r});
                (t.querySelector('input') || t).click(); }})()""")
            need = ("Array.from(document.querySelectorAll('.av-player')).some(e => e.offsetParent && e.querySelector('canvas'))"
                    if tab != "2D" else "true")
            T[f"Movement tab {tab}"] = c.until(need + " || Array.from(document.querySelectorAll('.av-mv-panel .av-quiet')).some(e => e.offsetParent)", timeout=90)
        t0 = time.monotonic()
        started = c.js("""(() => { const v = Array.from(document.querySelectorAll('video')).find(x => x.offsetParent && x.closest('#av-main'));
            if (!v) return false; v.muted = true; v.play(); return true; })()""")
        if started:
            while time.monotonic() - t0 < 30:
                c.pump(0.05)
                if c.js("Array.from(document.querySelectorAll('video')).some(x => x.offsetParent && x.currentTime > 0)"):
                    T["overlay video starts"] = round(time.monotonic() - t0, 3)
                    break
        c.pump(2.0)
        rows = list(c.responses.values())
        by: dict = {}
        for r in rows:
            k = kind(r)
            by.setdefault(k, {"requests": 0, "bytes": 0})
            by[k]["requests"] += 1
            by[k]["bytes"] += r["bytes"]
        res["payload_by_kind"] = by
        res["largest"] = [{"kind": kind(r), "kB": round(r["bytes"] / 1024, 1), "status": r["status"], "encoding": r["encoding"] or "none",
                           "accept_ranges": r["accept_ranges"] or "-", "path": r["url"].split("?")[0].rsplit("/", 1)[-1][:60]}
                          for r in sorted(rows, key=lambda r: -r["bytes"])[:12]]
        vids = [r for r in rows if kind(r) == "video"]
        res["video"] = {"responses": len(vids), "statuses": sorted({r["status"] for r in vids}),
                        "accept_ranges": sorted({r["accept_ranges"] for r in vids})}
        text = [r for r in rows if r["mime"].startswith(("text/", "application/json", "application/javascript")) and r["bytes"] > 20000]
        res["uncompressed_text_over_20kB"] = sum(1 for r in text if not r["encoding"])
    finally:
        c.close()
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:7860")
    p.add_argument("--user", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--area", choices=("qisens", "public"), required=True, help="the data area of the run (Chrome's cache goes there)")
    p.add_argument("--label", default="")
    a = p.parse_args()
    pw = os.environ.get("AV_PASSWORD")
    if not pw:
        print("set AV_PASSWORD in the environment", file=sys.stderr)
        return 2
    area = ROOT / "data" / a.area
    (area / "tmp").mkdir(exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="ui_timing_", dir=area / "tmp"))
    try:
        res = measure(a.base.rstrip("/"), a.user, pw, a.run, profile)
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    res["label"] = a.label
    out = area / "validation"
    out.mkdir(exist_ok=True)
    (out / f"ui_timing_{a.run}{'_' + a.label if a.label else ''}.json").write_text(json.dumps(res, indent=1))
    print(f"{a.label or 'run'} {a.run}")
    for k, v in res["times_s"].items():
        print(f"  {k:<42} {'timeout' if v is None else f'{v:.2f} s'}")
    for k, v in res["payload_by_kind"].items():
        print(f"  payload {k:<34} {v['requests']:>3} requests {v['bytes'] / 1e6:8.2f} MB")
    print(f"  video: {res['video']}; text responses over 20 kB without compression: {res['uncompressed_text_over_20kB']}")
    print(f"  3D or 4D files loaded before any Movement tab was opened: {res.get('player_files_before_any_tab')}")
    for e in res.get("open_run_events", [])[:8]:
        print(f"  open run event fn {e['fn_index']}: {e['s']:.2f} s  {e['what'][:80]}")
    for r in res["largest"][:8]:
        print(f"  {r['kB']:>9} kB {r['kind']:<22} {r['status']} enc={r['encoding']:<5} ranges={r['accept_ranges']:<6} {r['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
