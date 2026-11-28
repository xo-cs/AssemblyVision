"""Group c of w12-robustness: three local text only LLMs write the phase descriptions (rule.md, amendment of 2026-11-28).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision
    python experiments/w12-robustness/group_c/generate.py

One model at a time: llama-server (third_party/llama.cpp) on GPU1 under .locks/gpu1.lock, one chat request with the prompt
of the rule, temperature 0.7, seed 20261128, at most 4,096 new tokens; then the server stops. Saves next to this file, per
model: <key>.cmd.txt (the exact server command line and the request), <key>.raw.json (the response) and <key>.raw.md (the
text as the model wrote it). No image, video or file of a run goes into any model.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SERVER = ROOT / "third_party/llama.cpp/build/bin/llama-server"
MODELS = {"llm_qwen3": "Qwen3-4B-Q4_K_M.gguf", "llm_llama32": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
          "llm_gemma3": "google_gemma-3-4b-it-Q4_K_M.gguf"}
GPU, PORT, SEED, TEMP, MAX_TOKENS = 1, 8095, 20261128, 0.7, 4096
PROMPT = """You write visual descriptions for a video classifier. Below are 5 work phases of train bogie maintenance in a factory. For each phase, write 22 short sentences describing what a camera would SEE during that phase: tools, parts, what the hands and body do, the state of the parts, wide and close views. Do not use the phase names inside the sentences.
1. Removal: the worker takes components off the train or bogie (for example wheelsets, covers, brake parts) and moves them away.
2. Disassembly: the worker takes the removed components apart into smaller parts (unbolting, separating, opening housings).
3. Cleaning: the worker cleans the parts (washing, brushing, wiping, blowing off dirt, grease or rust).
4. Adjustment and Inspection: the worker checks and measures the parts (gauges, measuring tools, visual checks) and adjusts them.
5. Assembly and Installation: the worker puts the parts back together and mounts them onto the bogie or train (fitting, bolting, tightening).
Format exactly: '## <phase name>' then 22 lines starting with '- '."""


def wait_ready(proc, log):
    for _ in range(300):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited ({proc.returncode}); see {log}")
        time.sleep(1)
    raise RuntimeError("llama-server did not become ready")


def main() -> int:
    lock = open(ROOT / ".locks" / f"gpu{GPU}.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)        # refuses at once if GPU1 is busy
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(GPU))
    for key, gguf in MODELS.items():
        if (HERE / f"{key}.raw.json").is_file():               # one generation per model, never a second one
            print(f"{key}: already generated, kept")
            continue
        cmd = [str(SERVER), "-m", str(ROOT / "models/text_llm" / gguf), "--host", "127.0.0.1", "--port", str(PORT),
               "-ngl", "99", "-c", "8192", "--parallel", "1", "--no-webui", "--jinja", "--seed", str(SEED)]
        body = {"messages": [{"role": "user", "content": PROMPT}], "temperature": TEMP, "seed": SEED,
                "max_tokens": MAX_TOKENS, "chat_template_kwargs": {"enable_thinking": False}}
        (HERE / f"{key}.cmd.txt").write_text("server: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 " + " ".join(cmd)
                                             + "\n\nrequest POST /v1/chat/completions:\n" + json.dumps(body, indent=1) + "\n")
        log = HERE / f"{key}.server.log"
        with open(log, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        try:
            wait_ready(proc, log)
            t0 = time.monotonic()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                resp = json.loads(r.read())
            (HERE / f"{key}.raw.json").write_text(json.dumps(resp, indent=1) + "\n")
            text = resp["choices"][0]["message"]["content"]
            (HERE / f"{key}.raw.md").write_text(text + ("\n" if not text.endswith("\n") else ""))
            u = resp.get("usage", {})
            print(f"{key}: {u.get('completion_tokens')} tokens in {time.monotonic() - t0:.1f} s, finish {resp['choices'][0].get('finish_reason')}")
        finally:
            proc.terminate()
            proc.wait(30)
    lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
