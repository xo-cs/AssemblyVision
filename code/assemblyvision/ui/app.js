() => {
  // keep the page light even when the browser prefers dark mode
  document.body.classList.remove("dark");
  // Open buttons in HTML tables (Runs page): hand the run ID to the hidden
  // textbox #av-open-run, whose input event opens the run on the Results page
  document.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-open-run]");
    if (!btn) return;
    const box = document.querySelector("#av-open-run textarea, #av-open-run input");
    if (!box) return;
    box.value = btn.dataset.openRun + "|" + Date.now();
    box.dispatchEvent(new Event("input", { bubbles: true }));
  });

  // --- Live camera ------------------------------------------------------------------
  // Gradio 5.50 webcam (#lv-camera): a "grant webcam access" element first, then one
  // record button whose icon title is "start recording", "spinner" (waiting) or
  // "stop recording". Start live and Stop drive these, so one click does it all.
  const cam = () => document.getElementById("lv-camera");
  const recButton = () => cam()?.querySelector(".button-wrap button");
  const state = () => {
    const b = recButton();
    if (!b) return "no-access";
    if (b.querySelector('[title="stop recording"]')) return "open";
    if (b.querySelector('[title="spinner"]')) return "waiting";
    return "closed";
  };
  const liveRunning = () => !!document.querySelector('[data-live-status="running"]');
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  window.avCam = {
    // ask for the camera at once on the Start click (the browser may show its permission prompt)
    access() {
      const grant = cam()?.querySelector('[title="grant webcam access"]');
      if (grant) (grant.firstElementChild || grant).click();
    },
    // after the session started on the server: wait for the camera, then start sending frames
    async recordIfLive() {
      for (let i = 0; i < 300 && !document.querySelector("[data-live-status]"); i++) await sleep(100);
      if (!liveRunning()) return;
      for (let i = 0; i < 600; i++) {                        // up to 60 s for the permission prompt
        if (state() === "no-access") this.access();
        const v = cam()?.querySelector("video");
        if (state() === "closed" && v && v.videoWidth > 0) { recButton().click(); return; }
        if (state() === "open" || state() === "waiting" || !liveRunning()) return;
        await sleep(100);
      }
    },
    stop() {
      if (state() === "open" || state() === "waiting") recButton().click();
    },
  };

  // --- frame rate -----------------------------------------------------------------------
  // The webcam offers a frame every stream_every (0.04 s); only as many as the
  // session accepts (data-live-rate in the live panel, stepped down by the server)
  // are sent over the stream websocket, so the SSH tunnel carries no unused frames.
  const send = WebSocket.prototype.send;
  WebSocket.prototype.send = function (data) {
    if (this.url && this.url.includes("/stream/")) {
      const el = document.querySelector("[data-live-rate]");
      const fps = el ? parseFloat(el.dataset.liveRate) : 0;
      const now = performance.now();
      if (fps > 0 && this._avLast && now - this._avLast < 900 / fps) return;
      this._avLast = now;
    }
    return send.call(this, data);
  };
  // webcam frames are JPEG data URLs at the browser default quality (about 0.92):
  // 0.75 keeps them sharp enough for pose and makes them much smaller
  const toDataURL = HTMLCanvasElement.prototype.toDataURL;
  HTMLCanvasElement.prototype.toDataURL = function (type, quality) {
    if (type === "image/jpeg" && quality === undefined) return toDataURL.call(this, type, 0.75);
    return toDataURL.call(this, type, quality);
  };
}
