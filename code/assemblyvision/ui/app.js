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

  // --- comparison mode (Results, Overlay video card) ------------------------------------
  // Raw footage (the run's constant frame rate video) left and the overlay right,
  // fullscreen, one control bar; the overlay follows the raw video within 0.1 s.
  // The normal player and its own fullscreen button are not touched.
  /*COMPARE_SYNC*/
  const openCompare = (rawSrc, offset) => {
    const normal = document.querySelector("#av-overlay video");
    if (!normal || document.getElementById("av-compare")) return;
    const t0 = normal.currentTime || 0;
    normal.pause();
    const box = document.createElement("div");
    box.id = "av-compare";
    box.innerHTML =
      '<button class="av-cmp-close" type="button" title="Close (Esc)" aria-label="Close comparison">&#x2715;</button>' +
      '<div class="av-cmp-panes">' +
      '<figure><figcaption>Raw footage</figcaption><video class="av-cmp-lead" preload="auto" playsinline muted></video></figure>' +
      '<figure><figcaption>AssemblyVision analysis</figcaption><video class="av-cmp-follow" preload="auto" playsinline muted></video></figure>' +
      "</div>" +
      '<div class="av-cmp-bar"><button class="av-cmp-play" type="button">Play</button>' +
      '<input class="av-cmp-seek" type="range" min="0" max="0" step="0.01" value="0" aria-label="Seek">' +
      '<span class="av-cmp-time">0:00 / 0:00</span></div>';
    document.body.appendChild(box);
    const lead = box.querySelector(".av-cmp-lead"), follow = box.querySelector(".av-cmp-follow");
    const play = box.querySelector(".av-cmp-play"), seek = box.querySelector(".av-cmp-seek");
    const clock = box.querySelector(".av-cmp-time");
    let raf = 0, dragging = false;
    const sync = () => {
      const t = avSyncTime(lead.currentTime, follow.currentTime, offset, 0.1);
      if (t !== null && !follow.seeking) follow.currentTime = t;
    };
    const show = () => {
      const t = Math.max(0, lead.currentTime - offset), d = follow.duration || 0;
      if (!dragging) seek.value = t;
      clock.textContent = avClock(t) + " / " + avClock(d);
      play.textContent = lead.paused ? "Play" : "Pause";
    };
    const loop = () => { sync(); show(); raf = lead.paused ? 0 : requestAnimationFrame(loop); };
    lead.addEventListener("play", () => { sync(); follow.play().catch(() => {}); if (!raf) raf = requestAnimationFrame(loop); show(); });
    lead.addEventListener("pause", () => { follow.pause(); sync(); show(); });
    lead.addEventListener("seeked", () => { sync(); show(); });
    lead.addEventListener("timeupdate", () => { sync(); show(); });
    follow.addEventListener("loadedmetadata", () => { seek.max = follow.duration || 0; show(); });
    play.addEventListener("click", () => { if (lead.paused) lead.play().catch(() => {}); else lead.pause(); });
    seek.addEventListener("input", () => { dragging = true; lead.currentTime = parseFloat(seek.value) + offset; });
    seek.addEventListener("change", () => { dragging = false; sync(); });
    const close = () => {
      document.removeEventListener("fullscreenchange", onFs);
      document.removeEventListener("keydown", onKey, true);
      if (raf) cancelAnimationFrame(raf);
      for (const v of [lead, follow]) { v.pause(); v.removeAttribute("src"); v.load(); }
      if (document.fullscreenElement === box) document.exitFullscreen().catch(() => {});
      box.remove();
    };
    const onFs = () => { if (document.fullscreenElement !== box) close(); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    box.querySelector(".av-cmp-close").addEventListener("click", close);
    document.addEventListener("keydown", onKey, true);
    lead.src = rawSrc;
    follow.src = normal.currentSrc || normal.src;
    lead.currentTime = t0 + offset;
    follow.currentTime = t0;
    if (box.requestFullscreen) {
      box.requestFullscreen().then(() => document.addEventListener("fullscreenchange", onFs)).catch(() => {});
    }
  };
  document.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-compare-raw]");
    if (btn) openCompare(btn.dataset.compareRaw, parseFloat(btn.dataset.compareOffset || "0") || 0);
  });

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
