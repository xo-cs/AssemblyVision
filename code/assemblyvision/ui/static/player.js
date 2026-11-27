// AssemblyVision player for the Heatmap page (3D: GVHMR skeleton with hand and foot voxels and the
// floor map building up; 4D: the Pi3 scene with the person, trails, the camera and the aligned skeleton).
// One clock: its own timer, Play, Pause, a slider in video time, speed 0.5x, 1x, 2x; the run's overlay
// video follows the clock (and the clock follows the video when the video is played or seeked).
// At the end of a shot part it continues with the next part (each part keeps its own world frame).
// All files come from the logged in app (Gradio's file route); nothing else is contacted.
import * as THREE from "./three/three.module.min.js";
import { OrbitControls } from "./three/OrbitControls.js";

const BONES = [[5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]];
const TRAIL_JOINTS = [9, 10, 15, 16];        // wrists and ankles
const clock = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const heat = (x) => {                         // 0..1 -> yellow to dark red (time spent)
  const c = new THREE.Color();
  c.setHSL(0.15 - 0.15 * x, 1.0, 0.62 - 0.32 * x);
  return c;
};

function el(tag, attrs = {}, parent = null) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => (k === "text" ? (e.textContent = v) : e.setAttribute(k, v)));
  if (parent) parent.appendChild(e);
  return e;
}

// the compact copies (gzip; 4D positions as int16 with an offset and a scale, ui/player_files.py) when the browser
// can unzip them, else the original files
const GZ = typeof DecompressionStream !== "undefined";
async function fetchBuf(url) {
  const r = await fetch(url);
  return url.endsWith(".gz") ? new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer()
                             : r.arrayBuffer();
}

async function loadPart(part) {
  if (part.data) return part.data;
  const useGz = GZ && part.json_gz && (!part.bin || part.bin_gz);
  const head = JSON.parse(new TextDecoder().decode(await fetchBuf(useGz ? part.json_gz : part.json)));
  let buf = null;
  if (part.bin) buf = await fetchBuf(useGz ? part.bin_gz : part.bin);
  const done = {};
  const block = (k) => {
    const b = head[k];
    if (!b) return null;
    if (done[k]) return done[k];
    const n = b.shape.reduce((a, x) => a * x, 1);
    let out;
    if (b.dtype === "q16") {                         // value = min + (q + 32767) * scale; -32768 is NaN
      const q = new Int16Array(buf, b.offset, n);
      out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = q[i] === -32768 ? NaN : b.min + (q[i] + 32767) * b.scale;
    } else {
      out = b.dtype === "f32" ? new Float32Array(buf, b.offset, n) : new Uint8Array(buf, b.offset, n);
    }
    done[k] = out;
    return out;
  };
  part.data = { head, block };
  return part.data;
}

export function mount(root) {
  if (root.dataset.mounted) return;
  root.dataset.mounted = "1";
  const cfg = JSON.parse(root.dataset.cfg);
  const dark = cfg.mode === "4d";
  root.innerHTML = "";
  root.classList.add("avp", dark ? "avp-dark" : "avp-light");
  const top = el("div", { class: "avp-row" }, root);
  const view = el("div", { class: "avp-view" }, top);
  let video = null;
  if (cfg.video) {
    const side = el("div", { class: "avp-side" }, top);
    video = el("video", { src: cfg.video, muted: "", playsinline: "", preload: "auto", class: "avp-video" }, side);
    video.muted = true;
  }
  const floorBox = cfg.mode === "3d" ? el("div", { class: "avp-floor" }, root) : null;
  const floor = floorBox ? el("canvas", { width: 640, height: 220 }, floorBox) : null;
  const bar = el("div", { class: "avp-bar" }, root);
  const play = el("button", { class: "avp-btn", text: "Play" }, bar);
  const slider = el("input", { type: "range", min: 0, max: 1000, value: 0, class: "avp-slider" }, bar);
  const label = el("span", { class: "avp-time", text: "0:00.0" }, bar);
  const speed = el("select", { class: "avp-speed" }, bar);
  ["0.5", "1", "2"].forEach((s) => { const o = el("option", { value: s, text: `${s}x` }, speed); if (s === "1") o.selected = true; });
  const partSel = el("select", { class: "avp-part" }, bar);
  cfg.parts.forEach((p, i) => el("option", { value: i, text: p.label }, partSel));
  let allAtOnce = null;
  if (cfg.mode === "3d") {
    const lab = el("label", { class: "avp-check" }, bar);
    allAtOnce = el("input", { type: "checkbox" }, lab);
    lab.appendChild(document.createTextNode(" show all at once"));
  }
  const note = el("div", { class: "avp-note" }, root);

  // three.js scene
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  view.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(dark ? 0x0b0d12 : 0xf4f5f7);
  const cam = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
  const pxScale = { value: 300 };          // pixels per world unit at distance 1 (point sizes of the 4D scene)
  const orbit = new OrbitControls(cam, renderer.domElement);
  orbit.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const world = new THREE.Group();          // the part's frame, turned so that up is +y
  scene.add(world);
  const resize = () => {
    const w = view.clientWidth || 640, h = Math.round(w * 0.62);
    renderer.setSize(w, h);
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(view);
  resize();

  let cur = -1, t = cfg.parts[0].t0, playing = false, last = 0, drawn = null, seekTarget = null, lastVt = null, vtSince = 0;
  const seekVideo = (x) => {                  // a seek of ours; its "seeked" event is not taken as the user's
    if (!video) return;
    seekTarget = x;
    if (video.readyState >= 1) video.currentTime = x;
  };
  const parts = cfg.parts;

  function clear() {
    while (world.children.length) {
      const o = world.children.pop();
      o.geometry?.dispose?.();
      o.material?.dispose?.();
    }
  }

  // --- 3D: skeleton, voxels building up, floor map ------------------------------------------------
  function build3d(d) {
    const h = d.head;
    world.quaternion.identity();
    const J = h.joints, F = h.frames.length;
    const skel = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x2b5c9e }));
    skel.geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(BONES.length * 6), 3));
    world.add(skel);
    const vox = h.voxels, nV = vox.length / 3;
    const box = new THREE.BoxGeometry(h.voxel_m * 0.9, h.voxel_m * 0.9, h.voxel_m * 0.9);
    const inst = new THREE.InstancedMesh(box, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.85 }), nV);
    inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    world.add(inst);
    const grid = new THREE.GridHelper(Math.max(4, h.extent * 2), Math.max(8, Math.round(h.extent * 20)), 0xb8bec8, 0xd8dce2);
    grid.position.set(h.center[0], h.ground, h.center[2]);
    world.add(grid);
    cam.position.set(h.center[0] + h.extent * 1.4, h.ground + h.extent * 1.1, h.center[2] + h.extent * 1.6);
    orbit.target.set(h.center[0], h.ground + 0.8, h.center[2]);
    // per voxel and per floor cell: the frame index of every visit (sorted), to count visits up to a frame
    const visits = Array.from({ length: nV }, () => []);
    h.vox_ids.forEach((ids, f) => ids.forEach((v) => v >= 0 && visits[v].push(f)));
    const cvis = Array.from({ length: h.cells_x * h.cells_z }, () => []);
    h.cell_ids.forEach((ids, f) => ids.forEach((c) => c >= 0 && cvis[c].push(f)));
    const kind = h.vox_kind;                   // 0 hands, 1 feet
    const m4 = new THREE.Matrix4(), zero = new THREE.Matrix4().makeScale(0, 0, 0);
    const count = (arr, f) => { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= f) lo = m + 1; else hi = m; } return lo; };
    drawn = (f) => {
      const p = skel.geometry.attributes.position.array;
      BONES.forEach(([a, b], i) => { for (let k = 0; k < 3; k++) { p[i * 6 + k] = J[f * 51 + a * 3 + k]; p[i * 6 + 3 + k] = J[f * 51 + b * 3 + k]; } });
      skel.geometry.attributes.position.needsUpdate = true;
      const upto = allAtOnce?.checked ? F - 1 : f;
      let maxS = 1e-6;
      const sec = visits.map((v) => count(v, upto) / h.fps / 2);
      sec.forEach((s) => (maxS = Math.max(maxS, s)));
      for (let v = 0; v < nV; v++) {
        if (sec[v] <= 0) { inst.setMatrixAt(v, zero); continue; }
        const s = 0.45 + 0.55 * Math.sqrt(sec[v] / maxS);
        m4.makeScale(s, s, s).setPosition(vox[v * 3], vox[v * 3 + 1], vox[v * 3 + 2]);
        inst.setMatrixAt(v, m4);
        const c = heat(Math.sqrt(sec[v] / maxS));
        if (kind[v] === 1) c.offsetHSL(0.5, 0, 0);   // feet in blue tones, hands in red tones
        inst.setColorAt(v, c);
      }
      inst.instanceMatrix.needsUpdate = true;
      if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
      // the floor map below: seconds per cell so far
      const ctx = floor.getContext("2d"), W = floor.width, H = floor.height;
      ctx.fillStyle = "#f4f5f7"; ctx.fillRect(0, 0, W, H);
      const cs = cvis.map((v) => count(v, upto) / h.fps / 2);
      const mx = Math.max(1e-6, ...cs);
      const s2 = Math.min(W / h.cells_x, (H - 22) / h.cells_z, 28);             // centred, at most 28 px per cell
      const ox = (W - s2 * h.cells_x) / 2, oz = (H - 22 - s2 * h.cells_z) / 2;
      ctx.strokeStyle = "#dde1e7"; ctx.strokeRect(ox - 0.5, oz - 0.5, s2 * h.cells_x + 1, s2 * h.cells_z + 1);
      cs.forEach((s, i) => {
        if (s <= 0) return;
        const x = i % h.cells_x, z = Math.floor(i / h.cells_x);
        ctx.fillStyle = `#${heat(Math.sqrt(s / mx)).getHexString()}`;
        ctx.fillRect(ox + x * s2, oz + z * s2, s2 - 1, s2 - 1);
      });
      ctx.fillStyle = "#3a4250"; ctx.font = "12px sans-serif";
      ctx.fillText(`Feet on the floor, ${Math.round(h.cell_m * 100)} cm cells, up to ${mx.toFixed(1)} s per cell`, 6, H - 6);
    };
  }

  // --- 4D: scene points, person, camera, trails, skeleton ------------------------------------------
  function build4d(d) {
    const h = d.head, B = d.block;
    const up = new THREE.Vector3(...(h.up || [0, -1, 0])).normalize();       // Pi3 frames are y down unless aligned
    world.quaternion.setFromUnitVectors(up, new THREE.Vector3(0, 1, 0));
    const pts = (P, C, size) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(P, 3));
      g.setAttribute("color", new THREE.BufferAttribute(C, 3, true));
      return new THREE.Points(g, new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true }));
    };
    const bgP = B("bg_pts"), bgS = B("bg_size");
    // the size of the scene: 5th to 95th percentile box of the background (a long lens sees a small, far scene)
    const ext = (() => {
      const n = bgP.length / 3, step = Math.max(1, Math.floor(n / 5000)), ax = [[], [], []];
      for (let i = 0; i < n; i += step) for (let k = 0; k < 3; k++) ax[k].push(bgP[i * 3 + k]);
      const q = (a, f) => { a.sort((x, y) => x - y); return a[Math.floor(f * (a.length - 1))]; };
      const lo = ax.map((a) => q(a, 0.05)), hi = ax.map((a) => q(a, 0.95));
      return { center: new THREE.Vector3(...lo.map((v, k) => (v + hi[k]) / 2)), size: Math.hypot(...hi.map((v, k) => v - lo[k])) || 1 };
    })();
    const depth = ext.size;
    if (bgS) {        // round points of their own world size (the gap to the neighbours), smaller with distance
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(bgP, 3));
      g.setAttribute("color", new THREE.BufferAttribute(B("bg_col"), 3, true));
      g.setAttribute("psize", new THREE.BufferAttribute(bgS, 1));
      const mat = new THREE.ShaderMaterial({
        uniforms: { scale: pxScale },
        vertexShader: `attribute float psize; varying vec3 vC; uniform float scale;
          void main() { vC = color; vec4 mv = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = clamp(psize * scale / -mv.z, 1.0, 24.0); gl_Position = projectionMatrix * mv; }`,
        fragmentShader: `varying vec3 vC;
          void main() { vec2 q = gl_PointCoord - 0.5; if (dot(q, q) > 0.25) discard; gl_FragColor = vec4(vC, 1.0); }`,
        vertexColors: true,
      });
      world.add(new THREE.Points(g, mat));
    } else {
      world.add(pts(bgP, B("bg_col"), depth / 260));
    }
    const S = h.frames.length, P = h.person_pts.shape[1];
    const pp = B("person_pts"), pc = B("person_col");
    const person = pts(new Float32Array(P * 3), new Uint8Array(P * 3), depth / 250);
    world.add(person);
    const c2w = B("c2w");
    const path = new THREE.BufferGeometry().setFromPoints(Array.from({ length: S }, (_, i) =>
      new THREE.Vector3(c2w[i * 16 + 3], c2w[i * 16 + 7], c2w[i * 16 + 11])));
    world.add(new THREE.Line(path, new THREE.LineBasicMaterial({ color: 0xffd166 })));
    const fr = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff }));
    fr.geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(16 * 3), 3));
    world.add(fr);
    const asp = h.size[1] / h.size[0], fs = depth * 0.12;
    const corners = [[-asp, -1, 1.6], [asp, -1, 1.6], [asp, 1, 1.6], [-asp, 1, 1.6]].map((c) => c.map((x) => x * fs * 0.5));
    const trail = B("trail"), tf = h.trail_frames, nT = tf.length;
    const trailLine = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true }));
    world.add(trailLine);
    let skel = null, sk = null;
    if (h.aligned) {
      sk = B("skeleton");
      skel = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x7fd6ff }));
      skel.geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(BONES.length * 6), 3));
      world.add(skel);
    }
    // look at the scene's centre from the first camera's side, a little above, at 1.3 times its size
    const target = ext.center.clone().applyQuaternion(world.quaternion);
    const fwd = new THREE.Vector3(c2w[2], c2w[6], c2w[10]).applyQuaternion(world.quaternion).normalize();
    cam.position.copy(target).addScaledVector(fwd, -depth * 1.3).add(new THREE.Vector3(0, depth * 0.45, 0));
    cam.near = depth / 200; cam.far = depth * 50; cam.updateProjectionMatrix();
    orbit.target.copy(target);
    const nearest = (arr, f) => { let lo = 0, hi = arr.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < f) lo = m + 1; else hi = m; } return lo; };
    drawn = (f) => {
      const run_f = h.frames[0] + f;
      const s = nearest(h.frames, run_f);
      const pa = person.geometry.attributes.position.array, ca = person.geometry.attributes.color.array;
      let n = 0;
      for (let i = 0; i < P; i++) {
        const o = (s * P + i) * 3;
        if (!Number.isFinite(pp[o])) continue;
        pa[n * 3] = pp[o]; pa[n * 3 + 1] = pp[o + 1]; pa[n * 3 + 2] = pp[o + 2];
        ca[n * 3] = pc[o]; ca[n * 3 + 1] = pc[o + 1]; ca[n * 3 + 2] = pc[o + 2];
        n++;
      }
      person.geometry.setDrawRange(0, n);
      person.geometry.attributes.position.needsUpdate = true;
      person.geometry.attributes.color.needsUpdate = true;
      // camera frustum at this time
      const M = new THREE.Matrix4().fromArray(Array.from(c2w.slice(s * 16, s * 16 + 16))).transpose();
      const apex = new THREE.Vector3().setFromMatrixPosition(M);
      const wc = corners.map((c) => new THREE.Vector3(...c).applyMatrix4(M));
      const fa = fr.geometry.attributes.position.array;
      let k = 0;
      const put = (v) => { fa[k++] = v.x; fa[k++] = v.y; fa[k++] = v.z; };
      wc.forEach((c) => { put(apex); put(c); });
      wc.forEach((c, i) => { put(c); put(wc[(i + 1) % 4]); });
      fr.geometry.attributes.position.needsUpdate = true;
      // rainbow trails over the last 2 s
      const ti = nearest(tf, run_f), t0 = nearest(tf, run_f - 2 * h.fps);
      const seg = [], col = [];
      for (const j of TRAIL_JOINTS) {
        for (let i = t0; i < ti; i++) {
          const a = (i * 17 + j) * 3, b = ((i + 1) * 17 + j) * 3;
          if (![trail[a], trail[b]].every(Number.isFinite)) continue;
          seg.push(trail[a], trail[a + 1], trail[a + 2], trail[b], trail[b + 1], trail[b + 2]);
          const c = new THREE.Color().setHSL(((i - t0) / Math.max(1, ti - t0)) * 0.8, 1, 0.55);
          col.push(c.r, c.g, c.b, c.r, c.g, c.b);
        }
      }
      trailLine.geometry.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
      trailLine.geometry.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
      if (skel) {
        const sf = h.skeleton_frames, si = Math.min(nearest(sf, run_f), sf.length - 1), q = skel.geometry.attributes.position.array;
        BONES.forEach(([a, b], i) => { for (let kk = 0; kk < 3; kk++) { q[i * 6 + kk] = sk[(si * 17 + a) * 3 + kk]; q[i * 6 + 3 + kk] = sk[(si * 17 + b) * 3 + kk]; } });
        skel.geometry.attributes.position.needsUpdate = true;
      }
    };
  }

  async function show(i, tStart) {
    const p = parts[i];
    note.textContent = "Loading " + p.label + " ...";
    const d = await loadPart(p);
    if (cur !== i) {
      clear();
      cur = i;
      (cfg.mode === "3d" ? build3d : build4d)(d);
      partSel.value = String(i);
    }
    t = tStart ?? p.t0;
    note.textContent = cfg.mode === "4d" ? (d.head.aligned ? "Skeleton placed in the scene."
      : "Skeleton not placed: " + (d.head.align_reason || "no alignment")) : "";
    seekVideo(t);
    frame();
  }

  function frame() {
    const p = parts[cur];
    if (!p || !drawn) return;
    const f = Math.max(0, Math.min(p.n - 1, Math.round((t - p.t0) * p.fps)));
    drawn(f);
    slider.value = String(Math.round(((t - p.t0) / Math.max(1e-6, p.t1 - p.t0)) * 1000));
    label.textContent = clock(t);
  }

  let timer = null;
  function tick() {
    if (!playing) return;
    const now = performance.now();
    const dt = last ? (now - last) / 1000 : 0;
    last = now;
    // the video leads while it plays and moves; if it stalls (loading, seeking) for 0.5 s the own clock goes on
    const vt = video ? video.currentTime : null;
    if (vt !== null && vt !== lastVt) { lastVt = vt; vtSince = now; }
    const videoMoves = video && !video.paused && !video.seeking && now - vtSince < 500;
    if (videoMoves && Math.abs(vt - t) < 1.0) t = vt;
    else t += dt * parseFloat(speed.value);
    if (t > parts[cur].t1) {
      if (cur + 1 < parts.length) { show(cur + 1).then(() => { if (video && playing) video.play().catch(() => {}); }); }
      else { setPlaying(false); t = parts[cur].t1; }
    }
    frame();
  }

  function setPlaying(on) {
    playing = on;
    play.textContent = on ? "Pause" : "Play";
    last = 0;
    if (video) {
      video.playbackRate = parseFloat(speed.value);
      if (on) { if (Math.abs(video.currentTime - t) > 0.2) seekVideo(t); video.play().catch(() => {}); }
      else video.pause();
    }
    clearInterval(timer);
    if (on) timer = setInterval(tick, 30);      // the clock runs on a timer; drawing stays on animation frames
  }

  play.addEventListener("click", () => setPlaying(!playing));
  speed.addEventListener("change", () => { if (video) video.playbackRate = parseFloat(speed.value); });
  slider.addEventListener("input", () => {
    const p = parts[cur];
    t = p.t0 + (parseInt(slider.value) / 1000) * (p.t1 - p.t0);
    seekVideo(t);
    frame();
  });
  partSel.addEventListener("change", () => show(parseInt(partSel.value)));
  allAtOnce?.addEventListener("change", () => frame());
  if (video) {
    video.addEventListener("play", () => { if (!playing) setPlaying(true); });
    video.addEventListener("pause", () => { if (playing && !video.ended) setPlaying(false); });
    video.addEventListener("loadedmetadata", () => seekVideo(t));
    video.addEventListener("seeked", () => {
      if (seekTarget !== null && Math.abs(video.currentTime - seekTarget) < 0.3) { seekTarget = null; return; }
      if (Math.abs(video.currentTime - t) < 0.3) return;
      t = video.currentTime;
      const i = parts.findIndex((p) => t >= p.t0 && t <= p.t1);
      if (i >= 0 && i !== cur) show(i, t); else frame();
    });
  }
  const loop = () => {
    if (!root.isConnected) {                   // replaced (another run or view): stop for good
      clearInterval(timer);
      if (video) video.pause();
      renderer.dispose();
      return;
    }
    if (!root.offsetParent) { requestAnimationFrame(loop); return; }      // hidden panel: no drawing
    orbit.update();
    pxScale.value = renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2));
    renderer.render(scene, cam);
    // paused: keep the video on the player's time (a seek may have been lost while the video was loading)
    if (video && !playing && !video.seeking && video.readyState >= 1 && Math.abs(video.currentTime - t) > 0.3) seekVideo(t);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  show(0);
}
