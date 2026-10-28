// Comparison mode time rule (inlined into app.js by app.py; tested with node).
// lead: raw footage time, follow: overlay time, offset: raw time of the overlay's
// first frame. Returns the time to seek the overlay to, or null when it is
// already within tol seconds of the raw footage.
function avSyncTime(lead, follow, offset, tol) {
  const target = Math.max(0, lead - (offset || 0));
  return Math.abs(follow - target) > (tol === undefined ? 0.1 : tol) ? target : null;
}
function avClock(s) {
  s = Math.max(0, Math.floor(s || 0));
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}
