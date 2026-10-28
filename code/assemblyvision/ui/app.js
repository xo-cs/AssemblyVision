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
}
