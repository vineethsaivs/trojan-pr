// Stage keys: S stage mode, P replay, R and T reserved for the arena cards.
(function () {
  var html = document.documentElement;

  document.addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return;
    var t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    var k = e.key.toLowerCase();
    if (k === "s") {
      var on = html.classList.toggle("stage");
      try { localStorage.setItem("pl-stage", on ? "1" : "0"); } catch (_) {}
    } else if (k === "p") {
      location.href = "/replay";
    } else if (k === "r" || k === "t") {
      // ponytail: reveal (R) and 5 s timer (T) land with the arena cards; no cards, no action.
      if (!document.querySelector(".arena-card")) return;
    }
  });

  // One click per run: the POST starts real cloud work.
  document.addEventListener("submit", function (e) {
    var b = e.target.querySelector("button[type=submit]");
    if (b) { b.disabled = true; b.textContent = "Starting run"; }
  });

  // When a live run finishes, bring the verdict band into view.
  document.addEventListener("htmx:afterSwap", function (e) {
    var band = document.querySelector("#run .band");
    if (band && !document.querySelector("#run[hx-get]")) band.scrollIntoView({ block: "nearest" });
  });
})();
