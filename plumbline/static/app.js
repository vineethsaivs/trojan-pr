// Stage keys: S stage mode, P replay; on /arena T timer, R reveal, 0 reset.
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
    } else if (k === "r" || k === "t" || k === "0") {
      if (!document.querySelector(".arena-card")) return;
      if (k === "r") reveal(true); else if (k === "0") reveal(false); else timer();
    }
  });

  // Arena (PLAN 6.6): T starts a 5 s count, R shows the stored verdicts, 0 resets.
  var tick = null;
  function reveal(on) {
    var a = document.getElementById("arena"), b = document.getElementById("reveal"), t = document.getElementById("timer");
    if (!a) return;
    a.classList.toggle("revealed", on);
    if (b) b.textContent = on ? "Hide the verdicts" : "Reveal the verdicts";
    clearInterval(tick);
    if (t) { t.hidden = true; t.classList.remove("done"); }
    var band = on && document.querySelector(".is-trojan .card-band");
    if (band) band.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }
  function timer() {
    var t = document.getElementById("timer"), n = 5;
    if (!t) return;
    clearInterval(tick);
    t.hidden = false; t.classList.remove("done"); t.textContent = n;
    tick = setInterval(function () {
      n -= 1;
      t.textContent = n;
      if (n <= 0) { clearInterval(tick); t.classList.add("done"); }
    }, 1000);
  }
  document.addEventListener("click", function (e) {
    if (e.target && e.target.id === "reveal") {
      reveal(!document.getElementById("arena").classList.contains("revealed"));
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
