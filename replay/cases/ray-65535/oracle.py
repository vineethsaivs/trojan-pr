# Oracle for ray-project/ray #65535 (bug introduced by #28187, d59d8c6).
# Math restated from the docstring ("rounded to an integer increment of q"), not from either implementation:
#   1. grid membership: y / q is an integer (equivalently q*round(y/q) == y, re-quantizing is a no-op)
#   2. bounds: lo <= y <= hi for bounded domains
# Control: the same domain at q=2 must pass at every SHA (shows the check is not trivially failing).
# Tolerance: |y/q - round(y/q)| <= 1e-9 * max(1, |y/q|). Correct outputs are np.round(x/q)*q, so y/q
# carries at most ~1e-15 relative float error; buggy outputs are continuous, typical miss ~0.25 on y/q,
# about 8 orders of magnitude above the tolerance. A single off-grid sample is a FAIL.
import base64, json, subprocess, sys, types, importlib.util, pathlib
import numpy as np

HERE = pathlib.Path(__file__).parent
REPO, PATH = "ray-project/ray", "python/ray/tune/search/sample.py"
SHAS = [
    ("pre-intro", "e63b405b61b30984bf16a3ccc2a6332e73631347", "PASS"),
    ("intro #28187", "d59d8c6268ef2f0bef49c229c208fcdaed21362a", "FAIL"),
    ("fix-parent", "5b114d0bb01577ab7ef6635ece1929431b170b9f", "FAIL"),
    ("fix #65535", "e936a4b09490ee881265004fba5acb2dbd5a5a74", "PASS"),
]
N, SEED = 1000, 0

# Minimal stub for the only ray import sample.py needs.
ann = types.ModuleType("ray.util.annotations")
ann.PublicAPI = ann.DeveloperAPI = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)
ann.RayDeprecationWarning = type("RayDeprecationWarning", (DeprecationWarning,), {})
for name in ("ray", "ray.util"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["ray.util.annotations"] = ann


def fetch(sha):
    f = HERE / "fetched" / f"{sha}.py"
    if not f.exists():
        f.parent.mkdir(exist_ok=True)
        b64 = subprocess.check_output(["gh", "api", f"repos/{REPO}/contents/{PATH}?ref={sha}", "--jq", ".content"])
        f.write_bytes(base64.b64decode(b64))
    spec = importlib.util.spec_from_file_location(f"sample_{sha[:7]}", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def violations(ys, q, lo, hi):
    off = [y for y in ys if abs(y / q - round(y / q)) > 1e-9 * max(1.0, abs(y / q))]
    oob = [y for y in ys if (lo is not None and y < lo) or (hi is not None and y > hi)]
    return off, oob


results = {}
for label, sha, expect in SHAS:
    s = fetch(sha)
    cases = [  # (name, domain, q, lo, hi, is_control)
        ("quniform(-10,10,1)", s.quniform(-10, 10, 1), 1, -10, 10, False),
        ("qloguniform(1,100,1)", s.qloguniform(1, 100, 1), 1, 1, 100, False),
        ("qrandn(0,5,1)", s.qrandn(0, 5, 1), 1, None, None, False),
        ("quniform(-10,10,2) ctl", s.quniform(-10, 10, 2), 2, -10, 10, True),
    ]
    verdict, ctl_ok = "PASS", True
    for name, dom, q, lo, hi, ctl in cases:
        ys = [float(y) for y in dom.sample(size=N, random_state=np.random.RandomState(SEED))]
        off, oob = violations(ys, q, lo, hi)
        bad = bool(off or oob)
        if ctl:
            ctl_ok = not bad
        elif bad:
            verdict = "FAIL"
        eg = f" e.g. {off[0]:.4f}" if off else ""
        print(f"  {label:13s} {sha[:7]} {name:24s} off-grid {len(off):4d}/{N} out-of-bounds {len(oob)}{eg}")
    ok = verdict == expect and ctl_ok
    results[label] = (verdict, expect, ctl_ok)
    print(f"{label:13s} {sha[:7]}: {verdict} (expected {expect}, control {'PASS' if ctl_ok else 'FAIL'}) {'OK' if ok else 'MISMATCH'}")

all_ok = all(v == e and c for v, e, c in results.values())
print("ORACLE MATCHES HISTORY" if all_ok else "ORACLE DOES NOT MATCH HISTORY")
print(json.dumps({k: v[0] for k, v in results.items()}))
sys.exit(0 if all_ok else 1)
