"""Oracle for unsloth #10842: Q-GaLore's QGaLoreAdamW8bit advanced the Adam step counter twice per step.

The oracle restates Adam (Kingma and Ba 2015, Algorithm 1) and the GaLore refresh schedule. It never
reads or copies Unsloth's or bitsandbytes' update code; it only drives the optimizer through its
public API (construct, step, state) and counts calls to the projector's SVD helper.

  A. Closed form. Constant gradient g: m_hat = g and v_hat = g^2 exactly at every t, so each update
     is lr*g/(|g|+eps). After N steps from p0: p = p0 - N*lr*g/(|g|+eps).
  B. Reference. Varying gradients: float64 Adam written from the paper must match the optimizer.
  C. Counter. After N step() calls, state["step"] == N (exact integer).
  D. Schedule. With update_proj_gap=G, K steps refresh the projection ceil(K/G) times (exact integer).

Tolerance for A and B: absolute 1e-5. Params under min_8bit_size=4096 elements keep fp32 optimizer
state, so there is no 8-bit quantization noise. fp32 rounding on p~1 is ~6e-8 per step, and on an
update of size ~lr=0.1 it is ~1e-8, so 5 steps accumulate well under 1e-6. The bug moves p by ~3e-2,
over 1000x the tolerance.

Usage: python oracle.py   (torch, bitsandbytes>=0.50 for CPU optimizer kernels, gh CLI for sources)
"""
import importlib, math, os, subprocess, sys
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO, FILES = "unslothai/unsloth", ("q_galore_adamw.py", "q_galore_projector.py")
SHAS = [
    ("pre-intro 11606c50 (parent of #4511)", "11606c502587d318ba21c7c2325d31e043f1cc0e"),
    ("intro 45d0a343 (PR #4511 merge)", "45d0a343b5a46c7a711db40be6edc3381ead0733"),
    ("fix-parent a092906c", "a092906c98a1e9ffbc809e2bfa1261ffe3d507fa"),
    ("fix 677a988d (PR #10842)", "677a988dd5a9bd57dc7036e3c0dec102c9c1b6ad"),
]
TOL = 1e-5
B1, B2, EPS = 0.9, 0.999, 1e-8


def fetch(sha):
    """Put the two module files at `sha` into src_<sha8>/qg. Returns None if absent at that SHA."""
    pkg = os.path.join(HERE, f"src_{sha[:8]}", "qg")
    if not os.path.exists(os.path.join(pkg, FILES[0])):
        blobs = []
        for f in FILES:
            r = subprocess.run(["gh", "api", f"repos/{REPO}/contents/unsloth/optimizers/{f}?ref={sha}",
                                "-H", "Accept: application/vnd.github.raw"], capture_output=True)
            if r.returncode:
                return None
            blobs.append(r.stdout)
        os.makedirs(pkg, exist_ok=True)
        open(os.path.join(pkg, "__init__.py"), "w").close()
        for f, b in zip(FILES, blobs):
            open(os.path.join(pkg, f), "wb").write(b)
    for k in [k for k in sys.modules if k == "qg" or k.startswith("qg.")]:
        del sys.modules[k]
    sys.path.insert(0, os.path.dirname(pkg))
    try:
        return importlib.import_module("qg.q_galore_adamw")
    finally:
        sys.path.pop(0)


def adam_paper(p0, grads, lr):
    """Adam, Algorithm 1 of the paper, float64, no weight decay. t starts at 1 and advances once."""
    p, m, v = p0, 0.0, 0.0
    for t, g in enumerate(grads, start=1):
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        p -= lr * (m / (1 - B1 ** t)) / (math.sqrt(v / (1 - B2 ** t)) + EPS)
    return p


def run_steps(mod, grads, lr=0.1):
    p = nn.Parameter(torch.ones(2))
    opt = mod.QGaLoreAdamW8bit([p], lr=lr, betas=(B1, B2), eps=EPS, weight_decay=0.0)
    for g in grads:
        p.grad = torch.full((2,), g)
        opt.step()
    return p[0].item(), opt.state[p]["step"]


def refreshes(mod, steps=8, gap=4):
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(16, 8))
    group = {"params": [w], "lr": 1e-3, "weight_decay": 0.0, "rank": 2, "update_proj_gap": gap,
             "scale": 1.0, "proj_type": "std", "quant": False, "cos_threshold": -1.0,
             "gamma_proj": 1.0, "queue_size": 5}
    opt = mod.QGaLoreAdamW8bit([group], lr=1e-3, weight_decay=0.0)
    cls = mod.GaLoreProjector
    raw = cls.__dict__["_compute_orthogonal"]
    fn = raw.__func__ if isinstance(raw, staticmethod) else raw
    calls = [0]
    def counted(*a, **k):
        calls[0] += 1
        return fn(*a, **k)
    cls._compute_orthogonal = staticmethod(counted) if isinstance(raw, staticmethod) else counted
    try:
        for _ in range(steps):
            w.grad = torch.randn(16, 8)
            opt.step()
    finally:
        cls._compute_orthogonal = raw
    return calls[0], -(-steps // gap)


def verdict(ok):
    return "PASS" if ok else "FAIL"


const = [0.3] * 3
varying = [0.3, -0.1, 0.5, 0.2, -0.4]
want_a = 1.0 - len(const) * 0.1 * 0.3 / (0.3 + EPS)
want_b = adam_paper(1.0, varying, 0.1)
print(f"bitsandbytes {__import__('bitsandbytes').__version__}, torch {torch.__version__}, tol {TOL}")
print(f"A want p={want_a:.6f} (3 const steps) | B want p={want_b:.6f} (5 varying steps) | C want step=N | D want ceil(K/G)")
for name, sha in SHAS:
    mod = fetch(sha)
    if mod is None:
        print(f"{name:38s} N/A: unsloth/optimizers/q_galore_adamw.py does not exist at this SHA")
        continue
    pa, ca = run_steps(mod, const)
    pb, cb = run_steps(mod, varying)
    got_r, want_r = refreshes(mod)
    ok = [abs(pa - want_a) <= TOL, abs(pb - want_b) <= TOL, ca == len(const) and cb == len(varying),
          got_r == want_r]
    print(f"{name:38s} {'PASS' if all(ok) else 'FAIL'}  "
          f"A p={pa:.6f} err={pa - want_a:+.2e} {verdict(ok[0])} | "
          f"B p={pb:.6f} err={pb - want_b:+.2e} {verdict(ok[1])} | "
          f"C step={ca},{cb} want 3,5 {verdict(ok[2])} | "
          f"D svd={got_r} want {want_r} {verdict(ok[3])}")
