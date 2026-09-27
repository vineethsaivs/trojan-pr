# Oracle for deepspeedai/DeepSpeed #8533 (bug introduced by #7953): Muon orthogonalization must be
# scale invariant and non-degenerate.
#
# The math (restated, not copied from either implementation):
#   Muon replaces a gradient G = U S V^T by an approximation of its polar factor U V^T. Newton-Schulz
#   first divides by ||G||_F, so the result depends only on the direction of G:
#     (R1) f(c*G) == f(G) for every c > 0 with c*G representable in G's dtype.
#     (R2) f(G) is finite and non-zero whenever G is finite and non-zero.
#     (R3) f(G) points along U V^T (computed here independently with a float64 SVD).
#
# Tolerance, from arithmetic:
#   fp16 unit roundoff u = 2^-11 = 4.9e-4. The normalized input carries relative error ~u, and 5 NS
#   steps (quintic slope <= a = 3.4445 near 0, < 1 near the fixed point) keep the Frobenius-weighted
#   error at a few 1e-3 (measured: fp16 fixed code 1 - cos ~1e-5; bf16 control, u = 2^-8, 1 - cos
#   ~1.2e-3; measured cos(f(G), UV^T) ~0.985 to 0.987). We require
#   cos(f(cG), f(G)) >= 0.99, i.e. relative error <= ~0.14, two orders of magnitude of slack.
#   (R3): Muon's coefficients (3.4445, -4.7750, 2.0315) deliberately stop at U S' V^T with S' in
#   roughly [0.5, 1.5], not exactly U V^T, so cos(f(G), UV^T) is < 1 by design. S' spread over
#   [0.5, 1.5] still gives cos >= ~0.9; we require >= 0.9.
#   The bug gives an exact zero matrix (cos = 0) or NaN, so no reasonable tolerance can hide it.
#
# Usage: python oracle.py   (fetches deepspeed/runtime/zero/muon/original_muon.py at each SHA with
# read-only `gh api`, caches it under src/, runs the target function eagerly on CPU).
import ast, base64, json, os, subprocess, sys, types
import torch

REPO, PATH = "deepspeedai/DeepSpeed", "deepspeed/runtime/zero/muon/original_muon.py"
SHAS = [
    ("pre-intro  5b441027 (parent of #7953)", "5b441027b76938e4c31c0b316a99ce23b028e016", "pass"),
    ("introduce  8a77f381 (#7953 merge)", "8a77f381a4d8f3bf65747ac571b94324eaa4a29a", "fail"),
    ("fix-parent 0d4aa274", "0d4aa274731857561ebb479552d4223a99a0cedc", "fail"),
    ("fix        1663b5b5 (#8533 merge)", "1663b5b59aaf58ec6accb1aca326e120e6b1c279", "pass"),
]
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(sha):
    cache = os.path.join(HERE, "src", f"{sha}.py")
    if not os.path.exists(cache):
        out = subprocess.check_output(["gh", "api", f"repos/{REPO}/contents/{PATH}?ref={sha}", "--jq", ".content"])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        open(cache, "wb").write(base64.b64decode(out))
    return open(cache).read()


def load(src):
    """Exec the orthogonalizer from the file, eagerly (strip @compiler.compile), on a GPU-like accelerator
    stub (fp16/bf16 supported). Gram NS if the SHA has it, else standard NS (pre-#7953 default)."""
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    target = "zeropower_via_gram_newtonschulz" if "zeropower_via_gram_newtonschulz" in fns else "zeropower_via_newtonschulz5"
    keep = [fns[k] for k in (target, "ns_compute_dtype") if k in fns]
    for n in keep:
        n.decorator_list = []  # torch.compile/inductor on CPU fuses away the fp16 rounding and hides the bug
    acc = types.SimpleNamespace(is_fp16_supported=lambda: True, is_bf16_supported=lambda: True)
    g = {"torch": torch, "get_accelerator": lambda: acc}
    exec(compile(ast.Module(keep, []), "original_muon.py", "exec"), g)
    return target, g[target]


def cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))  # 0/0 -> nan, which fails every >= check


def polar(G):
    U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    return U @ Vh


G = torch.randn(32, 128, generator=torch.Generator().manual_seed(0))  # wide, so the Gram path runs
# 1e4*G: max |elem| ~4e4 fits fp16 (65504) but ||1e4*G||_F ~6.4e5 does not. 1e5*G is a normal fp32 grad
# under an ordinary loss scale. fp16 input only at 1e4: (1e5*G).half() is already inf before the call.
CASES = [(torch.float32, 1e4), (torch.float32, 1e5), (torch.float16, 1e4)]

all_ok = True
for label, sha, expect in SHAS:
    target, f = load(fetch(sha))
    base = f(G, 5)
    P = polar(G)
    ok, rows = True, []
    for dt, c in CASES:
        out = f((c * G).to(dt), 5)
        fin = bool(torch.isfinite(out).all())
        nrm = float(out.float().norm())
        c_inv, c_pol = cos(out, base), cos(out, P)
        good = fin and nrm > 0 and c_inv >= 0.99 and c_pol >= 0.9
        ok &= good
        rows.append(f"    in={str(dt)[6:]:8s} c={c:.0e}  finite={fin!s:5s} ||f||={nrm:7.3f}  "
                    f"cos(f(cG),f(G))={c_inv:+.5f}  cos(f(cG),UV^T)={c_pol:+.4f}  {'ok' if good else 'VIOLATED'}")
    verdict = "PASS" if ok else "FAIL"
    match = (verdict == "PASS") == (expect == "pass")
    all_ok &= match
    print(f"{verdict}  {label}  [{target}]  base: ||f(G)||={float(base.norm()):.3f} cos(f(G),UV^T)={cos(base, P):+.4f}"
          f"  expected={expect.upper()} {'(as expected)' if match else '(UNEXPECTED)'}")
    print("\n".join(rows))

print(f"\nORACLE {'BEHAVES AS EXPECTED' if all_ok else 'MISMATCH'}: fails on buggy SHAs, passes on controls")
sys.exit(0 if all_ok else 1)
