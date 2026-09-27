"""Oracle for unsloth-zoo #1240: GRPO reverse KL k3(d) = e^d - d - 1 must be >= 0 and accurate.

The math, restated (not copied from either implementation):
  For any real d = log p_ref - log p_policy, k3(d) = e^d - d - 1 >= 0 (exp is convex, e^d >= 1 + d),
  and k3(d) = d^2/2 + d^3/6 + O(d^4). The reference is float64 expm1(d) - d on the float32 values
  the function actually sees. Checks: mean_kl >= 0 (exact, no tolerance) and mean_kl close to it.

Usage: python oracle.py [SHA ...]   (default: pre-intro, intro, fix parent, fix)
"""
import ast, math, sys, urllib.error, urllib.request
import torch

EPS32 = torch.finfo(torch.float32).eps  # 1.19e-7
URL = "https://raw.githubusercontent.com/unslothai/unsloth-zoo/{}/unsloth_zoo/rl_replacements.py"
SHAS = {
    "9119c88b43bb742f7211c594c793ca4c4c16ad43": "pre-intro (parent of PR #45)",
    "86232127f90830bd7b33cc2ec968c5079d91f525": "intro (PR #45)",
    "17d7f22fd8d21b2981e7948cf154dbf489a975bd": "fix parent (buggy, 2026-09)",
    "ba721e43c0573a13d1ca74fda57b1dccb8113185": "fix (PR #1240)",
}
PROBES = (1e-3, 1e-4, 1e-6, -1e-4)


def load(sha):
    try:
        src = urllib.request.urlopen(URL.format(sha), timeout=30).read().decode()
    except urllib.error.HTTPError as e:
        return None, f"file absent at this SHA (HTTP {e.code})"
    fn = next((n for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef) and n.name == "grpo_compute_loss"), None)
    if fn is None:
        return None, "grpo_compute_loss absent at this SHA"
    fn.decorator_list = []  # drop @torch.compile: eager is the math under test
    ns = {"torch": torch, "math": math}
    exec(compile(ast.Module([fn], []), sha, "exec"), ns)
    return (ns["grpo_compute_loss"], fn.args.args[0].arg), None


def run(fn, first_arg, d):
    """One sequence of 2 tokens with log p_ref - log p_new = d. Advantages 0, so loss == beta*KL."""
    if first_arg == "ref":  # 2025-05 onward: takes per-token logps
        ref = torch.full((1, 2), 0.0 if d > 0 else d)  # valid logps (<= 0); both subtractions exact
        new = torch.full((1, 2), -d if d > 0 else 0.0)
        out = fn(ref, new, new, None, torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2), 1.0, torch.zeros(1))
        delta, d_err = ref.double() - new.double(), 0.0  # fn sees exactly these float32 values
    else:  # 2025-02 original: takes logits (old_logits = reference model), vocab of 2
        new_logits = torch.zeros(1, 2, 2)
        ref_logits = new_logits.clone()
        ref_logits[..., 0] = 2 * d  # d(log softmax)/dx = 1/2 at x = 0, so delta ~= d
        out = fn(ref_logits, new_logits, torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2), 1.0, torch.zeros(1))
        lp = lambda z: z.double()[..., 0] - torch.logsumexp(z.double(), -1)
        delta = lp(ref_logits) - lp(new_logits)
        # fn computes each logp (~ -0.69) in float32: gather, exp, sum, log, subtract, a few eps each.
        # Budget 8*eps32 absolute on d; it moves k3 by |k3'(d)| * 8*eps32 = |expm1(d)| * 8*eps32.
        d_err = 8 * EPS32
    exact = float((torch.expm1(delta) - delta).mean())
    ad = float(delta.abs().max())
    # Tolerance from arithmetic:
    #  1e-2 * exact: slack for the d^3 and higher terms and the mean, far below the bug's error.
    #  2*eps32*|d|: float32 expm1(d) - d carries an absolute error about the spacing of d and expm1(d).
    #   That is ~12% of d^2/2 at d=1e-6 but 0.2% at d=1e-4; a pure relative bound would fail the fix.
    #  |expm1(d)|*d_err: propagated input rounding (logits path only).
    tol = 1e-2 * exact + 2 * EPS32 * ad + math.expm1(ad) * d_err
    return float(out[2]), exact, tol


def check(sha, label):
    loaded, why = load(sha)
    if loaded is None:
        print(f"{sha[:12]} {label}: N/A, {why}")
        return None
    ok = True
    for d in PROBES:
        got, exact, tol = run(*loaded, d)
        nonneg, close = got >= 0.0, abs(got - exact) <= tol
        ok &= nonneg and close
        print(f"  d={d:+.0e}  mean_kl={got:+.3e}  exact={exact:.3e}  tol={tol:.1e}  nonneg={nonneg}  close={close}")
    print(f"{sha[:12]} {label}: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    shas = sys.argv[1:] or list(SHAS)
    res = {s: check(s, SHAS.get(s, "")) for s in shas}
    # Expected: intro and fix parent FAIL, fix PASS, pre-intro N/A.
    expect = {"86232127f90830bd7b33cc2ec968c5079d91f525": False,
              "17d7f22fd8d21b2981e7948cf154dbf489a975bd": False,
              "ba721e43c0573a13d1ca74fda57b1dccb8113185": True}
    bad = [s[:12] for s, want in expect.items() if s in res and res[s] is not want]
    print("ORACLE BEHAVES AS EXPECTED" if not bad else f"UNEXPECTED: {bad}")
    sys.exit(1 if bad else 0)
