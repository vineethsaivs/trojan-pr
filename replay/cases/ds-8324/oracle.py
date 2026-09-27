# Oracle for deepspeedai/DeepSpeed#8324: FLOPs of a binary elementwise op (torch.mul/add)
# must equal the number of elements in the broadcast result.
# Usage: python oracle.py   (fetches profiler.py at each SHA via read-only `gh api`, cached here)
import ast, base64, json, os, subprocess, sys, torch

REPO, PATH = "deepspeedai/DeepSpeed", "deepspeed/profiling/flops_profiler/profiler.py"
SHAS = [("pre-intro (parent of #1591)", "7f58853c2e181604d2e5f5ae308524788ad0cd28"),
        ("introducing #1591", "082f392a939896e2fef782eabee2200b5e340902"),
        ("fix parent", "7fabe4735f6d59834ac576c79ef69f50004690e4"),
        ("fix #8324", "b8bfd260465b05a22009c0d23a17646cf57fd156")]
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(sha):
    cache = os.path.join(HERE, f"src_{sha[:7]}.py")
    if not os.path.exists(cache):
        out = subprocess.check_output(["gh", "api", f"repos/{REPO}/contents/{PATH}?ref={sha}"])
        open(cache, "wb").write(base64.b64decode(json.loads(out)["content"]))
    return open(cache).read()


def extract(src):
    """Exec only _prod and _elementwise_flops_compute from the file (no deepspeed import)."""
    keep = [n for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name in ("_prod", "_elementwise_flops_compute")]
    if not any(n.name == "_elementwise_flops_compute" for n in keep):
        return None
    ns = {"torch": torch}
    exec(compile(ast.Module(body=keep, type_ignores=[]), PATH, "exec"), ns)
    return ns["_elementwise_flops_compute"]


def broadcast_numel(a, b):
    """The math, restated: right-align shapes, pad the shorter on the LEFT with 1s,
    each dim pair must match or one side is 1, result dim is the larger. Count = product."""
    n = max(len(a), len(b))
    a, b = (1,) * (n - len(a)) + tuple(a), (1,) * (n - len(b)) + tuple(b)
    count = 1
    for x, y in zip(a, b):
        assert x == y or 1 in (x, y), f"not broadcastable: {a} {b}"
        count *= max(x, y)
    return count


CASES = [((2, 3, 4), (4,)),              # minimal: activation * bias
         ((8,), (2, 8)),
         ((3, 1), (5,)),
         ((32, 128, 4096), (4096,)),     # activation * per-feature weight (128x over when buggy)
         ((32, 128, 1), (4096,)),        # 32x under when buggy
         ((2, 3, 4), (2, 3, 4)),         # equal rank control
         ((4, 1, 7), (1, 5, 7)),         # equal rank control
         ((16,), (32, 16))]              # rank mismatch that is coincidentally right when buggy

# Tolerance: exact integer equality. Counting elements is integer arithmetic with no
# rounding, so any difference is a bug (the old 5% tolerance on model totals hid this).
verdicts = {}
for label, sha in SHAS:
    f = extract(fetch(sha))
    print(f"== {label} {sha[:7]}")
    if f is None:
        print("   N/A: _elementwise_flops_compute does not exist (torch.mul/add not counted at this SHA)")
        verdicts[label] = "N/A"
        continue
    bad = 0
    for a, b in CASES:
        want = broadcast_numel(a, b)
        x, y = torch.randn(a), torch.randn(b)
        assert want == torch.mul(x, y).numel()  # cross-check the restated math against a real execution
        got = f(x, y)[0]
        swapped = f(y, x)[0]                     # metamorphic: operand order must not matter
        padded = f(x, y.reshape((1,) * max(0, len(a) - len(b)) + tuple(b)))[0]  # nor left 1-padding
        ok = got == swapped == padded == want
        bad += not ok
        print(f"   {str(a):>16} op {str(b):<10} want={want:>11,} got={got:>13,} swap={swapped:>13,} "
              f"pad={padded:>13,} ratio={got / want:.4g} {'PASS' if ok else 'FAIL'}")
    verdicts[label] = "PASS" if bad == 0 else f"FAIL ({bad}/{len(CASES)} cases)"
    print(f"   -> {verdicts[label]}")

print("\nSUMMARY:", json.dumps(verdicts))
expected = verdicts["introducing #1591"].startswith("FAIL") and verdicts["fix #8324"] == "PASS"
print("ORACLE BEHAVES AS EXPECTED (fails on bug, passes on fix):", expected)
sys.exit(0 if expected else 1)
