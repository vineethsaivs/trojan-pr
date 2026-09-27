"""Oracle replay for sentence-transformers #4019 (BatchHardSoftMarginTripletLoss overflow).

Fetches the real loss modules at each SHA (gh api, cached in src_<sha>/), execs them with only
the package-internal import lines removed, calls the public forward() with a pass-through
embedder, and checks the result against a float64 restatement of the math.

Usage: venv/bin/python oracle.py
"""
import base64, json, math, os, re, subprocess, sys
import torch

REPO = "huggingface/sentence-transformers"
HERE = os.path.dirname(os.path.abspath(__file__))
OLD = ("sentence_transformers/losses/", "BatchHardTripletLoss.py", "BatchHardSoftMarginTripletLoss.py")
NEW = ("sentence_transformers/sentence_transformer/losses/", "batch_hard_triplet.py", "batch_hard_soft_margin_triplet.py")
SHAS = [  # (sha, role, layout, expected verdict)
    ("c9c8dfccdeac53f0592bb64da68df85fe789547a", "pre-intro (parent of e7f7cbe)", OLD, "NO-IMPL"),
    ("e7f7cbe4c4e694ec1457daa6e56b5bbf856d9719", "introducing commit, PR #299", OLD, "FAIL"),
    ("af1f4efcbbd3dc990b7e9b2ae281fb0843bc699e", "fix parent (main, 2026)", NEW, "FAIL"),
    ("f3db086191749a0707547acd1943236871e8195c", "fix commit, PR #4019", NEW, "PASS"),
]


def fetch(sha, path):
    local = os.path.join(HERE, f"src_{sha}", os.path.basename(path))
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        out = subprocess.check_output(["gh", "api", f"repos/{REPO}/contents/{path}?ref={sha}"])
        open(local, "wb").write(base64.b64decode(json.loads(out)["content"]))
    return open(local).read()


def load_loss_class(sha, layout):
    d, base, soft = layout
    ns = {"__name__": f"st_{sha[:7]}"}
    for f in (base, soft):  # base module first so the soft-margin module finds BatchHardTripletLoss in ns
        src = re.sub(r"(?m)^from (sentence_transformers\S*|\.\S*) import .*$", "", fetch(sha, d + f))
        exec(compile(src, f"{sha[:7]}/{f}", "exec"), ns)
    return ns["BatchHardSoftMarginTripletLoss"]


def reference(emb, labels):
    """The math, restated in float64 (never the implementation):
    L = (1/B) * sum_a softplus(max_{p!=a, y_p=y_a} ||e_a-e_p|| - min_{y_n!=y_a} ||e_a-e_n||),
    positive term 0 when an anchor has no positive, softplus(x) = max(x,0) + log1p(exp(-|x|))."""
    dist = lambda i, j: math.sqrt(sum((a - b) ** 2 for a, b in zip(emb[i], emb[j])))
    softplus = lambda x: max(x, 0.0) + math.log1p(math.exp(-abs(x)))
    B = len(emb)
    return sum(softplus(max([dist(a, p) for p in range(B) if p != a and labels[p] == labels[a]], default=0.0)
                        - min(dist(a, n) for n in range(B) if labels[n] != labels[a])) for a in range(B)) / B


# Tolerance from arithmetic: every input is an integer or 0.5 with squares <= 1004^2 < 2^24, so the
# dot products, squared distances and sqrt (IEEE sqrt of a perfect square is exact) carry no rounding.
# What is left is the subtraction, softplus (a few ulp) and the mean over B <= 4 terms: under ~10 ulp,
# i.e. ~6e-7 relative in fp32 and ~1e-15 in fp64. Bounds: fp32 rtol 1.3e-6 + atol 1e-5 (torch.testing
# defaults), fp64 rtol 1e-12. The decisive check is finiteness, which needs no tolerance.
RTOL = {torch.float32: 1.3e-6, torch.float64: 1e-12}
ATOL = 1e-5
CASES = [  # gap x = d(a,p) - d(a,n) for the worst anchor; fp32 exp overflows at x > ln(fp32 max) = 88.72
    ("benign, gap 0.5", [[0.0], [1.0], [0.5]], [0, 0, 1], torch.float32),
    ("gap 88, just under cliff", [[0.0], [89.0], [1.0]], [0, 0, 1], torch.float32),
    ("gap 99, minimal repro", [[0.0], [100.0], [1.0]], [0, 0, 1], torch.float32),
    ("PR #4019 test, fp32", [[0.0], [1000.0], [2.0], [1004.0]], [0, 0, 1, 1], torch.float32),
    ("PR #4019 test, fp64", [[0.0], [1000.0], [2.0], [1004.0]], [0, 0, 1, 1], torch.float64),
]


def check(cls, emb_list, labels_list, dtype):
    emb = torch.tensor(emb_list, dtype=dtype, requires_grad=True)
    loss_fn = cls(lambda feats: {"sentence_embedding": feats})  # pass-through "model"
    loss = loss_fn([emb], torch.tensor(labels_list))
    if loss is None:
        return None
    loss.backward()
    ref = reference(emb_list, labels_list)
    val, gfin = loss.item(), bool(torch.isfinite(emb.grad).all())
    ok = math.isfinite(val) and gfin and abs(val - ref) <= ATOL + RTOL[dtype] * abs(ref)
    return ok, val, ref, gfin


all_met = True
for sha, role, layout, expected in SHAS:
    cls = load_loss_class(sha, layout)
    print(f"\n== {sha[:7]}  {role}")
    results = [check(cls, e, l, dt) for _, e, l, dt in CASES]
    if any(r is None for r in results):
        verdict = "NO-IMPL"
        print("   batch_hard_triplet_soft_margin_loss is a `pass` stub (forward returns None): no control possible")
    else:
        for (name, *_), (ok, val, ref, gfin) in zip(CASES, results):
            print(f"   {name:26s} loss={val:<20.7g} ref={ref:<14.7g} grads_finite={gfin!s:5s} {'PASS' if ok else 'FAIL'}")
        verdict = "PASS" if all(r[0] for r in results) else "FAIL"
    all_met &= verdict == expected
    print(f"   VERDICT {verdict}  (expected {expected})")
print(f"\nALL EXPECTED OUTCOMES MET: {all_met}")
sys.exit(0 if all_met else 1)
