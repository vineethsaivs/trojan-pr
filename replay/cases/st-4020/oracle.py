"""st-4020 oracle: huggingface/sentence-transformers OnlineContrastiveLoss hard-pair mining.

Runs the loss as it existed at each SHA (source fetched with `gh api ...contents?ref=SHA`,
target method pulled out with ast, exec'd with a stub self) and compares against an
independent restatement of the loss definition. Needs torch only.

Oracle (restates the math, not either implementation):
  P = positive-pair distances, N = negative-pair distances, m = margin.
  hard positives: p in P with p > min(N)   (only when N is empty: p > mean(P))
  hard negatives: n in N with n < max(P)   (only when P is empty: n < mean(N))
  loss = sum p^2 over hard P + sum relu(m - n)^2 over hard N
  d loss / d p = 2p for hard p, d loss / d n = -2 relu(m - n) for hard n, else 0.
  Metamorphic: adding an easy positive (below every negative distance) must not change the loss.

Tolerance: float64 sums of <= 4 squared terms of values in [0, 1] are exact to ~1e-15,
so rtol 1e-6, atol 1e-9 leaves 6+ orders of headroom. The bug gap is >= 0.16 absolute.
"""
import ast, base64, json, subprocess, sys, warnings
import torch, torch.nn.functional as F
from torch import Tensor

REPO = "huggingface/sentence-transformers"
OLD = "sentence_transformers/losses/OnlineContrastiveLoss.py"
NEW = "sentence_transformers/sentence_transformer/losses/online_contrastive.py"
SHAS = [  # (label, sha, path, method, expect)
    ("pre-intro c9baad9 (parent of intro)", "c9baad916f83924670220937dd68913338e3eabf", OLD, "forward", "absent"),
    ("intro 2900310 (2020-08-04)", "2900310707f3ed22faea08695cbe787c0e3b13f7", OLD, "forward", "FAIL"),
    ("fix-parent f3db086 (main before #4020)", "f3db086191749a0707547acd1943236871e8195c", NEW, "compute_loss_from_embeddings", "FAIL"),
    ("fix c5d2dc8 (#4020)", "c5d2dc851284af7fe49e14b5fe83530ed17b76ba", NEW, "compute_loss_from_embeddings", "PASS"),
]
RTOL, ATOL, M = 1e-6, 1e-9, 1.0


def fetch(sha, path):
    r = subprocess.run(["gh", "api", f"repos/{REPO}/contents/{path}?ref={sha}"], capture_output=True, text=True)
    return base64.b64decode(json.loads(r.stdout)["content"]).decode() if r.returncode == 0 else None


def extract(src, name):
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.returns = None
    for a in fn.args.args + fn.args.kwonlyargs:
        a.annotation = None
    ns = {"F": F, "torch": torch, "Tensor": Tensor, "warnings": warnings}
    exec(compile(ast.Module([fn], []), name, "exec"), ns)
    return ns[name]


class Stub:  # stands in for `self`: margin, |a-b| distance, model that echoes the embedding
    margin, _checked_labels = M, True
    distance_metric = staticmethod(lambda a, b: (a - b).abs().flatten())
    model = staticmethod(lambda f: {"sentence_embedding": f["sentence_embedding"]})


def run(fn, method, d, y):
    e = torch.tensor(d, dtype=torch.float64, requires_grad=True).unsqueeze(1)
    emb = [e, torch.zeros_like(e)]
    feats = [{"sentence_embedding": t} for t in emb] if method == "forward" else emb
    out = fn(Stub(), feats, torch.tensor(y))
    g = torch.autograd.grad(out, e, allow_unused=True)[0]
    return out.item(), [0.0] * len(d) if g is None else g.flatten().tolist()


def oracle(d, y):  # plain python, no torch, no reuse of the implementation
    P = [x for x, l in zip(d, y) if l == 1]
    N = [x for x, l in zip(d, y) if l == 0]
    hp = {i for i, (x, l) in enumerate(zip(d, y)) if l == 1 and x > (min(N) if N else sum(P) / len(P))}
    hn = {i for i, (x, l) in enumerate(zip(d, y)) if l == 0 and x < (max(P) if P else sum(N) / len(N))}
    loss = sum(d[i] ** 2 for i in hp) + sum(max(0.0, M - d[i]) ** 2 for i in hn)
    grad = [2 * x if i in hp else -2 * max(0.0, M - x) if i in hn else 0.0 for i, x in enumerate(d)]
    return loss, grad


close = lambda a, b: abs(a - b) <= ATOL + RTOL * abs(b)
CASES = [  # (distances, labels, note)
    ([0.7, 0.2], [1, 0], "1 pos + 1 neg, both hard"),
    ([0.7, 0.2, 0.6, 0.8], [1, 0, 0, 0], "1 pos, 3 neg: hard neg 0.6 must count"),
    ([0.7, 0.3, 0.1, 0.2], [1, 1, 1, 0], "3 pos, 1 neg: hard pos 0.3 must count"),
    ([0.1, 0.2, 0.9], [1, 0, 0], "1 easy pos: no neg is hard"),
    ([0.7, 0.4, 0.5, 0.9], [1, 1, 0, 0], "2 pos, 2 neg: no size-1 group (sanity)"),
]
META = (([0.7, 0.2], [1, 0]), ([0.7, 0.1, 0.2], [1, 1, 0]))  # add easy positive 0.1 < min(N)=0.2

ok_all = True
for label, sha, path, method, expect in SHAS:
    src = fetch(sha, path)
    print(f"== {label}  {path}@{sha[:7]}")
    if src is None:
        print(f"   file absent at this SHA (404): no pre-introduction version exists  [{'OK' if expect == 'absent' else 'UNEXPECTED'}]")
        ok_all &= expect == "absent"
        continue
    fn, fails = extract(src, method), 0
    for d, y, note in CASES:
        (got, g), (want, wg) = run(fn, method, d, y), oracle(d, y)
        ok = close(got, want) and all(close(a, b) for a, b in zip(g, wg))
        fails += not ok
        print(f"   {'PASS' if ok else 'FAIL'}  d={d} y={y}  loss={got:.4f} oracle={want:.4f}  "
              f"grad={[round(x, 3) for x in g]} oracle={[round(x, 3) for x in wg]}  ({note})")
    (a, _), (b, _) = run(fn, method, *META[0]), run(fn, method, *META[1])
    mok = close(b, a)
    fails += not mok
    print(f"   {'PASS' if mok else 'FAIL'}  metamorphic: add easy pos 0.1 -> loss {a:.4f} -> {b:.4f} (must be unchanged)")
    verdict = "PASS" if fails == 0 else "FAIL"
    print(f"   VERDICT {verdict} ({fails}/{len(CASES) + 1} checks failed), expected {expect}")
    ok_all &= verdict == expect
print("ALL AS EXPECTED" if ok_all else "UNEXPECTED RESULT")
sys.exit(0 if ok_all else 1)
