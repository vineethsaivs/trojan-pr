# Oracle for sentence-transformers GOR loss (intro PR #3654, fix PR #3921).
# Fetches the loss file at each ref from GitHub (read-only gh api GET), extracts
# compute_gor + compute_loss_from_embeddings with ast, runs them with torch only,
# and checks them against the math restated below (never the implementation's divisor).
#
# Math: P = ordered pairs (i, j), i != j, of the n rows. s_ij = cosine(e_i, e_j).
#   mean_term   = (mean_{P} s_ij)^2
#   second_term = max(mean_{P} s_ij^2 - 1/d, 0)
# Invariants: (1) finite for every n >= 1; (2) n = 1 means P is empty, nothing to
# regularize, so both terms are 0 and task + GOR == task; (3) n >= 2 matches a float64
# brute-force loop over P.
# Tolerance: implementation runs in fp32, oracle in fp64. One fp32 cosine over d=8 has
# error <= ~(d+2)*u = 10*6e-8 = 6e-7; averaging keeps that bound, squaring a value <= 1
# at most doubles it (~1.2e-6). atol 1e-5 is ~8x that bound. The bug is nan vs finite,
# so the tolerance does not decide the verdict.
import ast, base64, json, math, subprocess, torch

REPO = "huggingface/sentence-transformers"
OLD = "sentence_transformers/losses/GlobalOrthogonalRegularizationLoss.py"
NEW = "sentence_transformers/sentence_transformer/losses/global_orthogonal_regularization.py"
ATOL = 1e-5


def gh(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def fetch(ref, path):
    j = gh(f"repos/{REPO}/contents/{path}?ref={ref}")
    return None if j is None else base64.b64decode(j["content"]).decode()


def cos_sim(a, b):  # stub for sentence_transformers.util.cos_sim (a dependency, not the code under test)
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)
    return a @ b.T


def load(src, name):
    cls = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef) and n.name == "GlobalOrthogonalRegularizationLoss")
    fns = [f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name in ("compute_gor", "compute_loss_from_embeddings")]
    mod = ast.Module(body=[ast.ClassDef(name="G", bases=[], keywords=[], body=fns, decorator_list=[], type_params=[])], type_ignores=[])
    ns = {"torch": torch, "Tensor": torch.Tensor}
    exec(compile(ast.fix_missing_locations(mod), name, "exec"), ns)
    g = ns["G"]()
    g.similarity_fct, g.mean_weight, g.second_moment_weight, g.aggregation = cos_sim, 1.0, 1.0, "mean"
    return g


def oracle(e):  # float64 brute force over explicit pairs i != j
    rows = e.double().tolist()
    n, d = len(rows), len(rows[0])
    s = []
    for i in range(n):
        for j in range(n):
            if i != j:
                a, b = rows[i], rows[j]
                s.append(sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))))
    if not s:
        return 0.0, 0.0  # empty pair set: nothing to spread apart
    return (sum(s) / len(s)) ** 2, max(sum(x * x for x in s) / len(s) - 1.0 / d, 0.0)


def check(g):
    ok_all = True
    for n in (4, 3, 2, 1):
        torch.manual_seed(0)
        e = torch.randn(n, 8, requires_grad=True)
        task = e.pow(2).mean()
        terms = g.compute_loss_from_embeddings([e])
        m, s = terms["gor_mean"], terms["gor_second_moment"]
        total = task + m + s
        total.backward()
        om, os_ = oracle(e.detach())
        ok = (torch.isfinite(total).item() and abs(m.item() - om) <= ATOL and abs(s.item() - os_) <= ATOL
              and (n > 1 or abs(total.item() - task.item()) <= ATOL) and torch.isfinite(e.grad).all().item())
        ok_all &= ok
        print(f"    n={n} gor_mean={m.item():.6f} gor_sm={s.item():.6f} oracle=({om:.6f},{os_:.6f}) "
              f"task={task.item():.4f} task+gor={total.item():.4f} grad_finite={torch.isfinite(e.grad).all().item()} {'PASS' if ok else 'FAIL'}")
    return ok_all


intro = "2caeddff7d65bbac9642d55835caf10952532716"
pre = gh(f"repos/{REPO}/commits/{intro}")["parents"][0]["sha"]
refs = [
    ("pre-intro (parent of 2caeddff)", pre, OLD),
    ("intro 2caeddff (PR #3654, buggy)", intro, OLD),
    ("tag v6.1.0 (latest release)", "v6.1.0", NEW),
    ("fix-parent b806a61", "b806a61ab679f44beaf83e658b4d1b7176f62e39", NEW),
    ("fix 655121f (PR #3921)", "655121f9be2d2caac1cb7859ff62340cc9878258", NEW),
]
for label, ref, path in refs:
    src = fetch(ref, path)
    if src is None:
        print(f"{label} [{ref[:8]}]: file absent at this ref (GOR did not exist yet), NOT APPLICABLE")
        continue
    print(f"{label} [{ref[:8]}] {path}")
    print(f"  VERDICT: {'PASS' if check(load(src, path)) else 'FAIL'}")
