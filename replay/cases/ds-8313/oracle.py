# Oracle replay for deepspeedai/DeepSpeed #8313 (p-norm gradient clipping), CPU only.
# Fetches each file at each SHA (read-only gh api, cached in ./src), ast-extracts the target
# function, runs it with single-process stubs, and checks it against the math restated in plain
# Python floats (no torch.norm, no torch clip_grad_norm_, no DeepSpeed code in the oracle):
#   A  norm:      N_p(g) = (sum_j |g_j|^p)^(1/p) over ALL gradient entries, whatever the parameter split
#   B  clip:      g' = g * min(1, max_norm / N_p(g))       (only for utils.clip_grad_norm_)
#   C  partition: N_p is the same for one tensor [3,-4,2], for [3,-4]+[2], and for singletons
# Tolerance: inputs are small exact integers in fp32. Truth is computed in float64; fp32 pow/sum on
# 3 terms is off by a few ulp (~1e-7 relative), and DeepSpeed's +1e-6 in the clip denominator moves
# the coef by < 1e-6 relative. rtol=1e-5 (norm) and atol=1e-5 (grads, |g| <= 4) sit ~10x above that
# noise and ~4 orders of magnitude below the bug (37% at p=3, 489% at p=1).
import ast, os, subprocess, sys, types, torch
from math import inf

REPO = "deepspeedai/DeepSpeed"
HERE = os.path.dirname(os.path.abspath(__file__))
UTILS, S12, S3 = "deepspeed/runtime/utils.py", "deepspeed/runtime/zero/stage_1_and_2.py", "deepspeed/runtime/zero/stage3.py"
SHAS = [  # (label, sha, expect_pass)
    ("pre-intro   4f477328 (#4915 parent)", "4f477328c411270cf378a2318bc4f51c512ad2c8", True),
    ("INTRO #4915 961bc856", "961bc85624174e8ca8ee7626b3f3b53c6c768085", False),
    ("#5150       005afe12", "005afe124f56b2243785067a898134fa2bf8735c", False),
    ("fix parent  92843ad7", "92843ad706067f0ab551a06a079a8923dcde27b4", False),
    ("FIX #8313   37cf8f24", "37cf8f240b1eadb178df462d8b80d356beabcf8b", True),
]
ZERO_SHAS = SHAS[3:]  # ZeRO sites checked at fix parent vs fix (bug there predates #4915)
RTOL, ATOL, MAX_NORM = 1e-5, 1e-5, 1.0
G = [3.0, -4.0, 2.0]
SPLITS = {"[3,-4]+[2]": [G[:2], G[2:]], "one tensor": [G], "singletons": [[x] for x in G]}


def pnorm(xs, p):  # the math, restated
    return sum(abs(x) ** p for x in xs) ** (1.0 / p)


def fetch(path, sha):
    cache = f"{HERE}/src/{sha[:8]}_{os.path.basename(path)}"
    if not os.path.exists(cache):
        os.makedirs(f"{HERE}/src", exist_ok=True)
        raw = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw",
                              f"repos/{REPO}/contents/{path}?ref={sha}"], check=True, capture_output=True).stdout
        open(cache, "wb").write(raw)
    return open(cache).read()


class Acc:  # single-process CPU accelerator stub
    def device_name(self, *a): return "cpu"
    def current_device_name(self): return "cpu"
    FloatTensor = staticmethod(lambda x: torch.tensor(x, dtype=torch.float32))


STUBS = dict(
    torch=torch, inf=inf, get_accelerator=Acc, is_model_parallel_parameter=lambda p: False,
    dist=types.SimpleNamespace(all_reduce=lambda *a, **k: None, get_world_size=lambda group=None: 1,
                               ReduceOp=types.SimpleNamespace(SUM="sum", MAX="max")),  # world_size 1: no-op
    groups=types.SimpleNamespace(_get_data_parallel_group=lambda: None),
    get_norm_dtype=lambda: torch.float64, PIPE_REPLICATED="ds_pipe_replicated", instrument_w_nvtx=lambda f: f,
    mask_nan_or_inf_with_val_inplace=lambda t, device=None: t.masked_fill_(t.isnan() | t.isinf(), -1.0))
FAKE_SELF = types.SimpleNamespace(  # ZeRO optimizer attributes touched by get_grad_norm_direct
    dp_process_group=None, model_parallel_rank=0, device="cpu", _model_parallel_all_reduce=lambda **k: None,
    _assert_same_partition_group=lambda p: None, _get_param_partition_group=lambda p: None,
    _autoep_expert_parallel_group=lambda p: None)


def extract(path, sha, name):
    fn = next(n for n in ast.walk(ast.parse(fetch(path, sha))) if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(STUBS)
    exec(compile(ast.Module([fn], []), f"{sha[:8]}:{path}", "exec"), ns)
    return ns[name]


def mkparams(split):
    ps = []
    for chunk in split:
        t = torch.nn.Parameter(torch.zeros(len(chunk))); t.grad = torch.tensor(chunk); ps.append(t)
    return ps


def run_utils(sha):
    f = extract(UTILS, sha, "clip_grad_norm_")
    fails = []
    for p in (1, 2, 3):
        truth = pnorm(G, p)
        ps = mkparams(SPLITS["[3,-4]+[2]"])
        n = float(f(ps, max_norm=MAX_NORM, norm_type=p))
        coef = min(1.0, MAX_NORM / truth)
        gerr = max(abs(float(v) - x * coef) for v, x in zip(torch.cat([q.grad for q in ps]), G))
        parts = [round(float(f(mkparams(s), max_norm=1e9, norm_type=p)), 4) for s in SPLITS.values()]
        a, b, c = abs(n - truth) <= RTOL * truth, gerr <= ATOL, max(parts) - min(parts) <= RTOL * truth
        print(f"  utils.clip_grad_norm_ p={p}: truth={truth:.4f} got={n:.4f} [A {'ok' if a else 'FAIL'}]"
              f"  max|g'-contract|={gerr:.4f} [B {'ok' if b else 'FAIL'}]"
              f"  splits={parts} [C {'ok' if c else 'FAIL'}]")
        fails += [f"utils A p={p}"] * (not a) + [f"utils B p={p}"] * (not b) + [f"utils C p={p}"] * (not c)
    return fails


def run_zero(sha, path, tag):
    fn = extract(path, sha, "get_grad_norm_direct")
    f = lambda gs, ps, p: float(fn(FAKE_SELF, gs, ps, norm_type=p))
    fails = []
    for p in (1, 2, 3):
        truth = pnorm(G, p)
        gs = [torch.tensor(c) for c in SPLITS["[3,-4]+[2]"]]
        n = f(gs, mkparams(SPLITS["[3,-4]+[2]"]), p)
        a = abs(n - truth) <= RTOL * truth
        print(f"  {tag}.get_grad_norm_direct p={p}: truth={truth:.4f} got={n:.4f} [A {'ok' if a else 'FAIL'}]")
        fails += [f"{tag} A p={p}"] * (not a)
    return fails


if __name__ == "__main__":
    print(f"torch {torch.__version__}  python {sys.version.split()[0]}  grads [3,-4]+[2]  max_norm={MAX_NORM}")
    ok_all = True
    for label, sha, expect in SHAS:
        print(f"{label}")
        fails = run_utils(sha)
        if (label, sha, expect) in ZERO_SHAS:
            fails += run_zero(sha, S12, "zero12") + run_zero(sha, S3, "zero3")
        verdict = "PASS" if not fails else "FAIL"
        match = (not fails) == expect
        ok_all &= match
        print(f"  => {verdict} (expected {'PASS' if expect else 'FAIL'}){'' if match else '  <-- UNEXPECTED'}"
              f"{'  failed: ' + ', '.join(fails) if fails else ''}")
        # p=2 control: returned norm must be right at every SHA (square() == pow(2))
        assert f"utils A p=2" not in fails, label
    print("ORACLE MATCHES EXPECTATIONS" if ok_all else "ORACLE MISMATCH")
    sys.exit(0 if ok_all else 1)
