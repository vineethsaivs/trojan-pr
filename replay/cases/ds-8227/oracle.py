# Oracle for deepspeedai/DeepSpeed #8227 (1-bit LAMB get_lamb_coeffs aliasing).
# Fetches deepspeed/runtime/fp16/onebit/lamb.py at each SHA from GitHub (read-only),
# execs the real OnebitLamb class, and runs its REAL step() twice on CPU
# (warmup stage: pure torch, no comm backend needed).
#
# Relation (time-metamorphic, snapshot immutability): the trust ratios a caller reads
# after step t are a record of step t. Running step t+1 must not change that record.
#   snap = get_lamb_coeffs() after step 1; ref = independent copy of it
#   run step 2 (with different grads, so the ratios genuinely change)
#   require snap == ref and len(snap) == number of params
# Tolerance: exact equality, zero tolerance. No arithmetic happens between the read and
# the compare, so any difference at all means the caller's object was mutated.
import ast, base64, json, subprocess, sys, types
import numpy as np, torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

REPO, PATH = "deepspeedai/DeepSpeed", "deepspeed/runtime/fp16/onebit/lamb.py"
INTRO = "67a48aaa8906878b2ce244319e219155c85de46c"   # PR #970 merge, adds lamb.py
FIX_PARENT = "b39e07a717686e19c69560f78819182720de23ed"
FIX = "5dceeeab5347d1f5b2a7db8c037b78d9d41dfcc0"      # PR #8227 merge


def gh(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def load_class(sha):
    blob = gh(f"repos/{REPO}/contents/{PATH}?ref={sha}")
    if blob is None:
        return None
    tree = ast.parse(base64.b64decode(blob["content"]).decode())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OnebitLamb")
    ns = {"torch": torch, "np": np, "types": types,
          "_flatten_dense_tensors": _flatten_dense_tensors,
          "_unflatten_dense_tensors": _unflatten_dense_tensors,
          "get_accelerator": lambda: types.SimpleNamespace(empty_cache=lambda: None),
          "dist": types.SimpleNamespace(get_rank=lambda: 0)}  # only used on the init-step print
    exec(compile(ast.Module([cls], []), f"{sha[:7]}:{PATH}", "exec"), ns)
    return ns["OnebitLamb"]


def make_opt(Cls, params):
    # __init__ asserts an initialized distributed backend and builds NCCL/MPI handles,
    # so build via __new__ and set only what the warmup path of step() reads.
    opt = Cls.__new__(Cls)
    torch.optim.Optimizer.__init__(opt, params, dict(
        lr=1e-2, bias_correction=True, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        max_grad_norm=0.0, max_coeff=10.0, min_coeff=0.01))
    opt.__dict__.update(eps_mode=1, deepspeed=None, lamb_freeze_key=False, initialize=True,
                        freeze_step=10**9, coeff_beta=0.9, factor_max=4.0, factor_min=0.5,
                        factor_threshold=0.1, using_pipeline=False, lamb_coeffs=[],
                        size=1, divider=8,  # single worker: what __init__ derives from the comm handle
                        exp_avg_flat=[], dummy_exp_avg={}, corrected_tensor_sizes=[],
                        server_chunk_sizes=[], worker_errors=[], server_errors=[])
    return opt


def run(label, sha):
    Cls = load_class(sha)
    if Cls is None:
        print(f"{label:22s} {sha[:7]}  N/A  lamb.py does not exist at this commit")
        return None
    w = [torch.tensor([3.0, 4.0], requires_grad=True), torch.tensor([1.0, 2.0, 2.0], requires_grad=True)]
    opt = make_opt(Cls, w)
    g1 = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, -1.0, 0.5])]
    g2 = [torch.tensor([4.0, -1.0]), torch.tensor([0.1, 0.1, 3.0])]
    for p, g in zip(w, g1):
        p.grad = g.clone()
    opt.step()
    snap = opt.get_lamb_coeffs()
    ref = [float(x) for x in snap]           # independent record of step 1
    for p, g in zip(w, g2):
        p.grad = g.clone()
    opt.step()
    step2 = [float(x) for x in opt.lamb_coeffs]
    assert step2 != ref, "vacuous: step 2 did not change the ratios"
    after = [float(x) for x in snap]
    ok = len(after) == len(w) and after == ref
    print(f"{label:22s} {sha[:7]}  {'PASS' if ok else 'FAIL'}  step1_read={fmt(ref)}  "
          f"same_object_after_step2={fmt(after)}  step2_ratios={fmt(step2)}")
    return ok


def fmt(xs):
    return "[" + ", ".join(f"{x:.6f}" for x in xs) + "]"


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")  # deprecated add_(Number, Tensor) in 2021 code
    pre = gh(f"repos/{REPO}/commits/{INTRO}")["parents"][0]["sha"]
    res = [run("pre-intro (#970^)", pre), run("intro (#970)", INTRO),
           run("fix parent (#8227^)", FIX_PARENT), run("fix (#8227)", FIX)]
    print("torch", torch.__version__)
    assert res == [None, False, False, True], res
    print("oracle verdicts as expected: FAIL on intro and fix parent, PASS on fix")
