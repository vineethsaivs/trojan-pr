"""Oracle for unsloth #11337: GRPO eval_loss must not see the training grad-accum divisor.

Math (restated, not copied from either implementation):
  * An eval pass does no gradient accumulation, so for fixed weights and a fixed eval batch,
    eval_loss is the same whether the trainer ran training with GA=1, GA=4, or never trained.
  * In eval the policy is the old policy, so the importance ratio is exactly 1 and the clip is
    inactive. The GRPO token loss is then  -A_i + beta * k3(t),  k3 = exp(r) - r - 1,
    r = logp_ref - logp_policy  (Schulman's k3 KL estimator, as in the GRPO paper).
    With loss_type='grpo' (mean over tokens, then over sequences) and group advantages that sum
    to zero, the -A_i terms cancel and  eval_loss == beta * mean_i mean_t k3.

Code under test, fetched per SHA with `gh api .../contents/...?ref=SHA` into src/:
  unsloth/models/rl_replacements.py : the divisor that compute_loss forwards (lifted with ast
      and executed against a stub trainer in eval mode).
  unsloth_zoo/rl_replacements.py    : grpo_compute_loss, which divides by that divisor.
  zoo's grpo_accumulated_loss forwards **kwargs to grpo_compute_loss unchanged, so calling
  grpo_compute_loss directly with the forwarded divisor is the same arithmetic.
"""
import ast, inspect, math, sys, types
from pathlib import Path
import torch

SRC = Path(__file__).resolve().parent / "src"

# (label, unsloth sha, zoo sha paired in time, expected)
CASES = [
    ("pre-intro  (parent of #3390)", "0e766b28", "75106e95", "PASS"),
    ("introduce  (#3390 + zoo #308)", "45b1c7f7", "f690a5aa", "FAIL"),
    ("fix parent (after #6523)", "43e7dd30", "dcfe3905", "FAIL"),
    ("fix        (#11337)", "06c1bb40", "dcfe3905", "PASS"),
]

# Tolerance from arithmetic. The zoo code runs in fp32 (old zoo casts logits to fp32; we feed
# fp32 logps to new zoo). Unit roundoff u = 6e-8. The largest summands are |A| <= 1, and the
# reduction touches B*T = 24 of them, so abs error <= ~24 * u * 1 = 1.5e-6. beta*KL below is
# ~1e-2, so relative error <= ~1.5e-4. Use rel 1e-3 (about 7x margin). The bug is an exact
# factor k (rel error 1 - 1/k = 0.75 at k=4), so the gap is ~750x the tolerance.
REL_TOL = 1e-3


def lift_funcs(src, names, glb):
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in nodes:
        n.decorator_list = []
    ns = dict(glb)
    exec(compile(ast.Module(nodes, []), "<lifted>", "exec"), ns)
    return ns


def divisor_fn(unsloth_src):
    """Execute the shipped `current_gradient_accumulation_steps = ...` line of compute_loss.

    Returns None when compute_loss never forwards a divisor (pre-intro)."""
    tree = ast.parse(unsloth_src)
    cl = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "compute_loss"]
    assigns = [
        s for f in cl for s in ast.walk(f)
        if isinstance(s, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "current_gradient_accumulation_steps" for t in s.targets)
    ]
    if not assigns:
        return None, "not forwarded"
    stmt = assigns[0]
    called = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    helpers = lift_funcs(unsloth_src, called, {})  # e.g. _unsloth_grpo_accumulation_steps at the fix
    code = compile(ast.Module([stmt], []), "<compute_loss divisor>", "exec")

    def run(trainer):
        ns = dict(helpers, self=trainer)
        exec(code, ns)
        return ns["current_gradient_accumulation_steps"]
    return run, ast.unparse(stmt.value)


# Fixed eval batch. Plain torch only; the oracle side is fp64.
torch.manual_seed(0)
B, T, V, beta = 4, 6, 8, 0.1
ids = torch.randint(0, V, (B, T))
new_logits = torch.randn(B, T, V, dtype=torch.float64)
ref_logits = new_logits + torch.randn(B, T, V, dtype=torch.float64)
old_logits = new_logits.clone()                       # eval: policy == old policy, ratio == 1
mask = torch.ones(B, T)
adv = torch.tensor([1.0, -1.0, 0.5, -0.5])            # one group, advantages sum to zero

def logp(logits):
    return torch.log_softmax(logits, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)

r = logp(ref_logits) - logp(new_logits)
ORACLE = beta * (torch.exp(r) - r - 1).mean().item()   # beta * mean k3 KL, fp64

# Trainer states an eval can run under. HF Trainer sets current_gradient_accumulation_steps in
# the training loop and never clears it; evaluate() puts the model in eval mode.
STATES = {
    "no train": types.SimpleNamespace(model=types.SimpleNamespace(training=False)),
    "after GA=1": types.SimpleNamespace(model=types.SimpleNamespace(training=False),
                                        current_gradient_accumulation_steps=1),
    "after GA=4": types.SimpleNamespace(model=types.SimpleNamespace(training=False),
                                        current_gradient_accumulation_steps=4),
}


def run_case(unsloth_sha, zoo_sha):
    div, expr = divisor_fn((SRC / f"unsloth_{unsloth_sha}.py").read_text())
    loss_fn = lift_funcs((SRC / f"zoo_{zoo_sha}.py").read_text(), {"grpo_compute_loss"},
                         {"torch": torch, "math": math})["grpo_compute_loss"]
    params = list(inspect.signature(loss_fn).parameters)
    if params[0] == "ref_logits":          # older zoo takes logits and does log_softmax itself
        ref, new, old = (x.float() for x in (ref_logits, new_logits, old_logits))
    else:                                   # newer zoo takes per-token logps
        ref, new, old = (logp(x).float() for x in (ref_logits, new_logits, old_logits))
    pos = [ref, new, old] + ([None] if "sampling_per_token_logps" in params else []) + [ids, mask, beta, adv]

    out = {}
    for name, trainer in STATES.items():
        try:
            kw = {} if div is None else {"current_gradient_accumulation_steps": div(trainer)}
            out[name] = loss_fn(*pos, loss_type="grpo", **kw)[0].item()
        except Exception as e:
            out[name] = f"{type(e).__name__}"
    return expr, out


def close(a, b):
    return isinstance(a, float) and abs(a - b) <= REL_TOL * abs(b)


print(f"oracle eval_loss = beta * mean KL = {ORACLE:.8f}   (rel tol {REL_TOL:g})")
all_as_expected = True
for label, us, zs, expected in CASES:
    expr, out = run_case(us, zs)
    verdict = "PASS" if all(close(v, ORACLE) for v in out.values()) else "FAIL"
    all_as_expected &= verdict == expected
    cells = "  ".join(f"{k}: {v:.8f}" if isinstance(v, float) else f"{k}: {v}" for k, v in out.items())
    ga4 = out["after GA=4"]
    ratio = f"{ORACLE / ga4:.6f}" if isinstance(ga4, float) else "n/a"
    print(f"{verdict}  {label}  unsloth@{us} zoo@{zs}\n"
          f"      divisor = {expr}\n      {cells}  oracle/GA4 = {ratio}  (expected {expected})")
print("oracle separates buggy from fixed as expected" if all_as_expected else "UNEXPECTED verdicts")
sys.exit(0 if all_as_expected else 1)
