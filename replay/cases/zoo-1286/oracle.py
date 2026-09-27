"""Oracle for unslothai/unsloth-zoo #1286: token-mean GRPO-family loss on an empty token set.

Math restated (not copied from either implementation):
  DAPO / BNPO loss = (1/|S|) * sum_{t in S} l_t, with S the valid (unmasked) tokens and
  l_t = -min(r_t*A, clip(r_t, 1-eps, 1+eps)*A), r_t = exp(logp_new - logp_old)   (PPO clipped surrogate).
  |S| is num_items_in_batch / num_processes for dapo/cispo (mask.sum() for bnpo).
  An empty sum is exactly 0, so an empty S must give loss == 0.0 and a gradient of exactly 0.

Checks per SHA:
  C1 empty set   : mask all 0, N=0 (as float and as tensor)  -> loss == 0.0, grad all == 0 (exact).
  C2 reference   : random mask with one fully masked row, N = mask.sum() -> loss == reference.
  C3 finiteness  : mask all 1 but N=0 (the count Unsloth's text path produced) -> loss and grad finite.
Tolerance (C2): |loss - ref| <= 1e-5 * sum|l_t|/N. The reference is float64, but an implementation may
compute log-probs in float32 (the logits-input SHAs do), so the bound is float32's: forward error of a
20-term sum plus an 11-way logsumexp is about (20+11)*eps32*sum|l_t| = 3.7e-6*sum|l_t|, rounded up to
1e-5. A real normalizer bug moves the value by >= 1/N (7.7% at N=13), far outside the bound.
C1 and C3 are exact (0.0 and isfinite are not approximate).

Usage: python oracle.py <path to unsloth-zoo git clone>
"""
import ast, math, subprocess, sys, torch

REPO = sys.argv[1]
SHAS = [("pre-intro f690a5aa^", "f690a5aa^"), ("intro f690a5aa (#308)", "f690a5aa"),
        ("fix-parent ac41049b", "ac41049b"), ("fix 937d9434 (#1286)", "937d9434")]
B, T, V, EPS = 4, 5, 11, 0.2


def load(sha):
    src = subprocess.check_output(["git", "-C", REPO, "show", f"{sha}:unsloth_zoo/rl_replacements.py"], text=True)
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "grpo_compute_loss")
    fn.decorator_list = []
    ns = {"torch": torch, "math": math}
    exec(compile(ast.Module([fn], []), sha, "exec"), ns)
    params = [a.arg for a in fn.args.args]
    return ns["grpo_compute_loss"], params


def inputs(seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, V, dtype=torch.float64, generator=g)
    old_logits = logits + 0.3 * torch.randn(B, T, V, dtype=torch.float64, generator=g)
    ids = torch.randint(0, V, (B, T), generator=g)
    adv = torch.randn(B, dtype=torch.float64, generator=g)
    return logits, old_logits, ids, adv


def logp(logits, ids):
    return torch.log_softmax(logits, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def call(f, params, loss_type, mask, n_items, seed=0):
    """Run the SHA's function; return (loss, grad wrt its 'new' input). Adapts to the SHA's signature."""
    logits, old_logits, ids, adv = inputs(seed)
    takes_logits = "new_logits" in params
    new = (logits if takes_logits else logp(logits, ids)).clone().requires_grad_(True)
    old = old_logits if takes_logits else logp(old_logits, ids)
    pos = [None, new, old] + ([None] if "sampling_per_token_logps" in params else []) + [ids, mask, 0.0, adv]
    loss = f(*pos, loss_type=loss_type, num_items_in_batch=n_items, num_processes=1,
             current_gradient_accumulation_steps=1, max_completion_length=T,
             epsilon_low=EPS, epsilon_high=EPS)[0]
    loss.backward()
    return loss, new.grad


def reference(mask, seed=0):
    logits, old_logits, ids, adv = inputs(seed)
    r = torch.exp(logp(logits, ids) - logp(old_logits, ids))
    A = adv[:, None]
    l = -torch.minimum(r * A, r.clamp(1 - EPS, 1 + EPS) * A)
    n = mask.sum().item()
    return ((l * mask).sum() / n).item() if n > 0 else 0.0, (l.abs() * mask).sum().item() / max(n, 1)


def check(f, params, loss_type):
    out, ok = [], True
    for label, n0 in [("N=0.0", 0.0), ("N=tensor(0)", torch.tensor(0.0))]:
        loss, g = call(f, params, loss_type, torch.zeros(B, T), n0)
        good = loss.item() == 0.0 and bool((g == 0).all())
        ok &= good
        out.append(f"C1 empty {label:<11} loss={loss.item()!r} grad_all_zero={bool((g == 0).all())} -> {'PASS' if good else 'FAIL'}")
    if loss_type in ("dapo", "bnpo"):
        mask = (torch.rand(B, T, generator=torch.Generator().manual_seed(1)) > 0.3).double()
        mask[3] = 0  # one fully masked sample: must not change the result
        loss, _ = call(f, params, loss_type, mask, mask.sum().item())
        ref, scale = reference(mask)
        rtol = 1e-5
        err = abs(loss.item() - ref)
        good = err <= rtol * scale
        ok &= good
        out.append(f"C2 reference N={int(mask.sum())}  loss={loss.item():.15g} ref={ref:.15g} err={err:.2e} "
                    f"bound={rtol * scale:.2e} -> {'PASS' if good else 'FAIL'}")
    if loss_type != "bnpo":
        loss, g = call(f, params, loss_type, torch.ones(B, T), 0.0)
        good = bool(torch.isfinite(loss)) and bool(torch.isfinite(g).all())
        ok &= good
        out.append(f"C3 mask=1,N=0     loss={loss.item()!r} grad_finite={bool(torch.isfinite(g).all())} -> {'PASS' if good else 'FAIL'}")
    return ok, out


print(f"torch {torch.__version__}, float64 inputs, B={B} T={T} V={V}, beta=0, eps={EPS}, num_processes=1")
for label, sha in SHAS:
    f, params = load(sha)
    print(f"\n== {label}   ({len(params)} positional params, input={'logits' if 'new_logits' in params else 'logps'})")
    verdict = True
    for lt in ("dapo", "cispo", "bnpo"):
        try:
            ok, lines = check(f, params, lt)
        except ValueError as e:  # branch does not exist at this SHA
            print(f"  [{lt}] N/A: {e}")
            continue
        verdict &= ok
        for s in lines:
            print(f"  [{lt}] {s}")
    print(f"  VERDICT {sha}: {'PASS' if verdict else 'FAIL'}")
