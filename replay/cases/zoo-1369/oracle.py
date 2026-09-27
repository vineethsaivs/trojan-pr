"""zoo-1369 oracle: grad-accumulation invariance of Unsloth's DAPO loss (unslothai/unsloth-zoo #1369).

Math (restated, not copied from any SHA): DAPO is a token-mean objective. The loss one optimizer
step sees, summed over its GA micro-batches, must equal
    sum_{tokens t in the step's window} l_t  /  (#tokens in the window)
whatever steps_per_generation (spg) is. With beta=0 and no old policy the importance ratio is
exp(logp - logp.detach()) = exp(0) = 1 exactly, so l_t = -A_row. The expected value below is
computed in plain Python from the inputs only; it never calls or mirrors the code under test.

Method: fetch unsloth_zoo/rl_replacements.py at each SHA (read-only gh api GET), ast-extract
grpo_compute_loss, exec it with only torch and math, feed it micro-batches the way the trainer
does: num_items_in_batch = token count of the whole generation batch (spg micro-batches).
Every SHA gets the steps_per_generation kwarg (the fix's caller passes it, rl_fix L1462;
older SHAs read kwargs with .get and ignore it).

Tolerance: the intro SHA casts logits to fp32, so assume fp32 everywhere (eps = 2**-23).
Each window sums at most TOK (10) tokens per micro-batch plus GA micro-batch results, so the
recursive-summation bound is n*eps*sum|l_t|/N with n = TOK + GA. Use 4x that as margin:
    tol = 4 * (TOK + GA) * 2**-23 * mean|l_t|    (about 2e-5 * mean|l_t| at GA=32).
The bug's error is |GA/spg - 1| * |expected| >= 0.5 * |expected|, orders of magnitude larger.
"""
import ast, base64, json, math, os, random, subprocess, sys
import torch

REPO, PATH = "unslothai/unsloth-zoo", "unsloth_zoo/rl_replacements.py"
SHAS = [  # (label, sha, API: "logits4"=pre-intro signature, "logits"=(B,T,V) logits, "logps"=per-token logps)
    ("pre-intro 75106e95 (parent of #308)", "75106e9553eb59a9e6e433953156d84ccbb4b5f4", "logits4"),
    ("intro     f690a5aa (#308)",           "f690a5aaa3eccab272f6b64c990a93a7a64a0b60", "logits"),
    ("fix-parent 863cce0e",                 "863cce0ece3b61d1a0ebdd0bd46e6afcafc06435", "logps"),
    ("fix       a7640549 (#1369)",          "a764054980dfbee745cb0a6e124722734fd53e10", "logps"),
]
HERE = os.path.dirname(os.path.abspath(__file__))
EPS32 = 2.0 ** -23


def fetch(sha):
    cache = os.path.join(HERE, "src", f"{sha}.py")
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        out = subprocess.check_output(["gh", "api", f"repos/{REPO}/contents/{PATH}?ref={sha}", "--jq", ".content"])
        open(cache, "wb").write(base64.b64decode(out))
    return open(cache).read()


def extract(src, name):
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "grpo_compute_loss")
    g = {"torch": torch, "math": math}
    exec(compile(ast.Module([fn], []), name, "exec"), g)
    return g["grpo_compute_loss"]


def call(f, api, logp, mask, adv, kw):
    ids = torch.zeros(logp.shape, dtype=torch.long)
    if api == "logps":
        return f(None, logp, None, None, ids, mask, 0.0, adv, **kw)[0]
    # 2-way logits whose log_softmax at token 0 equals logp: log(p) - log(p + (1-p)) = log(p)
    logits = torch.stack([logp, torch.log1p(-logp.exp())], -1)
    if api == "logits":
        return f(None, logits, None, None, ids, mask, 0.0, adv, **kw)[0]
    return f(None, logits, None, ids, mask, 0.0, adv, **kw)[0]  # pre-intro: no sampling_per_token_logps arg


def make_micro(n, B, T, TOK, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        mask = (torch.randperm(B * T, generator=g) < TOK).reshape(B, T).double()
        logp = -torch.rand(B, T, generator=g, dtype=torch.float64) - 0.1  # valid log-probs < 0
        adv = torch.randn(B, generator=g, dtype=torch.float64)
        out.append((logp, mask, adv))
    return out


def check(f, api, spg, ga, B, T, TOK, seed=0):
    micro = make_micro(max(spg, ga), B, T, TOK, seed)
    window = micro[:ga]  # the micro-batches of one optimizer step
    got = 0.0
    for i, (logp, mask, adv) in enumerate(window):
        gen = micro[(i // spg) * spg:(i // spg) * spg + spg]  # generation batch this micro-batch came from
        kw = dict(loss_type="dapo", num_items_in_batch=float(sum(m.sum() for _, m, _ in gen)),
                  num_processes=1, current_gradient_accumulation_steps=ga, steps_per_generation=spg,
                  epsilon_high=5.0)
        got += float(call(f, api, logp, mask, adv, kw))
    # oracle: plain-Python token mean of l_t = -A_row over the window's masked tokens
    terms = [-float(adv[b]) for _, mask, adv in window for b in range(B) for t in range(T) if mask[b, t] == 1]
    expected = sum(terms) / len(terms)
    tol = 4 * (TOK + ga) * EPS32 * (sum(abs(x) for x in terms) / len(terms))
    return got, expected, abs(got - expected) <= tol, tol


CONFIGS = [  # (label, spg, GA, B, T, TOK)
    ("minimal 1 token/mb", 4, 2, 1, 1, 1),
    ("control spg==GA",    2, 2, 3, 5, 10),
    ("spg>GA",             4, 2, 3, 5, 10),
    ("spg<GA",             1, 2, 3, 5, 10),
    ("TRL Lite-PPO DAPO",  8, 32, 3, 5, 10),
]

if __name__ == "__main__":
    verdicts = {}
    for label, sha, api in SHAS:
        f = extract(fetch(sha), f"{sha[:8]}:{PATH}")
        rows = []
        for clabel, spg, ga, B, T, TOK in CONFIGS:
            try:
                got, exp, ok, tol = check(f, api, spg, ga, B, T, TOK)
            except ValueError as e:
                print(f"{label}  {clabel:20s} N/A: {e}")
                rows.append(None)
                continue
            rows.append(ok)
            print(f"{label}  {clabel:20s} spg={spg:<2d} GA={ga:<2d} got={got:+.8f} expected={exp:+.8f} "
                  f"ratio={got / exp:.4f} tol={tol:.1e} {'PASS' if ok else 'FAIL'}")
        v = "N/A (no dapo branch)" if all(r is None for r in rows) else ("PASS" if all(rows) else "FAIL")
        verdicts[label] = v
        print(f"{label}  VERDICT {v}\n")
    print(json.dumps(verdicts, indent=1))
    ok = (verdicts[SHAS[1][0]] == "FAIL" and verdicts[SHAS[2][0]] == "FAIL" and verdicts[SHAS[3][0]] == "PASS")
    print("ORACLE DISCRIMINATES (buggy FAIL, fix PASS):", ok)
    sys.exit(0 if ok else 1)
