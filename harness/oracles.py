"""Oracle templates. Each restates the math a change must preserve; none reads the
implementation under test. The planner picks a family and fills typed parameters;
it never writes code. Every template returns a Verdict with the number it measured,
the threshold it used, and why that threshold (so the receipt can show both)."""
import copy, math
from dataclasses import dataclass, field, asdict
import torch

EPS = {torch.float64: 2.22e-16, torch.float32: 1.19e-7, torch.float16: 9.77e-4, torch.bfloat16: 7.81e-3}
DECLARED = (ValueError, AssertionError, NotImplementedError, KeyError)   # a stated rejection
BINDING = (AttributeError, ImportError, NameError, TypeError, SyntaxError)  # fixture cannot bind: ERROR


@dataclass
class Verdict:
    family: str
    status: str                 # PASS | FAIL | ERROR  (ERROR = harness could not run; never a detection)
    metric: float | None = None
    threshold: float | None = None
    detail: str = ""
    witness: dict = field(default_factory=dict)
    def to_json(self): return asdict(self)


def _f(x):
    if isinstance(x, torch.Tensor):
        return x.detach().double()
    return torch.as_tensor(x, dtype=torch.float64)


def _finite(x):
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values() if isinstance(v, (int, float, torch.Tensor)))
    if isinstance(x, (list, tuple)):
        return all(_finite(v) for v in x if isinstance(v, (int, float, torch.Tensor)))
    return bool(torch.isfinite(_f(x)).all())


def _call(fn, *a, **k):
    """Run fn; classify the outcome the way testing.md does: a declared rejection
    (ValueError with a message) is behaviour, an arithmetic exception is a bug."""
    try:
        return "ok", fn(*a, **k)
    except HarnessError:  # adapter-side binding failure: ERROR, never a detection
        raise
    except DECLARED as e:
        return "declared", f"{type(e).__name__}: {e}"
    except BINDING as e:
        raise HarnessError(f"{type(e).__name__}: {e}") from e
    except Exception as e:  # ZeroDivisionError, FloatingPointError, OverflowError, RuntimeError ...
        return "raised", f"{type(e).__name__}: {e}"


class HarnessError(Exception):
    """The fixture could not bind to the target (missing symbol, other signature).
    Reported as ERROR, never as a detection."""


# 1 ---------------------------------------------------------------------------
def ga_invariance(step, make_batch, micro: int, accum: int, tokens_per_row: int = 1, safety: float = 8.0):
    """accum micro-batches of `micro` rows must give the same loss and gradients as one
    batch of micro*accum rows. step(list_of_batches) -> (loss: float, grads: {name: Tensor}),
    starting from the same weights every call. Tolerance: fp32 eps, grown with the square
    root of the number of summed terms (rows * tokens), times a stated safety factor."""
    B = micro * accum
    big = make_batch(B)
    chunks = [tuple(t[i * micro:(i + 1) * micro] for t in big) for i in range(accum)]
    l1, g1 = step([big])
    lk, gk = step(chunks)
    tol = safety * EPS[torch.float32] * math.sqrt(B * tokens_per_row)
    worst, where = abs(lk / l1 - 1.0), "loss"
    for n in g1:
        a, b = _f(g1[n]), _f(gk[n])
        rel = ((a - b).norm() / (a.norm() + 1e-30)).item()
        if rel > worst:
            worst, where = rel, n
    ok = worst <= tol
    return Verdict("ga_invariance", "PASS" if ok else "FAIL", worst, tol,
                   f"{accum}x{micro} vs 1x{B}: loss ratio {lk / l1:.4f} (must be 1.0000); worst rel gap at {where}",
                   {"micro": micro, "accum": accum, "loss_1xB": l1, f"loss_{accum}x{micro}": lk})


# 2 ---------------------------------------------------------------------------
def dtype_shadow(fn, make_inputs, target_dtype: str, n_accum: int = 1, safety: float = 8.0):
    """Run the SAME code twice: inputs in the target dtype, and inputs promoted to float64
    (ints promoted to float). The narrow result may differ from the wide one by at most
    what the target type can explain: safety * eps_t * (sqrt(n_accum)*|ref| + input_scale)
    for floats, 0.5 for integer outputs (round to nearest)."""
    dt = getattr(torch, target_dtype) if target_dtype != "int" else None
    narrow, wide = make_inputs(dt), make_inputs("wide")
    s1, y = _call(fn, *narrow)
    s2, yr = _call(fn, *wide)
    if s1 != "ok" or s2 != "ok":
        return Verdict("dtype_shadow", "FAIL" if s1 == "raised" else "ERROR", detail=f"narrow={y} wide={yr}")
    y, yr = _f(y), _f(yr)
    if dt is None:
        budget = torch.full_like(yr, 0.5)
        why = "integer result must be the nearest integer to the float64 value (|err| <= 0.5)"
    else:
        scale = max((_f(t).abs().max().item() for t in narrow if isinstance(t, torch.Tensor)), default=1.0)
        eps = EPS[dt]
        budget = safety * eps * (math.sqrt(n_accum) * yr.abs() + scale)
        why = f"{safety} * eps({target_dtype})={eps:.3g} * (sqrt({n_accum})*|ref| + input_scale {scale:.3g})"
    ratio = ((y - yr).abs() / budget)
    m = ratio.max().item() if ratio.numel() else 0.0
    i = int(ratio.argmax()) if ratio.numel() > 1 else 0
    return Verdict("dtype_shadow", "PASS" if m <= 1.0 else "FAIL", m, 1.0,
                   f"worst |narrow - float64| / budget = {m:.3g}; budget = {why}",
                   {"narrow": y.flatten()[i].item(), "float64": yr.flatten()[i].item()})


# 3 ---------------------------------------------------------------------------
def decomposition(fn, whole, pieces: int, combine: str, p: float = 2.0, safety: float = 8.0):
    """Splitting the input must not change the answer the maths defines.
    combine='same':  fn(pieces) == fn([whole])        (a norm over a list of tensors)
    combine='sum':   fn(whole) == sum(fn(piece))
    combine='mean_by_count': fn(whole) == sum(n_i * fn(piece_i)) / sum(n_i)
    combine='pnorm': fn(whole) == (sum fn(piece)**p) ** (1/p)"""
    flat = whole.reshape(-1)
    parts = list(torch.tensor_split(flat, pieces))
    if combine == "same":
        s1, lhs = _call(fn, parts)
        s2, rhs = _call(fn, [flat])
    else:
        s1, lhs = _call(fn, whole)
        outs = [_call(fn, q) for q in parts]
        s2 = "ok" if all(o[0] == "ok" for o in outs) else "raised"
        vals = [_f(o[1]) for o in outs] if s2 == "ok" else []
        if s2 == "ok":
            rhs = {"sum": lambda: sum(vals),
                   "mean_by_count": lambda: sum(len(q) * v for q, v in zip(parts, vals)) / flat.numel(),
                   "pnorm": lambda: sum(v ** p for v in vals) ** (1.0 / p)}[combine]()
    if s1 != "ok" or s2 != "ok":
        return Verdict("decomposition", "FAIL" if "raised" in (s1, s2) else "ERROR", detail=f"{lhs}")
    lhs, rhs = _f(lhs), _f(rhs)
    rel = ((lhs - rhs).abs() / (rhs.abs() + 1e-30)).max().item()
    tol = safety * EPS[torch.float32] * math.sqrt(flat.numel())
    return Verdict("decomposition", "PASS" if rel <= tol else "FAIL", rel, tol,
                   f"combine={combine} p={p}: pieces give {lhs.flatten()[0].item():.6g}, whole gives {rhs.flatten()[0].item():.6g}",
                   {"pieces": pieces, "p": p})


# 4 ---------------------------------------------------------------------------
def bounds(fn, sweep, lo: float, hi: float, atol: float = 0.0):
    """Every value over the sweep must be finite and inside [lo, hi] (from the config,
    not from the code). A declared ValueError is allowed; an arithmetic exception is not."""
    for x in sweep:
        s, v = _call(fn, x)
        if s == "declared":
            continue
        if s == "raised":
            return Verdict("bounds", "FAIL", None, None, f"raised at {x}: {v}", {"at": x})
        v = _f(v).item()
        if not math.isfinite(v) or v < lo - atol or v > hi + atol:
            return Verdict("bounds", "FAIL", v, None, f"value {v} outside [{lo}, {hi}] at {x}", {"at": x, "value": v})
    return Verdict("bounds", "PASS", None, None, f"{len(sweep)} points inside [{lo}, {hi}]")


# 5 ---------------------------------------------------------------------------
def monotonic(fn, sweep, direction: str, atol: float = 0.0):
    """Over the sweep, fn must never move against `direction` (nonincreasing | nondecreasing)."""
    prev = None
    for x in sweep:
        s, v = _call(fn, x)
        if s != "ok":
            return Verdict("monotonic", "FAIL" if s == "raised" else "ERROR", None, None, f"{v} at {x}", {"at": x})
        v = _f(v).item()
        if prev is not None:
            bad = v > prev[1] + atol if direction == "nonincreasing" else v < prev[1] - atol
            if bad:
                return Verdict("monotonic", "FAIL", v - prev[1], atol,
                               f"{direction} violated: f({prev[0]})={prev[1]:.6g} then f({x})={v:.6g}",
                               {"x0": prev[0], "x1": x})
        prev = (x, v)
    return Verdict("monotonic", "PASS", None, atol, f"{direction} over {len(sweep)} points")


# 6 ---------------------------------------------------------------------------
def no_mutation(fn, make_args, check_aliasing: bool = False):
    """Arguments the caller still owns must be unchanged after the call; optionally the
    result must not share storage with any input."""
    args = make_args()
    snap = copy.deepcopy(args)
    s, out = _call(fn, *args)
    if s == "raised":
        return Verdict("no_mutation", "FAIL", detail=out)
    for i, (a, b) in enumerate(zip(args, snap)):
        same = torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
        if not same:
            return Verdict("no_mutation", "FAIL", detail=f"argument {i} changed by the call",
                           witness={"before": repr(b)[:200], "after": repr(a)[:200]})
    if check_aliasing and isinstance(out, torch.Tensor):
        ptrs = {a.untyped_storage().data_ptr() for a in args if isinstance(a, torch.Tensor)}
        if out.untyped_storage().data_ptr() in ptrs:
            return Verdict("no_mutation", "FAIL", detail="result aliases an input")
    return Verdict("no_mutation", "PASS", detail=f"{len(args)} arguments unchanged")


# 7 ---------------------------------------------------------------------------
def finite_extremes(fn, make_inputs, scales, expect: str = "finite", backward: bool = False,
                    rtol: float | None = None):
    """Large FINITE inputs must give finite outputs (and finite grads if backward).
    expect='finite_nonzero' also forbids an all-zero result; expect='scale_invariant'
    requires fn(s*x) == fn(x) within rtol (orthogonalisation, normalisation, argmax...)."""
    base = None
    for s in scales:
        args = make_inputs(s)
        st, y = _call(fn, *args)
        if st != "ok":
            return Verdict("finite_extremes", "FAIL" if st == "raised" else "ERROR", None, None, f"{y} at scale {s}", {"scale": s})
        if not _finite(y):
            return Verdict("finite_extremes", "FAIL", None, None, f"non-finite output at scale {s}: {y}", {"scale": s})
        if expect == "finite_nonzero" and _f(y).abs().max().item() == 0.0:
            return Verdict("finite_extremes", "FAIL", 0.0, None, f"all-zero output at scale {s}", {"scale": s})
        if backward:
            y.sum().backward()
            g = [a.grad for a in args if isinstance(a, torch.Tensor) and a.grad is not None]
            if not all(_finite(t) for t in g):
                return Verdict("finite_extremes", "FAIL", None, None, f"non-finite gradient at scale {s}", {"scale": s})
        if expect == "scale_invariant":
            if base is None:
                base = _f(y)
            else:
                rel = ((_f(y) - base).norm() / (base.norm() + 1e-30)).item()
                if rel > rtol:
                    return Verdict("finite_extremes", "FAIL", rel, rtol, f"output changed with input scale {s}", {"scale": s})
    return Verdict("finite_extremes", "PASS", None, rtol, f"finite at scales {list(scales)} ({expect})")


# 8 ---------------------------------------------------------------------------
def config_audit(groups, named_params, config: dict, rules):
    """Optimizer param groups must encode the config: every trainable parameter in
    exactly one group, lr and weight decay as configured. `rules` come from a fixed list."""
    ids = {}
    for gi, g in enumerate(groups):
        for p in g["params"]:
            ids.setdefault(id(p), []).append(gi)
    for name, p in named_params:
        if not p.requires_grad:
            continue
        where = ids.get(id(p), [])
        if "each_param_once" in rules and len(where) != 1:
            return Verdict("config_audit", "FAIL", detail=f"{name} is in {len(where)} groups")
        g = groups[where[0]]
        if "lr_matches_config" in rules and not math.isclose(g.get("lr", config["lr"]), config["lr"]):
            return Verdict("config_audit", "FAIL", detail=f"{name}: lr {g['lr']} != config {config['lr']}")
        if "decay_rule_ndim" in rules:
            want = config["weight_decay"] if p.dim() >= 2 else 0.0
            if not math.isclose(g.get("weight_decay", 0.0), want):
                return Verdict("config_audit", "FAIL", detail=f"{name} (ndim {p.dim()}): wd {g.get('weight_decay')} != {want}")
    return Verdict("config_audit", "PASS", detail=f"{len(ids)} params checked against {rules}")


# 9 ---------------------------------------------------------------------------
def reference(fn, ref, make_inputs, cases, safety: float = 8.0, n_accum: int = 1,
              rtol: float | None = None, why: str | None = None):
    """Compare to an independent implementation of the SAME maths, computed in float64
    (a torch function or a closed form from REFERENCES, never the code under test)."""
    worst, wit = 0.0, {}
    for c in cases:
        args = make_inputs(**c)
        s, y = _call(fn, *args)
        if s != "ok":
            return Verdict("reference", "FAIL" if s == "raised" else "ERROR", detail=f"{y} for {c}", witness=c)
        yr = _f(ref(*args))
        rel = ((_f(y) - yr).abs() / (yr.abs() + 1e-30)).max().item()
        if rel > worst:
            worst, wit = rel, {**c, "got": _f(y).flatten()[0].item(), "want": yr.flatten()[0].item()}
    tol = rtol if rtol is not None else safety * EPS[torch.float32] * math.sqrt(n_accum)
    why = why or f"{safety} * eps(fp32) * sqrt({n_accum})"
    return Verdict("reference", "PASS" if worst <= tol else "FAIL", worst, tol,
                   f"worst relative gap to float64 reference over {len(cases)} cases; tol = {why}", wit)


# 10 --------------------------------------------------------------------------
def edge_sweep(fn, edges: dict):
    """Configurations the tests never hit (batch of 1, empty input, all masked,
    warmup == total). Each must return finite values or raise a DECLARED error."""
    for name, args in edges.items():
        s, y = _call(fn, *args)
        if s == "raised":
            return Verdict("edge_sweep", "FAIL", detail=f"{name}: {y}", witness={"edge": name})
        if s == "ok" and not _finite(y):
            return Verdict("edge_sweep", "FAIL", detail=f"{name}: non-finite result {y}", witness={"edge": name})
    return Verdict("edge_sweep", "PASS", detail=f"{len(edges)} edges finite or declared")
