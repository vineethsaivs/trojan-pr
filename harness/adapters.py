"""Plan check (typed JSON) -> oracle call (PLAN 5.6). No string from a plan is ever evaluated:
targets resolve by getattr, args are built from typed specs. Runs inside gVisor only.

Outcome rule (5.2): resolving, constructing and signature.bind happen before the call and raise
HarnessError (ERROR). Inside the call, DECLARED errors are allowed and anything else, including a
TypeError from inside the target, is a FAIL."""
import copy, inspect, math, types
import torch
import oracles as O
from references import REFS, clip_grad_norm


class MissingTarget(O.HarnessError):
    kind = "missing_target"


class TargetRaised(RuntimeError):
    """A binding-type exception raised INSIDE the target: a bug, not a fixture problem."""


DTYPES = {"float64": torch.float64, "float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16, "int64": torch.int64, "bool": torch.bool}


# ---- typed args -----------------------------------------------------------------------------
def build(a, sv=None, scale=1.0, dt=None):
    """dt: None (as written), a torch float dtype (narrow run), 'wide' (float64 run)."""
    (kind, v), = a.items()
    if kind == "tensor":
        g = torch.Generator().manual_seed(v.get("seed", 0))
        shape, dist, val = v["shape"], v.get("dist", "randn"), v.get("value", 1.0)
        t = {"randn": lambda: torch.randn(shape, generator=g, dtype=torch.float64),
             "rand": lambda: torch.rand(shape, generator=g, dtype=torch.float64),
             "zeros": lambda: torch.zeros(shape, dtype=torch.float64),
             "ones": lambda: torch.ones(shape, dtype=torch.float64),
             "arange": lambda: torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape),
             "const": lambda: torch.full(shape, float(val), dtype=torch.float64)}[dist]()
        if v.get("scaled"):
            t = t * scale
        want = DTYPES[v.get("dtype", "float32")]
        if dt == "wide":
            want = torch.float64
        elif dt is not None and want.is_floating_point:
            want = dt
        t = t.to(want)
        if v.get("requires_grad") and t.is_floating_point():
            t.requires_grad_(True)
        return t
    if kind == "list":
        n, dist, val = v["n"], v.get("dist", "randn"), v.get("value", 1.0)
        if dist == "empty_strings":
            return [""] * n
        g = torch.Generator().manual_seed(0)
        return {"randn": lambda: torch.randn(n, generator=g).tolist(), "arange": lambda: list(range(n)),
                "const": lambda: [val] * n}[dist]()
    if kind == "int":
        return float(v) if dt == "wide" else v
    if kind in ("float", "str", "bool"):
        return v
    if kind == "none":
        return None
    if kind == "sweep":
        return sv
    if kind == "dict":
        return {k: build(x, sv, scale, dt) for k, x in v.items()}
    if kind == "items":
        return [build(x, sv, scale, dt) for x in v]
    if kind == "object":
        return types.SimpleNamespace(**{k: build(x, sv, scale, dt) for k, x in v.items()})
    raise O.HarnessError(f"unknown arg kind {kind}")


def resolve(mod, job, target):
    path, qual = target.split(":", 1)
    if path != job.get("path") and path[:-3].replace("/", ".") != job["module"]:
        raise O.HarnessError(f"target file {path} is not the job's file {job.get('path') or job['module']}")
    obj = mod
    for part in qual.split("."):
        name = part
        if part.startswith("__") and not part.endswith("__") and isinstance(obj, type):
            name = f"_{obj.__name__.lstrip('_')}{part}"      # private name mangling
        if not hasattr(obj, name):
            raise MissingTarget(f"{qual} not found at this commit")
        obj = getattr(obj, name)
    return obj


class KW(dict):
    """Rides as the last positional arg so oracles that call make_inputs() twice (dtype_shadow)
    keep each run's own callable and kwargs."""
    fn = None

    def __deepcopy__(self, memo):        # no_mutation snapshots values, never the callable
        k = KW({n: copy.deepcopy(v, memo) for n, v in self.items()})
        k.fn = self.fn
        return k

    def __eq__(self, other):
        return self.keys() == other.keys() and all(
            torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b for a, b in ((self[n], other[n]) for n in self))

    __hash__ = None


def _slice(x, rows, dim=0):
    if isinstance(x, torch.Tensor) and x.dim() > dim:
        return x.narrow(dim, 0, min(rows, x.shape[dim])).clone()
    return x


def _edge(edge, args, kwargs):
    if edge in ("batch_1", "empty"):
        n = 1 if edge == "batch_1" else 0
        return [_slice(a, n) for a in args], {k: _slice(v, n) for k, v in kwargs.items()}
    if edge == "single_token":
        return [_slice(a, 1, 1) for a in args], {k: _slice(v, 1, 1) for k, v in kwargs.items()}
    if edge == "all_masked":
        kw = {k: (torch.zeros_like(v) if "mask" in k and isinstance(v, torch.Tensor) else
                  0 if k.startswith("num_items") else v) for k, v in kwargs.items()}
        return list(args), kw
    if edge == "all_empty_strings":
        f = lambda x: [""] * len(x) if isinstance(x, list) and x and isinstance(x[0], str) else x
        return [f(a) for a in args], {k: f(v) for k, v in kwargs.items()}
    raise O.HarnessError(f"edge {edge} not supported by the call adapter")


def _config_map(call, params):
    """Flat {key: value} from params.config and every dict built in construct/kwargs."""
    out = {}
    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            out.setdefault(k, v)
            out[pre + k] = v
    for src in (call.get("construct", {}), call.get("kwargs", {}), params.get("config", {})):
        walk({k: build(v) for k, v in src.items()})
    return out


def _bound(b, cfg):
    if isinstance(b, dict):
        if b["config"] not in cfg:
            raise O.HarnessError(f"config key {b['config']} not in the plan's config")
        return float(cfg[b["config"]])
    return float(b)


def _method(target, m):
    """getattr with private-name mangling; a target that already IS the named method is used as is
    (plans often write target 'Cls.method' plus method 'method'); absent = missing_target."""
    names = [m]
    if m.startswith("__") and not m.endswith("__"):
        cls = target if isinstance(target, type) else type(target)
        names.append(f"_{cls.__name__.lstrip('_')}{m}")
    for n in names:
        if hasattr(target, n):
            return getattr(target, n)
    if callable(target) and getattr(target, "__name__", None) == m:
        return target
    raise MissingTarget(f"method {m} not found at this commit")


# ---- call adapter ---------------------------------------------------------------------------
class Call:
    def __init__(self, mod, job, call):
        self.c, self.obj = call, resolve(mod, job, call["target"])

    def prepare(self, sv=None, scale=1.0, dt=None, edge=None, extra_kwargs=None):
        c, target = self.c, self.obj
        try:
            if "construct" in c:
                target = target(**{k: build(v, sv, scale, dt) for k, v in c["construct"].items()})
            if "set_attr" in c:
                setattr(target, c["set_attr"], sv)
            fn = _method(target, c["method"]) if "method" in c else target
            args = [build(a, sv, scale, dt) for a in c.get("args", [])]
            kwargs = {k: build(v, sv, scale, dt) for k, v in c.get("kwargs", {}).items()}
            kwargs.update(extra_kwargs or {})
            if edge:
                args, kwargs = _edge(edge, args, kwargs)
            try:
                inspect.signature(fn).bind(*args, **kwargs)
            except ValueError:
                pass  # builtins without a signature: bind at call time
        except O.HarnessError:
            raise
        except Exception as e:
            raise O.HarnessError(f"binding: {type(e).__name__}: {e}") from e
        kw = KW(kwargs)
        kw.fn = fn
        return (*args, kw)

    def run(self, *a):
        kw, args = a[-1], a[:-1]
        try:
            out = kw.fn(*args, **kw)
        except O.DECLARED:
            raise
        except O.BINDING as e:
            raise TargetRaised(f"{type(e).__name__}: {e}") from e
        o = self.c.get("output")
        if o is None:
            return out
        return out[o] if isinstance(o, int) or isinstance(out, dict) else getattr(out, o)

    def at(self, sv):
        return self.run(*self.prepare(sv=sv))


def _sweep(s):
    return list(s["values"]) if s.get("values") else list(range(s["start"], s["stop"], s.get("step", 1)))


def run_call(chk, mod, job):
    fam, p, C = chk["family"], chk["params"], Call(mod, job, chk["call"])
    if fam in ("bounds", "monotonic"):
        pts = _sweep(p["sweep"])
        if len(pts) > 5000:
            raise O.HarnessError("sweep over 5000 points")
        if fam == "monotonic":
            return [O.monotonic(C.at, pts, p.get("direction", "nonincreasing"))]
        cfg = _config_map(chk["call"], p)
        lo = _bound(p["lo"], cfg) if "lo" in p else -math.inf
        hi = _bound(p["hi"], cfg) if "hi" in p else math.inf
        return [O.bounds(C.at, pts, lo, hi)]
    if fam == "finite_extremes":
        return [O.finite_extremes(C.run, lambda s: C.prepare(scale=s), p["scales"], p["expect"],
                                  p.get("backward", False), rtol=_scale_rtol(chk))]
    if fam == "dtype_shadow":
        td = p["target_dtype"]
        mk = lambda d: C.prepare(dt=None if td == "int" and d != "wide" else d)
        return [O.dtype_shadow(C.run, mk, td, p.get("n_accum", 1))]
    if fam == "no_mutation":
        return [O.no_mutation(C.run, C.prepare, p.get("check_aliasing", False))]
    if fam == "edge_sweep":
        return [O.edge_sweep(C.run, {e: C.prepare(edge=e) for e in p["edges"]})]
    if fam == "reference":
        if p["ref"] not in REFS:
            raise O.HarnessError(f"reference {p['ref']} needs another adapter")
        ref = REFS[p["ref"]]
        mk = lambda **case: C.prepare(extra_kwargs={k: build(v) for k, v in case.items()})
        return [O.reference(C.run, lambda *a: ref(*a[:-1], **a[-1]), mk, p["cases"])]
    if fam == "decomposition":
        whole = C.prepare()[0]
        return [O.decomposition(lambda x: C.run(x, C.prepare()[-1]), whole, p["pieces"], p["combine"],
                                (p.get("p") or [2.0])[0])]
    raise O.HarnessError(f"family {fam} needs another adapter")


def _scale_rtol(chk):
    """scale_invariant tolerance from the arithmetic (5.2, 5.9): safety * eps(dtype) * sqrt(iters),
    dtype = narrowest of the tensor args and the declared compute_dtype, iters = first int arg or a
    steps/iters kwarg (default 1). ds-8533: 8 * eps(fp16) * sqrt(5) = 1.7e-2."""
    dts = [a["tensor"].get("dtype", "float32") for a in chk["call"].get("args", []) if "tensor" in a]
    dts.append(chk["params"].get("compute_dtype", "float32"))   # declared internal precision
    eps = max(O.EPS[DTYPES[d]] for d in dts if DTYPES[d].is_floating_point)
    it = [v["int"] for k, v in chk["call"].get("kwargs", {}).items() if "int" in v and k in ("steps", "iters", "num_iters")]
    it += [a["int"] for a in chk["call"].get("args", []) if "int" in a]
    return 8.0 * eps * math.sqrt(max(it[:1] or [1]))


# ---- grad_norm adapter (clip_grad_norm_-shaped functions) -----------------------------------
def run_grad_norm(chk, mod, job):
    fam, p, c = chk["family"], chk["params"], chk["call"]
    target = resolve(mod, job, c["target"])
    kwargs = {k: build(v) for k, v in c.get("kwargs", {}).items()}
    whole = torch.tensor([3.0, -4.0, 2.0])   # default pieces [3,-4],[2] (5.6)

    def norm_of(tensors, norm_type):
        params = []
        for t in tensors:
            q = torch.nn.Parameter(torch.zeros_like(t))
            q.grad = t.clone()
            params.append(q)
        try:
            return target(params, **{**kwargs, "norm_type": norm_type})
        except O.DECLARED:
            raise
        except O.BINDING as e:
            raise TargetRaised(f"{type(e).__name__}: {e}") from e

    if fam == "decomposition":
        out = []
        for pv in p.get("p") or [2.0]:
            v = O.decomposition(lambda ts, pv=pv: norm_of(ts, pv), whole, p.get("pieces", 2), p.get("combine", "same"), p=pv)
            v.detail = f"p={pv:g}: " + v.detail
            out.append(v)
        return out
    if fam == "reference":
        pieces = list(torch.tensor_split(whole, 2))
        out = []
        for case in p["cases"]:
            nt = float(build(case.get("norm_type", {"float": 2.0})))
            mk = lambda nt=nt: (pieces, nt)
            v = O.reference(lambda ts, n: norm_of(ts, n),
                            lambda ts, n: clip_grad_norm(ts, norm_type=n, max_norm=kwargs.get("max_norm", 1.0)),
                            mk, [{}])
            v.detail = f"p={nt:g}, grads [3,-4],[2]: got {v.witness.get('got', float('nan')):.6g}, must be {v.witness.get('want', float('nan')):.6g}; " + v.detail
            out.append(v)
        return out
    raise O.HarnessError(f"family {fam} is not supported by grad_norm")


ADAPTERS = {"call": run_call, "grad_norm": run_grad_norm}


def run_plan_check(chk, mod, job):
    if chk["adapter"] not in ADAPTERS:
        raise O.HarnessError(f"adapter {chk['adapter']} is not available for this target")
    return ADAPTERS[chk["adapter"]](chk, mod, job)
