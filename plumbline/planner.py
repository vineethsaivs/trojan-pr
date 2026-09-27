"""Planner (PLAN 5.4): glm-5.3 fills typed check templates via one tool, submit_plan. It never writes
code and never sets a tolerance. Mandatory checks from the prepass always run; the planner can only
add (<= 6). Invalid after one repair turn: keep only the checks that validate on their own
(source 'planner_partial', rejects listed); none valid or over 45 s: mandatory only ('prepass_default')."""
import hashlib, json, os, re, time
import jsonschema
from plumbline import llm

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(HERE, "prompts", "planner.txt")).read()
SCHEMA = json.load(open(os.path.join(HERE, "plan_schema.json")))
# glm-5.3 and glm-5.3-flash spend the whole token budget reasoning on this prompt (8k: no answer;
# glm-5.3 at 24k: 129 s, no answer; NOTES Sat 20:15), so the plan's third fallback leads.
MODELS = ("deepseek-v4-flash-0731", "deepseek-v4-flash-0731")   # second = fresh conversation
PLANNER_VERSION = "v1.2"   # v1.1: normalize typed args (amendment 3); v1.2: drop type-name outputs (amendment 4)
FAMS_BY_ADAPTER = {"call": {"bounds", "monotonic", "finite_extremes", "dtype_shadow", "no_mutation",
                            "edge_sweep", "reference", "decomposition"},
                   "grad_norm": {"decomposition", "reference"}}
ADAPTER_CATALOG = """call: target "path.py:Qualname" (module-level function or class). Optional: construct {kwarg: Arg} to instantiate the class; method "name" to call; set_attr "name" set to the sweep value before each call; args [Arg]; kwargs {name: Arg}; output int or key.
  bounds/monotonic params: {sweep: {start, stop, step?} or {start, stop, values: [..]}, lo/hi: number or {"config": key from a construct/kwargs dict}, direction: nonincreasing|nondecreasing}
  finite_extremes: {scales: [2-6 numbers], expect: finite|finite_nonzero|scale_invariant, backward?, compute_dtype?: float16|bfloat16 if the code computes internally in that dtype}; tensor args with "scaled": true are multiplied by each scale
  dtype_shadow: {target_dtype: float32|float16|bfloat16|int, n_accum?}
  no_mutation: {check_aliasing?}
  edge_sweep: {edges: [batch_1|empty|all_masked|all_empty_strings|single_token]} applied to the args
  reference: {ref: broadcast_numel|cross_entropy_ignore_index|k3_kl_float64|softplus_float64|logsumexp_float64, cases: [{kwarg: Arg}]}
  decomposition: {pieces, combine: sum|mean_by_count|pnorm}; the first positional arg is split
grad_norm: for clip_grad_norm_-shaped functions (list of params with .grad -> total norm). call: {target, kwargs: {max_norm: Arg}}.
  decomposition: {pieces, combine: "same", p: [numbers]}; reference: {ref: "torch.clip_grad_norm_", cases: [{"norm_type": {"float": p}}]}
Arg: {"tensor": {"shape": [..], "dist": randn|rand|zeros|ones|arange|const, "value", "dtype": float32|float16|bfloat16|float64|int64|bool, "seed", "scaled", "requires_grad"}} | {"list": {"n", "dist": randn|arange|const|empty_strings, "value"}} | {"float": x} | {"int": n} | {"str": s} | {"bool": b} | {"none": true} | {"sweep": true} | {"dict": {k: Arg}} | {"items": [Arg]} | {"object": {k: Arg}}"""

TOOL = {"type": "function", "function": {
    "name": "submit_plan", "description": "Submit the check plan. Exactly once.",
    "parameters": {"type": "object", "required": ["summary", "declared_behavior_change", "checks"], "properties": {
        "summary": {"type": "string"}, "declared_behavior_change": {"type": "boolean"},
        "checks": {"type": "array", "items": {"type": "object",
                   "required": ["id", "family", "adapter", "call", "params", "pattern", "why"], "properties": {
            "id": {"type": "string", "description": "c1, c2, ..."},
            "family": {"type": "string", "enum": SCHEMA["$defs"]["check"]["properties"]["family"]["enum"]},
            "adapter": {"type": "string", "enum": sorted(FAMS_BY_ADAPTER)},
            "call": {"type": "object", "description": "target, construct, method, set_attr, args, kwargs, output"},
            "params": {"type": "object", "description": "per-family parameters from the adapter catalog"},
            "pattern": {"type": "string"}, "why": {"type": "string"}}}}}}}}


def mandatory_checks(pre):
    """Adapter defaults exist only for clip_grad_norm_-shaped functions (5.3)."""
    out = []
    for f in pre:
        for s in f["symbols"]:
            if re.search(r"(^|\.)clip_grad_norm_?$", s):
                t = f"{f['file']}:{s}"
                out += [{"id": f"m{len(out) + 1}", "family": "decomposition", "adapter": "grad_norm",
                         "call": {"target": t, "kwargs": {"max_norm": {"float": 1000000.0}}},
                         "params": {"pieces": 2, "combine": "same", "p": [2, 1, 3]}, "pattern": "power_root_order",
                         "why": "mandatory: splitting the gradients must not change their p-norm"},
                        {"id": f"m{len(out) + 2}", "family": "reference", "adapter": "grad_norm",
                         "call": {"target": t, "kwargs": {"max_norm": {"float": 1.0}}},
                         "params": {"ref": "torch.clip_grad_norm_", "cases": [{"norm_type": {"float": p}} for p in (2.0, 1.0, 3.0)]},
                         "pattern": "power_root_order", "why": "mandatory: same maths as torch.nn.utils.clip_grad_norm_"}]
    return out


def _flat_config(call, params):
    keys = set()
    def walk(a):
        if isinstance(a, dict):
            for k, v in a.items():
                if k == "dict" and isinstance(v, dict):
                    keys.update(v)
                walk(v)
    walk(call.get("construct", {})); walk(call.get("kwargs", {})); keys.update(params.get("config", {}))
    return keys


def semantic_errors(plan, touched_files, surface=None):
    errs, ids = [], set()
    for c in plan.get("checks", []):
        cid = c.get("id")
        if cid in ids:
            errs.append(f"{cid}: duplicate id")
        ids.add(cid)
        path = c["call"]["target"].split(":")[0]
        if path not in touched_files and c["call"]["target"] not in (surface or {}):
            errs.append(f"{cid}: target file {path} is not touched by the diff")
        if c["family"] not in FAMS_BY_ADAPTER.get(c["adapter"], set()):
            errs.append(f"{cid}: adapter {c['adapter']} does not support {c['family']} here")
        ref = (c.get("params") or {}).get("ref")
        if c["family"] == "reference" and (ref == "torch.clip_grad_norm_") != (c["adapter"] == "grad_norm"):
            errs.append(f"{cid}: ref torch.clip_grad_norm_ goes with adapter grad_norm, other refs with call")
        p = c.get("params", {})
        for side in ("lo", "hi"):
            b = p.get(side)
            if isinstance(b, dict) and b.get("config") not in _flat_config(c["call"], p):
                errs.append(f"{cid}: {side} config key {b.get('config')} is not in the call's config")
        s = p.get("sweep")
        if s and not s.get("values") and (s["stop"] - s["start"]) / s.get("step", 1) > 5000:
            errs.append(f"{cid}: sweep over 5000 points")
    if len(plan.get("checks", [])) > 6:
        errs.append("more than 6 added checks")
    return errs


ARG_KINDS = {"tensor", "list", "float", "int", "str", "bool", "none", "sweep", "dict", "items", "object"}
PATTERNS = set(SCHEMA["$defs"]["check"]["properties"]["pattern"]["enum"])


def _arg(v):
    """Wrap a bare value into the typed Arg grammar. Mechanical only: never changes what runs."""
    if isinstance(v, dict):
        if len(v) == 1 and next(iter(v)) in ARG_KINDS:
            (k, x), = v.items()
            if k == "dict" and isinstance(x, dict):
                return {"dict": {n: _arg(a) for n, a in x.items()}}
            if k == "object" and isinstance(x, dict):
                return {"object": {n: _arg(a) for n, a in x.items()}}
            if k == "items" and isinstance(x, list):
                return {"items": [_arg(a) for a in x]}
            return v
        if "shape" in v:
            return {"tensor": v}
        return {"dict": {n: _arg(a) for n, a in v.items()}}
    if isinstance(v, bool):
        return {"bool": v}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, float):
        return {"float": v}
    if v is None:
        return {"none": True}
    if isinstance(v, str):
        return {"str": v}
    if isinstance(v, list):
        return {"items": [_arg(a) for a in v]}
    return v


def _normalize(c):
    call, p = c.get("call"), c.get("params")
    if isinstance(call, dict):
        for k in ("construct", "kwargs"):
            if isinstance(call.get(k), dict):
                call[k] = {n: _arg(a) for n, a in call[k].items()}
        if isinstance(call.get("args"), list):
            call["args"] = [_arg(a) for a in call["args"]]
        if call.get("output") in ("int", "float", "str", "bool", "tensor", "list", "dict", "number"):
            call.pop("output")                                        # a type hint, not a key to select
        t = call.get("target", "")
        if ":" in t and "construct" in call and "method" not in call:   # "path:Cls.method" + construct
            path, qual = t.split(":", 1)
            parts = qual.split(".")
            if len(parts) >= 2 and parts[-2][:1].isupper():
                call["target"], call["method"] = f"{path}:{'.'.join(parts[:-1])}", parts[-1]
    if isinstance(p, dict):
        if isinstance(p.get("cases"), list):
            p["cases"] = [{n: _arg(a) for n, a in case.items()} if isinstance(case, dict) else case for case in p["cases"]]
        if isinstance(p.get("config"), dict):
            p["config"] = {n: _arg(a) for n, a in p["config"].items()}
    if c.get("pattern") not in PATTERNS:                              # prose in the pattern field
        if isinstance(c.get("pattern"), str) and not c.get("why"):
            c["why"] = c["pattern"][:240]
        c["pattern"] = "none"


def _parse(msg):
    if msg.tool_calls:
        return json.loads(msg.tool_calls[0].function.arguments)
    m = re.search(r"\{.*\}", msg.content or "", re.S)   # JSON-in-content fallback
    return json.loads(m.group(0)) if m else None


def _errors(plan, touched_files, surface):
    if plan is None:
        return ["no submit_plan call and no JSON object in content"]
    def _unstr(v):                                                  # tool args sometimes arrive as JSON strings
        try:
            return json.JSONDecoder().raw_decode(v.strip())[0] if isinstance(v, str) and v.strip()[:1] in "[{" else v
        except ValueError:
            return v
    plan["checks"] = _unstr(plan.get("checks"))
    for c in plan["checks"] if isinstance(plan["checks"], list) else []:
        if isinstance(c, dict):
            c.update({k: _unstr(c[k]) for k in ("call", "params") if k in c})
            _normalize(c)
    plan["summary"] = str(plan.get("summary", ""))[:400]        # display-only text: trim, never reject
    for c in plan.get("checks", []) if isinstance(plan.get("checks"), list) else []:
        if isinstance(c, dict) and isinstance(c.get("why"), str):
            c["why"] = c["why"][:240]
    v = sorted(jsonschema.Draft202012Validator(SCHEMA).iter_errors(plan), key=lambda e: list(e.path))
    errs = [f"{'/'.join(map(str, e.path)) or 'plan'}: {e.message[:200]}" for e in v[:8]]
    return errs or semantic_errors(plan, touched_files, surface)


def make_plan(diff, description, pre, surface=None, config=None, cap_s=45, models=MODELS):
    mandatory = mandatory_checks(pre)
    touched = {f["file"] for f in pre}
    user = (PROMPT.replace("{diff}", diff).replace("{description}", description or "")
            .replace("{prepass_json}", json.dumps(pre)).replace("{mandatory_json}", json.dumps(mandatory))
            .replace("{adapter_catalog}", ADAPTER_CATALOG).replace("{surface_map}", json.dumps(surface or {}))
            .replace("{config_json}", json.dumps(config or {})))
    t0, calls, errs, model, plan, source = time.time(), [], [], None, None, "prepass_default"
    rejected = []
    deadline = t0 + cap_s
    for model in models:
        msgs, last = [{"role": "user", "content": user}], None
        try:
            for turn in range(2):                                   # first try + one repair turn
                msg, rec = llm.chat("planner", model, msgs, tools=[TOOL], max_tokens=8000, timeout=cap_s,
                                    deadline=deadline, tool_choice={"type": "function", "function": {"name": "submit_plan"}})
                calls.append(rec)
                try:
                    cand = _parse(msg)
                except ValueError as e:
                    cand, errs = None, [f"arguments are not JSON: {e}"]
                else:
                    errs = _errors(cand, touched, surface)
                    last = cand if isinstance(cand, dict) else last
                if not errs:
                    plan, source = cand, "planner" if turn == 0 else "planner_repaired"
                    break
                if msg.tool_calls:
                    msgs += [msg.model_dump(exclude_none=True),
                             {"role": "tool", "tool_call_id": msg.tool_calls[0].id,
                              "content": "INVALID PLAN:\n" + "\n".join(errs)}]
                msgs.append({"role": "user", "content": "The plan failed validation (errors above: "
                             + "; ".join(errs)[:1500] + "). Call submit_plan once more with a corrected plan."})
        except Exception as e:                                      # API error or deadline: next model
            errs = [f"{model}: {type(e).__name__}: {str(e)[:200]}"]
        if not plan and last and isinstance(last.get("checks"), list):   # keep checks that validate alone
            keep, rejected = [], []
            for c in last["checks"][:6]:
                one = {"summary": str(last.get("summary", ""))[:400],
                       "declared_behavior_change": bool(last.get("declared_behavior_change")), "checks": [c]}
                e = _errors(one, touched, surface)
                (rejected.append({"id": c.get("id") if isinstance(c, dict) else None, "errors": e[:3]}) if e else keep.append(c))
            if keep:
                plan, source = {**one, "checks": keep}, "planner_partial"
        if plan or time.time() > deadline - 5:
            break
    added = plan["checks"] if plan else []
    out = {"source": source, "planner_version": PLANNER_VERSION, "model": model if plan else None, "latency_ms": int((time.time() - t0) * 1000),
           "summary": plan["summary"] if plan else "planner unavailable: mandatory checks only",
           "declared_behavior_change": bool(plan and plan["declared_behavior_change"]),
           "checks": mandatory + added, "mandatory_ids": [c["id"] for c in mandatory],
           "errors": errs if not plan else [], "rejected": rejected, "model_calls": calls}
    out["plan_sha256"] = hashlib.sha256(json.dumps(out["checks"], sort_keys=True).encode()).hexdigest()
    return out
