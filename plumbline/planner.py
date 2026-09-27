"""Planner (PLAN 5.4): glm-5.3 fills typed check templates via one tool, submit_plan. It never writes
code and never sets a tolerance. Mandatory checks from the prepass always run; the planner can only
add (<= 6). Invalid after one repair turn, or over 45 s: mandatory only, source 'prepass_default'."""
import hashlib, json, os, re, time
import jsonschema
from plumbline import llm

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(HERE, "prompts", "planner.txt")).read()
SCHEMA = json.load(open(os.path.join(HERE, "plan_schema.json")))
MODELS = ("glm-5.3", "glm-5.3-flash", "deepseek-v4-flash-0731")
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


def _parse(msg):
    if msg.tool_calls:
        return json.loads(msg.tool_calls[0].function.arguments)
    m = re.search(r"\{.*\}", msg.content or "", re.S)   # JSON-in-content fallback
    return json.loads(m.group(0)) if m else None


def _errors(plan, touched_files, surface):
    if plan is None:
        return ["no submit_plan call and no JSON object in content"]
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
    for model in models:
        msgs = [{"role": "user", "content": user}]
        try:
            for turn in range(2):                                   # first try + one repair turn
                left = cap_s - (time.time() - t0)
                if left < 5:
                    raise TimeoutError("planner cap reached")
                msg, rec = llm.chat("planner", model, msgs, tools=[TOOL], max_tokens=8000, timeout=left,
                                    tool_choice={"type": "function", "function": {"name": "submit_plan"}})
                calls.append(rec)
                try:
                    cand = _parse(msg)
                except ValueError as e:
                    cand, errs = None, [f"arguments are not JSON: {e}"]
                else:
                    errs = _errors(cand, touched, surface)
                if not errs:
                    plan, source = cand, "planner" if turn == 0 else "planner_repaired"
                    break
                if msg.tool_calls:
                    msgs += [msg.model_dump(exclude_none=True),
                             {"role": "tool", "tool_call_id": msg.tool_calls[0].id,
                              "content": "INVALID PLAN:\n" + "\n".join(errs)}]
                msgs.append({"role": "user", "content": "The plan failed validation (errors above: "
                             + "; ".join(errs)[:1500] + "). Call submit_plan once more with a corrected plan."})
            if plan or time.time() - t0 > cap_s - 5:
                break
        except Exception as e:                                      # API error or cap: try the next model
            errs = [f"{model}: {type(e).__name__}: {str(e)[:200]}"]
            if time.time() - t0 > cap_s - 5:
                break
    added = plan["checks"] if plan else []
    out = {"source": source, "model": model if plan else None, "latency_ms": int((time.time() - t0) * 1000),
           "summary": plan["summary"] if plan else "planner unavailable: mandatory checks only",
           "declared_behavior_change": bool(plan and plan["declared_behavior_change"]),
           "checks": mandatory + added, "mandatory_ids": [c["id"] for c in mandatory],
           "errors": errs if not plan else [], "model_calls": calls}
    out["plan_sha256"] = hashlib.sha256(json.dumps(out["checks"], sort_keys=True).encode()).hexdigest()
    return out
