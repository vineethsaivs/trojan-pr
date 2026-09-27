"""PLAN 8: stubbed LLM: valid -> planner; invalid then valid -> planner_repaired; garbage twice (both
attempts) -> prepass_default with mandatory checks only. Run on VM1: .venv/bin/python tests/control/test_planner.py"""
import json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("VULTR_INFERENCE_API_KEY", "stub")
from plumbline import planner
PRE = [{"file": "deepspeed/runtime/utils.py", "symbols": ["clip_grad_norm_"], "patterns": [], "families": ["reference"]}]
GOOD = {"summary": "s", "declared_behavior_change": False, "checks": [
    {"id": "c1", "family": "reference", "adapter": "grad_norm", "call": {"target": "deepspeed/runtime/utils.py:clip_grad_norm_",
     "kwargs": {"max_norm": {"float": 1.0}}}, "params": {"ref": "torch.clip_grad_norm_", "cases": [{"norm_type": {"float": 4.0}}]},
     "pattern": "power_root_order", "why": "w"}]}
BAD = {"summary": "s", "declared_behavior_change": False, "checks": [dict(GOOD["checks"][0], family="bounds")]}
def msg(args):
    tc = types.SimpleNamespace(id="t1", function=types.SimpleNamespace(arguments=args))
    m = types.SimpleNamespace(tool_calls=[tc] if args is not None else None, content="garbage" if args is None else None)
    m.model_dump = lambda **k: {"role": "assistant", "content": None}
    return m
def run(seq):
    it = iter(seq)
    planner.llm.chat = lambda *a, **k: (msg(next(it)), {"role": "planner", "model": "stub", "tokens_in": 1, "tokens_out": 1, "latency_ms": 1, "response_sha256": "x"})
    return planner.make_plan("diff", "desc", PRE, cap_s=45)
p = run([json.dumps(GOOD)]); assert p["source"] == "planner" and [c["id"] for c in p["checks"]] == ["m1", "m2", "c1"], p["source"]
p = run([json.dumps(BAD), json.dumps(GOOD)]); assert p["source"] == "planner_repaired", p["source"]
p = run([None, None, None, None]); assert p["source"] == "prepass_default" and [c["id"] for c in p["checks"]] == ["m1", "m2"], p
p = run([json.dumps(BAD), json.dumps(BAD), json.dumps(BAD), json.dumps(BAD)]); assert p["source"] == "prepass_default" and p["rejected"], p["source"]
print("test_planner: valid, repaired, garbage -> prepass_default, invalid-only -> rejected listed: PASS")
