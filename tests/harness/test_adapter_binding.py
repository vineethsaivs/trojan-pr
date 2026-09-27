"""Binding shapes the planner produced on dev cases (Sat 20:10): private (name-mangled) method, target
that already is the method (staticmethod), and an absent method -> ERROR kind missing_target.
Run: <python-with-torch> tests/harness/test_adapter_binding.py"""
import json, os, subprocess, sys, tempfile
HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "harness")
SRC = ("class Sched:\n    def __init__(self, lo):\n        self.lo = lo\n    def __step(self, x):\n        return self.lo + x\n"
       "class Loss:\n    @staticmethod\n    def value(x):\n        return 0.25\n")
def chk(cid, call):
    return {"id": cid, "family": "bounds", "adapter": "call", "call": {"target": "toy/m.py:" + call.pop("t"), **call},
            "params": {"sweep": {"start": 0, "stop": 3}, "lo": 0.0, "hi": 10.0}, "pattern": "none", "why": "t"}
checks = [chk("c1", {"t": "Sched", "construct": {"lo": {"int": 1}}, "method": "__step", "args": [{"sweep": True}]}),
          chk("c2", {"t": "Loss.value", "method": "value", "args": [{"sweep": True}]}),
          chk("c3", {"t": "Sched", "construct": {"lo": {"int": 1}}, "method": "gone", "args": [{"sweep": True}]})]
job = {"job_id": "t", "mode": "extract", "checks": checks, "files": {"toy/__init__.py": "", "toy/m.py": SRC},
       "root_pkg": "toy", "module": "toy.m", "path": "toy/m.py", "seed": 0, "per_check_timeout_s": 10}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(job, f)
out = [json.loads(l) for l in subprocess.run([sys.executable, "runjob.py", f.name], cwd=HARNESS, capture_output=True, text=True).stdout.splitlines()]
c = {l["id"]: l for l in out if l["ev"] == "check"}
assert c["c1"]["status"] == "PASS", c["c1"]
assert c["c2"]["status"] == "PASS", c["c2"]
assert c["c3"]["status"] == "ERROR" and c["c3"]["witness"]["kind"] == "missing_target", c["c3"]
print("test_adapter_binding: mangled private method, target-is-method, absent -> missing_target: PASS")
