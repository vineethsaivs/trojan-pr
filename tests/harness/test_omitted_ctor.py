"""Amendment 5: a required constructor arg the plan omits gets a stand-in. A method that never uses it
runs clean (a NaN it returns still FAILs); any use of the stand-in is ERROR, never FAIL.
Run: <python-with-torch> tests/harness/test_omitted_ctor.py (VM2: pl/runner:cpu under runsc)"""
import json, os, subprocess, sys, tempfile
HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "harness")
SRC = ("class Loss:\n    def __init__(self, model, w=1.0):\n        self.model, self.w = model, w\n"
       "    def ratio(self, n):\n        return self.w * n / (n * (n - 1))\n"
       "    def uses(self, n):\n        return self.model.encode(n)\n"
       "    def mults(self, n):\n        return self.model * n\n"
       "class Eager:\n    def __init__(self, model):\n        self.dim = model.dim\n    def f(self, n):\n        return n\n")
def chk(cid, method, sweep):
    return {"id": cid, "family": "bounds", "adapter": "call", "pattern": "none", "why": "t",
            "call": {"target": "toy/m.py:" + ("Eager" if method == "f" else "Loss"), "construct": {}, "method": method,
                     "args": [{"sweep": True}]}, "params": {"sweep": {"values": sweep}, "lo": -10.0, "hi": 10.0}}
checks = [chk("c1", "ratio", [2, 3]), chk("c2", "ratio", [1]), chk("c3", "uses", [2]), chk("c4", "mults", [2]),
          chk("c5", "f", [2])]
job = {"job_id": "t", "mode": "extract", "checks": checks, "files": {"toy/__init__.py": "", "toy/m.py": SRC},
       "root_pkg": "toy", "module": "toy.m", "path": "toy/m.py", "seed": 0, "per_check_timeout_s": 10}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(job, f)
out = [json.loads(l) for l in subprocess.run([sys.executable, "runjob.py", f.name], cwd=HARNESS, capture_output=True, text=True).stdout.splitlines()]
c = {l["id"]: l for l in out if l["ev"] == "check"}
assert c["c1"]["status"] == "PASS", c["c1"]                      # stand-in never touched: clean check
assert c["c2"]["status"] == "FAIL", c["c2"]                      # n=1: 1/0 still a detection
for k in ("c3", "c4", "c5"):                                    # attribute, arithmetic, use in __init__
    assert c[k]["status"] == "ERROR", c[k]
print("test_omitted_ctor: unused stand-in clean, real bug still FAIL, any use of the stand-in ERROR: PASS")
