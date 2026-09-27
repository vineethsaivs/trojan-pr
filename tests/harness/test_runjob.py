"""PLAN 8: a one-check job prints started, check, done; a target that sleeps gives ERROR timeout.
Run: <python-with-torch> tests/harness/test_runjob.py"""
import json, os, subprocess, sys, tempfile
HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "harness")
SRC = "import time\ndef ok(x):\n    return 0.5\ndef slow(x):\n    time.sleep(99)\n    return 0.5\n"
def check(cid, fn):
    return {"id": cid, "family": "bounds", "adapter": "call", "call": {"target": f"toy/m.py:{fn}", "args": [{"sweep": True}]},
            "params": {"sweep": {"start": 0, "stop": 3}, "lo": 0.0, "hi": 1.0}, "pattern": "none", "why": "t"}
job = {"job_id": "t", "mode": "extract", "checks": [check("c1", "ok"), check("c2", "slow")], "files": {"toy/__init__.py": "", "toy/m.py": SRC},
       "root_pkg": "toy", "module": "toy.m", "path": "toy/m.py", "seed": 0, "per_check_timeout_s": 2}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(job, f)
lines = [json.loads(l) for l in subprocess.run([sys.executable, "runjob.py", f.name], cwd=HARNESS, capture_output=True, text=True, timeout=60).stdout.splitlines()]
evs = [l["ev"] for l in lines]
assert evs == ["started", "check", "check", "done"], evs
c1, c2 = lines[1], lines[2]
assert c1["id"] == "c1" and c1["status"] == "PASS", c1
assert c2["id"] == "c2" and c2["status"] == "ERROR" and c2["witness"]["kind"] == "timeout", c2
print("test_runjob: started/check/done; sleeping target -> ERROR timeout in", c2["ms"], "ms: PASS")
