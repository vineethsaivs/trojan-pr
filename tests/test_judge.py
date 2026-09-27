"""PLAN 5.7 decision table + 'sandbox says PASS but metric > threshold -> detected'.
Run: python3 tests/test_judge.py (stdlib only)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plumbline.judge import cells, decide, recompute, verdict

P, F = {"status": "PASS"}, {"status": "FAIL"}
E_MISSING, E_OTHER = {"status": "ERROR", "kind": "missing_target"}, {"status": "ERROR", "kind": "binding"}
rows = [(P, F, "detected"), (P, P, "clean"), (F, F, "preexisting"), (F, P, "improved"),
        (E_MISSING, F, "detected_new"), (E_MISSING, P, "clean_new"), (E_OTHER, F, "inconclusive"),
        (P, E_OTHER, "inconclusive"), (None, P, "inconclusive")]
for b, h, want in rows:
    got = decide(b, h)
    assert got == want, (b, h, got, want)

# the sandbox's PASS is not trusted when the numbers disagree
lying = {"ev": "check", "id": "c1", "status": "PASS", "metric": 4.89, "threshold": 9.52e-07}
assert recompute(lying) == "FAIL"
base = cells([{"ev": "check", "id": "c1", "status": "PASS", "metric": 0.0, "threshold": 9.52e-07}], {"c1"})
head = cells([lying], {"c1"})
assert decide(base["c1"], head["c1"]) == "detected"
assert recompute({"status": "PASS", "metric": "nan", "threshold": 1.0}) == "FAIL"
assert recompute({"status": "PASS", "metric": None, "threshold": None}) == "PASS"   # categorical

# tampering: a duplicate id (code under test printing a fake PASS) becomes ERROR
dup = cells([{"ev": "check", "id": "c1", "status": "FAIL", "metric": 1.0, "threshold": 0.1},
             {"ev": "check", "id": "c1", "status": "PASS", "metric": 0.0, "threshold": 0.1}], {"c1"})
assert dup["c1"]["status"] == "ERROR" and dup["c1"]["kind"] == "tamper"
assert cells([{"ev": "check", "id": "zz", "status": "PASS"}], {"c1"})["zz"]["status"] == "ERROR"
assert cells([{"ev": "check", "id": "c1.2", "status": "PASS"}], {"c1"})["c1.2"]["status"] == "PASS"

T = {"c1": "a.py:f", "c2": "a.py:Cls"}
assert verdict({"c1": "detected", "c2": "clean"}, T, ["a.py:f"])[0] == "BLOCK"
assert verdict({"c1": "clean"}, T, ["a.py:f"])[0] == "PASS"
assert verdict({"c2.0": "clean_new"}, T, ["a.py:Cls.method"])[0] == "PASS"
assert verdict({"c1": "inconclusive"}, T, ["a.py:f"])[0] == "NOT COVERED"
assert verdict({"c1": "clean"}, T, ["a.py:f", "a.py:g"])[0] == "NOT COVERED"
assert verdict({}, T, [])[0] == "NOT COVERED"
assert verdict({"c1": "preexisting"}, T, ["a.py:f"])[0] == "NOT COVERED"
print(f"test_judge: {len(rows)} table rows + lying sandbox + tamper + 7 verdict cases: PASS")
