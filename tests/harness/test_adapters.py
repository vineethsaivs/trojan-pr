"""Dev-plan adapter test: each typed plan must FAIL at the corpus buggy SHA and PASS at the fixed
SHA, through runjob exactly as the sandbox runs it. Laptop: needs torch and proto/cache.
Run: PLUMBLINE_CACHE=proto/cache <python-with-torch> tests/harness/test_adapters.py"""
import json, os, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from plumbline.sentinel import corpus_case, make_job

plans = json.load(open(os.path.join(ROOT, "tests/harness/dev_plans.json")))
bad = 0
for cid, checks in plans.items():
    c = corpus_case(cid)
    for which, want in (("buggy_sha", "F"), ("fixed_sha", "P")):
        job = make_job("t", c, c[which], checks)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(job, f)
        out = subprocess.run([sys.executable, "runjob.py", f.name], cwd=os.path.join(ROOT, "harness"),
                             capture_output=True, text=True).stdout
        lines = [json.loads(l) for l in out.splitlines() if l.strip()]
        st = "".join(l["status"][0] for l in lines if l["ev"] == "check")
        ok = (want in st) if want == "F" else (set(st) == {"P"})
        bad += not ok
        print(f"{cid:10} {which[:5]} {st:6} {'ok' if ok else 'WRONG'}",
              [l["detail"][:70] for l in lines if l["ev"] == "check" and l["status"] != "PASS"][:2])
print("RESULT", "PASS" if not bad else f"FAIL ({bad} wrong)")
sys.exit(1 if bad else 0)
