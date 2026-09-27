# Oracle for deepspeedai/DeepSpeed#8334 (curriculum schedule starts below min_difficulty).
#
# The math (docs/_tutorials/curriculum-learning.md): fixed_linear / fixed_root ramp
#   d(t) = ((t/T) ** (1/root)) * (max - min) + min, quantized to multiples of difficulty_step,
# so the schedule lives in [min_difficulty, max_difficulty] and reaches max at t = T.
# Invariants checked (never reusing the implementation's floor/clamp code):
#   BOUND : min_difficulty <= d(t) <= max_difficulty for every t in [0, T]   (the bug check)
#   MONO  : d(t) is non-decreasing in t                                       (companion)
#   END   : d(T) == max_difficulty                                            (companion)
# Tolerance: zero. Everything is integer arithmetic (math.floor, int %), so equality and
# ordering are exact; any value below min is a violation, not rounding noise.
#
# Code is fetched with read-only `gh api .../contents?ref=SHA`, loaded with a stubbed
# deepspeed.utils.logger. Stdlib only, no torch, no GPU.
import base64, importlib.util, json, logging, subprocess, sys, types
from pathlib import Path

REPO = "deepspeedai/DeepSpeed"
DIR = "deepspeed/runtime/data_pipeline"
SRC = Path(__file__).parent / "src"
SHAS = [  # (label, sha, expected)
    ("pre-intro  504893a (parent of #1307)", "504893aea40004cf9916ddc3ca0ddbfd0e784c8d", "N/A"),
    ("introduce  b2b34ae (#1307)", "b2b34ae342d6f851226e995f2e1021d12e761093", "FAIL"),
    ("fix parent c7eed15", "c7eed1594d3de902796cf81ce6e7b84efe5979d8", "FAIL"),
    ("fix        8e09ed2 (#8334)", "8e09ed2acde363d5efcf3577f9586e9f4fcba672", "PASS"),
]
TRIGGER = [(8, 16), (1, 8), (10, 8)]   # min % step != 0
BENIGN = [(8, 8), (64, 16)]            # min % step == 0, false-positive controls
MAX, T = 1024, 100


def fetch(sha, name):
    out = SRC / sha[:7] / name
    if not out.exists():
        r = subprocess.run(["gh", "api", f"repos/{REPO}/contents/{DIR}/{name}?ref={sha}"],
                           capture_output=True, text=True)
        if r.returncode:
            return None  # 404: file does not exist at this SHA
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(json.loads(r.stdout)["content"]))
    return out


def load(sha):
    sched = fetch(sha, "curriculum_scheduler.py")
    if sched is None:
        return None
    du = types.ModuleType("deepspeed.utils"); du.logger = logging.getLogger("stub")
    pkg = types.ModuleType("dp"); pkg.__path__ = []
    for k in [k for k in sys.modules if k.startswith("dp.")]: del sys.modules[k]  # no stale constants
    sys.modules.update({"deepspeed": types.ModuleType("deepspeed"), "deepspeed.utils": du, "dp": pkg})
    for name in ("constants", "curriculum_scheduler"):  # constants.py only exists after #2585
        path = fetch(sha, f"{name}.py")
        if path:
            spec = importlib.util.spec_from_file_location(f"dp.{name}", path)
            mod = importlib.util.module_from_spec(spec); sys.modules[f"dp.{name}"] = mod
            spec.loader.exec_module(mod)
    return mod.CurriculumScheduler


def schedule(CS, mn, step, typ, total=T):
    cfg = {"curriculum_type": "seqlen", "min_difficulty": mn, "max_difficulty": MAX, "schedule_type": typ,
           "schedule_config": {"total_curriculum_step": total, "difficulty_step": step, "root_degree": 2}}
    s = CS(cfg)
    get = getattr(s, "get_difficulty", None) or s.update_difficulty  # API renamed in #1440
    return [get(t) for t in range(total + 1)]  # sequential, like the engine


def check(d, mn):
    bad = [t for t, v in enumerate(d) if not (mn <= v <= MAX)]
    errs = []
    if bad: errs.append(f"BOUND t={bad[0]} d={d[bad[0]]} < min={mn} ({len(bad)} steps)")
    if any(b < a for a, b in zip(d, d[1:])): errs.append("MONO")
    if d[-1] != MAX: errs.append(f"END d(T)={d[-1]}")
    return errs


ok_all = True
for label, sha, expected in SHAS:
    CS, fp = load(sha), 0
    if CS is None:
        verdict = "N/A"
        print(f"== {label}: curriculum_scheduler.py does not exist (404), no pre-introduction control")
    else:
        print(f"== {label}")
        fails = fp = 0
        for mn, step in TRIGGER + BENIGN:
            for typ in ("fixed_linear", "fixed_root"):
                d = schedule(CS, mn, step, typ)
                errs = check(d, mn)
                if (mn, step) in BENIGN: fp += bool(errs)  # false positive: oracle is wrong
                else: fails += bool(errs)
                print(f"  min={mn:<3} step={step:<3} {typ:12} d[0:4]={d[:4]} {'FAIL ' + '; '.join(errs) if errs else 'PASS'}")
        verdict = "FAIL" if fails else "PASS"
        print(f"  trigger configs failing: {fails}/{2 * len(TRIGGER)}, benign false positives: {fp}/{2 * len(BENIGN)}")
        # Tutorial-scale replay: min 8, INT8 step 16, T=15000; engine passes global_steps + 1.
        d = schedule(CS, 8, 16, "fixed_linear", 15000)
        print(f"  tutorial (min=8 step=16 T=15000): optimizer steps with d < min = {sum(v < 8 for v in d[1:])}")
    ok = verdict == expected and fp == 0
    ok_all &= ok
    print(f"  VERDICT {verdict} (expected {expected}) {'OK' if ok else 'MISMATCH'}")

print("ORACLE MATCHES HISTORY" if ok_all else "ORACLE DOES NOT MATCH HISTORY")
sys.exit(0 if ok_all else 1)
