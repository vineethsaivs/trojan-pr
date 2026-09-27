"""Bake-off runner (VM1), exactly as pre-registered in bakeoff/PROTOCOL.md. Resumable: one raw file
per job key case/which/model/prompt/run; existing files are skipped. Text only: nothing executes.
Run: python bakeoff/run.py [--runs 3] [--workers 8]"""
import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from plumbline import llm  # noqa: E402

INPUTS = os.environ.get("PLUMBLINE_INPUTS", "/var/lib/plumbline/inputs")
RAW = os.path.join(ROOT, "bakeoff", "raw")
H = ["ds-8313", "ds-8533", "st-3921", "ds-8334", "ray-65747", "st-4019",
     "st-3868", "ray-65535", "ray-65790", "zoo-1286", "unsloth-11337", "unsloth-11470"]
MODELS = json.load(open(os.path.join(ROOT, "bakeoff", "models.json")))["headline_reviewer"]
PROMPTS = {p: open(os.path.join(ROOT, "bakeoff", "prompts", f"{p}.txt")).read() for p in ("p0", "p1")}
MAX_TOKENS = 8000
GIT_SHA = open(os.path.join(ROOT, "GIT_SHA")).read().strip() if os.path.exists(os.path.join(ROOT, "GIT_SHA")) else "unknown"


def first_json(c):
    """First JSON object in the content (4.6)."""
    i = (c or "").find("{")
    while i >= 0:
        try:
            return json.JSONDecoder().raw_decode(c[i:])[0]
        except ValueError:
            i = c.find("{", i + 1)
    return None


def item(case, which):
    d = os.path.join(INPUTS, case, which)
    ci = json.load(open(os.path.join(INPUTS, case, "commits.json")))
    meta = json.load(open(os.path.join(d, "meta.json")))
    g2, src = open(os.path.join(d, "g2.diff")).read(), open(os.path.join(d, "scope.py")).read()
    scope = ci[which]["path"]
    user = (f"PR title: {meta.get('title', '')}\nPR body: {meta.get('body') or ''}\n\n<diff>{g2}</diff>\n"
            f"<file path=\"{scope}\">{src}</file>")
    return ci["repo"], user, {"g2": hashlib.sha256(g2.encode()).hexdigest(),
                              "scope": hashlib.sha256(src.encode()).hexdigest(),
                              "meta": hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest(),
                              "fetch_cmd": f"python bakeoff/fetch_inputs.py {case}"}


def one(job):
    case, which, model, prompt, run = job
    out = os.path.join(RAW, f"{case}-{which}", model, prompt, f"r{run}.json")
    if os.path.exists(out):
        return "skip"
    repo, user, sha = item(case, which)
    msgs = [{"role": "system", "content": PROMPTS[prompt].replace("{repo}", repo)}, {"role": "user", "content": user}]
    calls, status, parsed, content = [], "error", None, ""
    try:
        for turn in range(2):                     # first answer + one repair turn
            msg, rec = llm.chat("reviewer", model, msgs, max_tokens=MAX_TOKENS, temperature=None, timeout=300)
            calls.append(rec)
            content = msg.content or ""
            parsed = first_json(content)
            if parsed is not None:
                status = "ok" if turn == 0 else "repaired"
                break
            msgs += [{"role": "assistant", "content": content},
                     {"role": "user", "content": "Reply with only the JSON object described in the instructions."}]
    except Exception as e:
        status, content = "error", f"{type(e).__name__}: {str(e)[:300]}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"case": case, "which": which, "model": model, "prompt": prompt, "run": run, "status": status,
               "parsed": parsed, "content": content, "input_sha256": sha, "calls": calls,
               "timestamp": datetime.now(timezone.utc).isoformat(), "git_sha": GIT_SHA, "max_tokens": MAX_TOKENS},
              open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    return status


if __name__ == "__main__":
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 3
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    jobs = [(c, w, m, p, r) for r in range(1, runs + 1) for w in ("intro", "fix") for c in H for m in MODELS for p in PROMPTS]
    t0, counts = time.time(), {}
    with ThreadPoolExecutor(workers) as ex:
        for i, s in enumerate(ex.map(one, jobs), 1):
            counts[s] = counts.get(s, 0) + 1
            if i % 12 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {counts} {time.time() - t0:.0f} s", flush=True)
