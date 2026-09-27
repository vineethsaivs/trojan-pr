"""Sentinel control side (VM1 only): prepass -> planner -> one gVisor job per commit on VM2 ->
judge -> signed receipt -> SQLite. VM1 never executes PR or upstream code; it only fetches and
ships source. Historical mode replays a real PR: intro run = pre->intro (all 4 commits shown),
fix run = fixp->fix (the benign control)."""
import base64, hashlib, json, math, os, subprocess, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from plumbline import db, judge
from plumbline.fetch import fetch_closure, gh_raw
from plumbline.prepass import prepass

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "corpus")
INPUTS = os.environ.get("PLUMBLINE_INPUTS", "/var/lib/plumbline/inputs")
RECEIPTS = os.environ.get("PLUMBLINE_RECEIPTS", "/var/lib/plumbline/receipts")
SIGNING_KEY = os.environ.get("PLUMBLINE_SIGNING_KEY", "/etc/plumbline/signing.pem")
ROOTS = {"ds": ("deepspeed", ""), "ray": ("ray", "python/"), "st": ("sentence_transformers", ""),
         "unsloth": ("unsloth", ""), "zoo": ("unsloth_zoo", "")}
DOCKER_FLAGS = ("--runtime=runsc-trace --network none --read-only --tmpfs /work --tmpfs /tmp --cpus 1 --memory 3g "
                "--pids-limit 256 --cap-drop ALL --security-opt no-new-privileges --user 10001:10001")


def corpus_case(cid):
    """cid like ds-8313; corpus files use deepspeed-8313."""
    fname = cid.replace("ds-", "deepspeed-", 1) + ".yaml"
    c = dict(l.split(": ", 1) for l in open(os.path.join(CORPUS, fname)).read().splitlines() if ": " in l)
    c["root_pkg"], c["prefix"] = ROOTS[cid.split("-")[0]]
    c["repo"] = c["repo"].replace("https://github.com/", "")
    return c


def make_job(job_id, case, sha, checks, scope=None, per_check_timeout_s=60):
    full = scope or case["scope"]
    path = full[len(case["prefix"]):] if full.startswith(case["prefix"]) else full
    files = fetch_closure(case["repo"], sha, path, case["root_pkg"], src_prefix=case["prefix"])
    if path not in files:
        raise FileNotFoundError(f"{path} not found at {sha[:12]}")
    return {"job_id": job_id, "mode": "extract", "checks": checks, "files": files, "path": full,
            "root_pkg": case["root_pkg"], "module": path[:-3].replace("/", "."), "seed": 0,
            "per_check_timeout_s": per_check_timeout_s, "repo": case["repo"], "sha": sha}


def dispatch(job, harness="harness", timeout_s=300):
    import httpx  # VM1 venv only; job building needs no third-party deps
    body = {"job_id": job["job_id"], "harness": harness, "timeout_s": timeout_s,
            "files": {"job.json": base64.b64encode(json.dumps(job).encode()).decode()}}
    r = httpx.post(os.environ["SANDBOXD_URL"] + "/v1/jobs", json=body, timeout=timeout_s + 30,
                   headers={"Authorization": f"Bearer {os.environ['SANDBOXD_TOKEN']}"})
    r.raise_for_status()
    return r.json()


def gold_run(cid, sha):
    job = make_job(f"gold-{cid}-{sha[:8]}-{int(time.time())}", corpus_case(cid), sha,
                   [{"id": "g1", "gold": cid}])
    return dispatch(job)


# ---- historical PR mode ---------------------------------------------------------------------
def case_info(cid):
    ci = json.load(open(os.path.join(INPUTS, cid, "commits.json")))
    ci["root_pkg"], ci["prefix"] = ROOTS[cid.split("-")[0]]
    return ci


def _retarget(checks, old_path, new_path):
    out = json.loads(json.dumps(checks))
    for c in out:
        t = c["call"]["target"]
        if old_path and new_path and t.startswith(old_path + ":"):
            c["call"]["target"] = new_path + t[len(old_path):]
    return out


def run_commit(run_id, ci, label, checks, plan_path, harness="harness"):
    cm = ci["commits"][label]
    if not cm["path"]:   # file absent at this commit: every check's target is missing
        lines = [{"ev": "check", "id": c["id"], "family": c["family"], "status": "ERROR", "metric": None,
                  "threshold": None, "detail": "scope file absent at this commit", "witness": {"kind": "missing_target"}}
                 for c in checks]
        return {"label": label, "sha": cm["sha"], "lines": lines, "job": None}
    job = make_job(f"{run_id}-{label}", ci, cm["sha"], _retarget(checks, plan_path, cm["path"]), scope=cm["path"])
    res = dispatch(job, harness=harness)
    return {"label": label, "sha": cm["sha"], "lines": res["lines"], "job": res}


def _status(cells):
    st = [c["status"] for c in cells.values()]
    return "FAIL" if "FAIL" in st else "PASS" if "PASS" in st else "ERROR"


def _clean(x):
    """Receipt-safe JSON: non-finite floats become strings (canonical JSON forbids NaN)."""
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _infra():
    info = {"public_url": os.environ.get("PUBLIC_URL")}
    try:
        import httpx
        m = httpx.get("http://169.254.169.254/v1.json", timeout=2).json()
        info.update(vultr_instance=m.get("instanceid"), region=(m.get("region") or {}).get("regioncode"))
    except Exception as e:
        info["vultr_metadata_error"] = type(e).__name__
    try:
        info["netbird_ip"] = subprocess.run(["netbird", "status", "--ipv4"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        info["netbird_ip"] = None
    return info


def _git_sha():
    p = os.path.join(HERE, "..", "GIT_SHA")
    return open(p).read().strip() if os.path.exists(p) else "unknown"


def run_pr(cid, which="intro", harness="harness", run_id=None, conn=None, cap_s=45, plan=None):
    """plan: re-execute a stored plan as is (amendment 5), no planner call."""
    from plumbline import receipt                 # VM1 venv deps (cryptography, openai, jsonschema)
    from plumbline.planner import make_plan
    t0 = time.time()
    ci = case_info(cid)
    d = os.path.join(INPUTS, cid, which)
    diff, meta = open(os.path.join(d, "g2.diff")).read(), json.load(open(os.path.join(d, "meta.json")))
    base_l, head_l = ("pre", "intro") if which == "intro" else ("fixp", "fix")
    labels = ["pre", "intro", "fixp", "fix"] if which == "intro" else ["fixp", "fix"]
    hp, bp = ci["commits"][head_l]["path"], ci["commits"][base_l]["path"]
    run_id = run_id or f"{cid}-{which}-{uuid.uuid4().hex[:6]}"
    conn = conn or db.connect()
    subject = {"repo": ci["repo"], "pr": meta.get("number"), "url": meta.get("url"), "title": meta.get("title"),
               "base_sha": ci["commits"][base_l]["sha"], "head_sha": ci["commits"][head_l]["sha"],
               "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(), "kind": "hist", "case": cid, "which": which,
               "harness": harness, "planner_version": __import__("plumbline.planner", fromlist=["x"]).PLANNER_VERSION}
    if plan:
        subject["plan_from"] = plan.get("from_run")      # stored blind plan, re-executed as is
    db.upsert(conn, "runs", id=run_id, kind="hist", case_id=cid, subject=subject, status="planning",
              created_at=datetime.now(timezone.utc).isoformat())
    head_src = open(os.path.join(d, "scope.py")).read()
    base_src = gh_raw(ci["repo"], bp, ci["commits"][base_l]["sha"]) if bp else None
    pre = prepass(diff, {hp: head_src}, {bp: base_src} if base_src else None)
    if not plan:
        plan = make_plan(diff, f"{meta.get('title', '')}\n\n{meta.get('body') or ''}", pre, cap_s=cap_s)
    checks = plan["checks"]
    db.update_run(conn, run_id, status="running", plan={k: v for k, v in plan.items() if k != "model_calls"})
    ids = {c["id"] for c in checks}
    targets = {c["id"]: c["call"]["target"] for c in checks}
    res, per = {}, {}
    with ThreadPoolExecutor(4) as ex:   # each commit's cells land as soon as its job returns (live grid)
        futs = [ex.submit(run_commit, run_id, ci, l, checks, hp, harness) for l in labels]
        for f in as_completed(futs):
            r = f.result()
            res[r["label"]], per[r["label"]] = r, judge.cells(r["lines"], ids)
            for k, c in per[r["label"]].items():
                db.upsert(conn, "cells", run_id=run_id, check_id=k, commit_label=r["label"], sha=r["sha"],
                          family=c.get("family"), target=targets.get(k.split(".")[0]), status=c["status"],
                          metric=json.dumps(_clean(c.get("metric"))), threshold=json.dumps(_clean(c.get("threshold"))),
                          detail=c.get("detail"), witness=_clean(c.get("witness") or {}), decision=None)
    base, head = per[base_l], per[head_l]
    decisions = {cid_: judge.decide(base.get(cid_) or base.get(cid_.split(".")[0]), h) for cid_, h in head.items()}
    touched = [f"{f['file']}:{s}" for f in pre for s in f["symbols"]]
    verdict, reason = judge.verdict(decisions, targets, touched)
    results = []
    for l in labels:
        for k, c in per[l].items():
            row = {"check_id": k, "commit": l, "sha": res[l]["sha"], "family": c.get("family"), "status": c["status"],
                   "metric": c.get("metric"), "threshold": c.get("threshold"), "detail": c.get("detail"),
                   "decision": decisions.get(k) if l == head_l else None}
            results.append(row)
            db.upsert(conn, "cells", run_id=run_id, check_id=k, commit_label=l, sha=res[l]["sha"], family=c.get("family"),
                      target=targets.get(k.split(".")[0]), status=c["status"], metric=json.dumps(_clean(c.get("metric"))),
                      threshold=json.dumps(_clean(c.get("threshold"))), detail=c.get("detail"),
                      witness=_clean(c.get("witness") or {}), decision=row["decision"])
    jobs = [r["job"] for r in res.values() if r["job"]]
    j0 = jobs[0] if jobs else {}
    rec = _clean({"receipt_version": "plumbline/1", "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
                  "subject": subject, "verdict": verdict, "reason": reason,
                  "statuses": {l: _status(per[l]) for l in labels},
                  "plan": {k: plan[k] for k in ("source", "model", "latency_ms", "plan_sha256", "summary",
                                                 "declared_behavior_change", "checks", "mandatory_ids", "errors", "rejected")},
                  "results": results,
                  "execution": {"sandbox_host": os.environ["SANDBOXD_URL"], "runtime": j0.get("runtime"),
                                "image_id": j0.get("image_id"), "harness": harness, "harness_sha256": j0.get("harness_sha256"),
                                "docker_flags": DOCKER_FLAGS,
                                "jobs": [{"job_id": j["job_id"], "exit": j["exit_code"], "wall_ms": j["wall_ms"]} for j in jobs]},
                  "model_calls": plan["model_calls"], "infra": _infra(), "code": {"git_sha": _git_sha()}})
    rec = receipt.sign(rec, open(SIGNING_KEY, "rb").read())
    os.makedirs(RECEIPTS, exist_ok=True)
    path = os.path.join(RECEIPTS, f"{run_id}.json")
    json.dump(rec, open(path, "w"), indent=1, sort_keys=True)
    db.upsert(conn, "receipts", run_id=run_id, body=json.dumps(rec, sort_keys=True))
    wall = int((time.time() - t0) * 1000)
    db.update_run(conn, run_id, status="done", verdict=verdict, reason=reason, wall_ms=wall,
                  finished_at=datetime.now(timezone.utc).isoformat())
    return {"run_id": run_id, "plan": plan, "statuses": rec["statuses"], "verdict": verdict, "reason": reason,
            "receipt_path": path, "wall_ms": wall, "per": per, "decisions": decisions, "labels": labels}


# ---- repo mode (minigpt game PRs): base vs head, the whole small repo ships in each job -----------
SURFACE = None


def surface():
    global SURFACE
    if SURFACE is None:
        import yaml
        SURFACE = yaml.safe_load(open(os.path.join(HERE, "..", "harness", "surface", "minigpt.yaml")))
    return SURFACE


def run_repo(pr_id, base_files, head_files, diff, title, description, kind, harness="harness", conn=None, cap_s=45):
    """kind: hand | saboteur | honest | judge. base_files/head_files: {relpath: source} of target/minigpt."""
    from plumbline import receipt
    from plumbline.planner import make_plan, PLANNER_VERSION
    t0 = time.time()
    run_id = f"{pr_id}-{uuid.uuid4().hex[:6]}"
    conn = conn or db.connect()
    subject = {"repo": "target/minigpt", "pr": pr_id, "title": title, "kind": kind, "which": "repo",
               "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(), "harness": harness, "planner_version": PLANNER_VERSION}
    db.upsert(conn, "runs", id=run_id, kind=kind, case_id=pr_id, subject=subject, status="planning",
              created_at=datetime.now(timezone.utc).isoformat())
    pre = prepass(diff, {p: s for p, s in head_files.items() if p.endswith(".py")},
                  {p: s for p, s in base_files.items() if p.endswith(".py")}, surface())
    plan = make_plan(diff, f"{title}\n\n{description}", pre, surface=surface(), cap_s=cap_s)
    checks, ids = plan["checks"], {c["id"] for c in plan["checks"]}
    targets = {c["id"]: c["call"]["target"] for c in checks}
    db.update_run(conn, run_id, status="running", plan={k: v for k, v in plan.items() if k != "model_calls"})
    res, per = {}, {}
    def one(label, files):
        job = {"job_id": f"{run_id}-{label}", "mode": "repo", "checks": checks, "files": files, "path": None,
               "module": None, "root_pkg": None, "seed": 0, "per_check_timeout_s": 60}
        return label, dispatch(job, harness=harness)
    with ThreadPoolExecutor(2) as ex:
        for f in as_completed([ex.submit(one, "base", base_files), ex.submit(one, "head", head_files)]):
            label, r = f.result()
            res[label], per[label] = r, judge.cells(r["lines"], ids)
            for k, c in per[label].items():
                db.upsert(conn, "cells", run_id=run_id, check_id=k, commit_label=label, sha=None, family=c.get("family"),
                          target=targets.get(k.split(".")[0]), status=c["status"], metric=json.dumps(_clean(c.get("metric"))),
                          threshold=json.dumps(_clean(c.get("threshold"))), detail=c.get("detail"),
                          witness=_clean(c.get("witness") or {}), decision=None)
    base, head = per["base"], per["head"]
    decisions = {k: judge.decide(base.get(k) or base.get(k.split(".")[0]), h) for k, h in head.items()}
    touched = [f"{f['file']}:{s}" for f in pre for s in f["symbols"]]
    verdict, reason = judge.verdict(decisions, targets, touched)
    for k, d in decisions.items():
        conn.execute("UPDATE cells SET decision=? WHERE run_id=? AND check_id=? AND commit_label='head'", (d, run_id, k))
    conn.commit()
    results = [{"check_id": k, "commit": l, "family": c.get("family"), "status": c["status"], "metric": c.get("metric"),
                "threshold": c.get("threshold"), "detail": c.get("detail"), "decision": decisions.get(k) if l == "head" else None}
               for l in ("base", "head") for k, c in per[l].items()]
    j0 = res["head"]
    rec = _clean({"receipt_version": "plumbline/1", "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
                  "subject": subject, "verdict": verdict, "reason": reason,
                  "statuses": {l: _status(per[l]) for l in ("base", "head")},
                  "plan": {k: plan.get(k) for k in ("source", "planner_version", "model", "latency_ms", "plan_sha256", "summary",
                                                     "declared_behavior_change", "checks", "mandatory_ids", "errors", "rejected")},
                  "results": results,
                  "execution": {"sandbox_host": os.environ["SANDBOXD_URL"], "runtime": j0.get("runtime"), "image_id": j0.get("image_id"),
                                "harness": harness, "harness_sha256": j0.get("harness_sha256"), "docker_flags": DOCKER_FLAGS,
                                "jobs": [{"job_id": r["job_id"], "exit": r["exit_code"], "wall_ms": r["wall_ms"]} for r in res.values()]},
                  "model_calls": plan["model_calls"], "infra": _infra(), "code": {"git_sha": _git_sha()}})
    rec = receipt.sign(rec, open(SIGNING_KEY, "rb").read())
    os.makedirs(RECEIPTS, exist_ok=True)
    path = os.path.join(RECEIPTS, f"{run_id}.json")
    json.dump(rec, open(path, "w"), indent=1, sort_keys=True)
    db.upsert(conn, "receipts", run_id=run_id, body=json.dumps(rec, sort_keys=True))
    wall = int((time.time() - t0) * 1000)
    db.update_run(conn, run_id, status="done", verdict=verdict, reason=reason, wall_ms=wall,
                  finished_at=datetime.now(timezone.utc).isoformat())
    return {"run_id": run_id, "verdict": verdict, "reason": reason, "plan": plan, "per": per, "decisions": decisions,
            "receipt_path": path, "wall_ms": wall}


def ci_run(job_id, files, harness="harness", timeout_s=300):
    """pytest -q tests on the given repo files inside gVisor. -> (exit, tail)."""
    r = dispatch({"job_id": job_id, "mode": "ci", "checks": [], "files": files, "seed": 0}, harness=harness, timeout_s=timeout_s)
    ci = next((l for l in r["lines"] if l.get("ev") == "ci"), None)
    return (ci["exit"], ci["tail"]) if ci else (r["exit_code"], r["stderr_tail"][-2000:])


if __name__ == "__main__":
    # python -m plumbline.sentinel gold <case-id> <sha> [<sha> ...]
    cid, shas = sys.argv[2], sys.argv[3:]
    for sha in shas:
        res = gold_run(cid, sha)
        checks = [l for l in res["lines"] if l.get("ev") == "check"]
        print(json.dumps({"case": cid, "sha": sha[:12], "exit": res["exit_code"], "wall_ms": res["wall_ms"],
                          "statuses": [c["status"] for c in checks],
                          "details": [c["detail"][:90] for c in checks], "runtime": res["runtime"]}))
