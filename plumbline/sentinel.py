"""Sentinel control side (VM1 only): build a sandbox job for (case, commit) and dispatch it to
sandboxd over NetBird. VM1 never executes PR or upstream code; it only fetches and ships source."""
import base64, json, os, sys, time
from plumbline.fetch import fetch_closure

CORPUS = os.path.join(os.path.dirname(__file__), "..", "corpus")
ROOTS = {"ds": ("deepspeed", ""), "ray": ("ray", "python/"), "st": ("sentence_transformers", ""),
         "unsloth": ("unsloth", ""), "zoo": ("unsloth_zoo", "")}


def corpus_case(cid):
    """cid like ds-8313; corpus files use deepspeed-8313."""
    fname = cid.replace("ds-", "deepspeed-", 1) + ".yaml"
    c = dict(l.split(": ", 1) for l in open(os.path.join(CORPUS, fname)).read().splitlines() if ": " in l)
    c["root_pkg"], c["prefix"] = ROOTS[cid.split("-")[0]]
    c["repo"] = c["repo"].replace("https://github.com/", "")
    return c


def make_job(job_id, case, sha, checks, scope=None, per_check_timeout_s=60):
    path = (scope or case["scope"])[len(case["prefix"]):]
    files = fetch_closure(case["repo"], sha, path, case["root_pkg"], src_prefix=case["prefix"])
    if path not in files:
        raise FileNotFoundError(f"{path} not found at {sha[:12]}")
    return {"job_id": job_id, "mode": "extract", "checks": checks, "files": files,
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


if __name__ == "__main__":
    # python -m plumbline.sentinel gold <case-id> <sha> [<sha> ...]
    cid, shas = sys.argv[2], sys.argv[3:]
    for sha in shas:
        res = gold_run(cid, sha)
        checks = [l for l in res["lines"] if l.get("ev") == "check"]
        print(json.dumps({"case": cid, "sha": sha[:12], "exit": res["exit_code"], "wall_ms": res["wall_ms"],
                          "statuses": [c["status"] for c in checks],
                          "details": [c["detail"][:90] for c in checks], "runtime": res["runtime"]}))
