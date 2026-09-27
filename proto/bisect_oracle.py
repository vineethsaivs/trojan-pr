"""Find the commit that introduced a bug by running the ORACLE (not blame) over the
history of the file. Read-only GETs. Assumes the check passes (or cannot run) before
the introduction and fails from it until the fix."""
import json, subprocess, sys, time
import run_cases as R


def file_history(repo, path, head):
    out = subprocess.run(["gh", "api", "--paginate", f"repos/{repo}/commits?path={path}&sha={head}&per_page=100",
                          "--jq", ".[] | [.sha, .commit.committer.date, (.commit.message|split(\"\\n\")[0])] | @tsv"],
                         capture_output=True, text=True, check=True).stdout
    return [l.split("\t") for l in out.strip().splitlines()]          # newest first


def signature(vs):
    """Failure signature: which checks failed and how. Bisect only follows the SAME one."""
    return tuple(sorted((v.family, v.detail.split(":")[0].split(" at ")[0][:40]) for v in vs if v.status == "FAIL"))


def status(cid, case, sha, want=None):
    c = dict(case)
    try:
        mod, _, _ = R.load_target(c, sha)
        vs = R.run_plan(cid, mod)
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {e}"[:120]
    if any(v.status == "FAIL" for v in vs):
        sig = signature(vs)
        if want is not None and sig != want:
            return "ERROR", f"different failure {sig}"[:120]
        return "FAIL", next(v.detail for v in vs if v.status == "FAIL")[:120]
    if all(v.status == "PASS" for v in vs):
        return "PASS", ""
    return "ERROR", next(v.detail for v in vs if v.status == "ERROR")[:120]


def bisect(cid):
    case = R.load_yaml(f"{R.CORPUS}/{cid}.yaml")
    case["_root"], case["_prefix"] = R.ROOTS[cid.split("-")[0]]
    repo = case["repo"].replace("https://github.com/", "")
    hist = file_history(repo, case["scope"], case["buggy_sha"])
    lo, hi, runs = 0, len(hist) - 1, []                                   # hist[lo] fails (buggy)
    mod, _, _ = R.load_target(dict(case), case["buggy_sha"])
    want = signature(R.run_plan(cid, mod))                                 # the failure we are dating
    s_old, _ = status(cid, case, hist[hi][0], want); runs.append((hist[hi][0][:10], s_old))
    if s_old == "FAIL":
        return {"case": cid, "introduced": "at or before oldest file commit", "oldest": hist[hi], "runs": runs}
    while hi - lo > 1:                                                     # invariant: lo FAIL, hi not FAIL
        mid = (lo + hi) // 2
        s, why = status(cid, case, hist[mid][0], want); runs.append((hist[mid][0][:10], s))
        lo, hi = (mid, hi) if s == "FAIL" else (lo, mid)
    s_prev, why_prev = status(cid, case, hist[hi][0], want)
    kind = "regression" if s_prev == "PASS" else "at_or_before"   # ERROR below: symbol absent or API differs
    return {"case": cid, "file_commits": len(hist), "introducing": hist[lo], "previous": hist[hi],
            "previous_status": s_prev, "previous_why": why_prev, "kind": kind,
            "oracle_runs": len(runs) + 1, "runs": runs}


if __name__ == "__main__":
    for cid in sys.argv[1:]:
        t = time.time()
        r = bisect(cid)
        r["wall_s"] = round(time.time() - t, 1)
        print(json.dumps(r))
