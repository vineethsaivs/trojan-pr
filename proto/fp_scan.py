"""False-positive scan: treat every commit that touched the file as a PR, judge it
base (previous file commit) vs head with the differential rule, count detections.
Expected: exactly one detection, at the introducing commit, and none elsewhere."""
import json, sys, time
import bisect_oracle as B, run_cases as R


def decide(b, h):
    if "ERROR" in (b, h):
        return "inconclusive"
    return {("PASS", "FAIL"): "DETECTED", ("PASS", "PASS"): "clean",
            ("FAIL", "FAIL"): "preexisting", ("FAIL", "PASS"): "improved"}[(b, h)]


def scan(cid):
    case = R.load_yaml(f"{R.CORPUS}/{cid}.yaml")
    case["_root"], case["_prefix"] = R.ROOTS[cid.split("-")[0]]
    repo = case["repo"].replace("https://github.com/", "")
    hist = B.file_history(repo, case["scope"], case["fixed_sha"])        # include the fix commit
    st = {sha: B.status(cid, case, sha)[0] for sha, _, _ in hist}
    rows = []
    for (sha, date, msg), (psha, _, _) in zip(hist, hist[1:]):
        rows.append({"sha": sha[:10], "date": date[:10], "msg": msg[:60], "decision": decide(st[psha], st[sha])})
    counts = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    return {"case": cid, "commits_judged": len(rows), "counts": counts,
            "detected": [r for r in rows if r["decision"] == "DETECTED"],
            "improved": [r for r in rows if r["decision"] == "improved"]}


if __name__ == "__main__":
    for cid in sys.argv[1:]:
        t = time.time()
        r = scan(cid)
        r["wall_s"] = round(time.time() - t, 1)
        print(json.dumps(r))
