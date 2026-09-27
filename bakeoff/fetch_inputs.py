"""Laptop only (read-only gh; VMs have no GitHub token). For each H case: resolve the four commits
(pre, intro, fixp, fix) to full SHAs, find which candidate scope path exists at each, and fetch the
intro and fix PR inputs (PLAN 4.4): scope-file hunks only (g2.diff), title+body, full scope file.
Writes bakeoff/inputs/<id>/ (gitignored). Case facts come from research/cases_facts.json."""
import json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = ["ds-8313", "ds-8533", "st-3921", "ds-8334", "ray-65747", "st-4019",
     "st-3868", "ray-65535", "ray-65790", "zoo-1286", "unsloth-11337", "unsloth-11470"]


def gh(*args, ok_fail=False):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode and not ok_fail:
        raise RuntimeError(f"gh {' '.join(args)[:120]}: {r.stderr.strip()[:200]}")
    return r.stdout if r.returncode == 0 else None


def full_sha(repo, s):
    return gh("api", f"repos/{repo}/commits/{s}", "--jq", ".sha").strip()


def exists(repo, path, sha):
    return gh("api", f"repos/{repo}/contents/{path}?ref={sha}", "--jq", ".sha", ok_fail=True) is not None


def raw(repo, path, sha):
    return gh("api", f"repos/{repo}/contents/{path}?ref={sha}", "-H", "Accept: application/vnd.github.raw")


def scope_hunks(diff, path):
    """Keep only the file sections of `path` (either side of a rename)."""
    parts = re.split(r"(?m)^(?=diff --git )", diff)
    return "".join(p for p in parts if f" a/{path}" in p.split("\n", 1)[0] or f" b/{path}" in p.split("\n", 1)[0])


def fetch_case(c):
    repo = c["repo"].split()[0]
    paths = re.findall(r"[\w/.-]+\.py", c["file"])
    corpus = os.path.join(ROOT, "corpus", c["id"].replace("ds-", "deepspeed-", 1) + ".yaml")
    if os.path.exists(corpus):   # the corpus scope is the file's later path when it moved
        paths += re.findall(r"^scope: (\S+)", open(corpus).read(), re.M)
    norm = lambda q: os.path.basename(q)[:-3].replace("_", "").lower()
    keys = {norm(q) for q in paths}
    nfix = re.search(r"/pull/(\d+)", c["fix"]["pr"]).group(1)
    fix_files = json.loads(gh("pr", "view", nfix, "-R", c["repo"].split()[0], "--json", "files"))["files"]
    paths += [f["path"] for f in fix_files if f["path"].endswith(".py")
              and any(norm(f["path"]).startswith(k) or k.startswith(norm(f["path"])) for k in keys)]
    paths = list(dict.fromkeys(paths))
    commits = {"pre": c["pre"], "intro": c["intro"]["sha"], "fixp": c["fixp"], "fix": c["fix"]["sha"]}
    out = {"id": c["id"], "repo": repo, "function": c.get("function"), "commits": {}}
    for label, s in commits.items():
        sha = full_sha(repo, re.match(r"[0-9a-f]{7,40}", str(s)).group(0))
        path = next((p for p in paths if exists(repo, p, sha)), None)
        out["commits"][label] = {"sha": sha, "path": path}
    d = os.path.join(ROOT, "bakeoff", "inputs", c["id"])
    for kind, label in (("intro", "intro"), ("fix", "fix")):
        n = re.search(r"/pull/(\d+)", c[kind]["pr"]).group(1)
        path = out["commits"][label]["path"]
        os.makedirs(os.path.join(d, kind), exist_ok=True)
        full = gh("pr", "diff", n, "-R", repo, ok_fail=True)
        if full is not None:
            g2 = scope_hunks(full, path) if path else ""
        else:  # diff over GitHub's 20000-line limit: rebuild the scope file's section from the files API
            files = json.loads(gh("api", f"repos/{repo}/pulls/{n}/files", "--paginate", "--slurp"))
            f = next((f for page in files for f in page if f["filename"] == path), None)
            g2 = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{f['patch']}\n" if f and f.get("patch") else ""
        meta = json.loads(gh("pr", "view", n, "-R", repo, "--json", "title,body,number,url"))
        open(os.path.join(d, kind, "g2.diff"), "w").write(g2)
        json.dump(meta, open(os.path.join(d, kind, "meta.json"), "w"), indent=1)
        open(os.path.join(d, kind, "scope.py"), "w").write(raw(repo, path, out["commits"][label]["sha"]) if path else "")
        out[kind] = {"pr": int(n), "g2_lines": g2.count("\n"), "path": path}
    json.dump(out, open(os.path.join(d, "commits.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    facts = {c["id"]: c for c in json.load(open(os.path.join(ROOT, "research/cases_facts.json")))["cases"]}
    for cid in sys.argv[1:] or H:
        try:
            o = fetch_case(facts[cid])
            print(cid, {k: (v["sha"][:8], (v["path"] or "ABSENT").split("/")[-1]) for k, v in o["commits"].items()},
                  "intro g2", o["intro"]["g2_lines"], "fix g2", o["fix"]["g2_lines"])
        except Exception as e:
            print(cid, "ERROR", type(e).__name__, str(e)[:160])
