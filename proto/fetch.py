"""Control-plane side of function-extract: fetch a file at a SHA plus the sibling
modules the static pre-pass says must be real. Read-only GitHub GETs, disk cache."""
import json, os, subprocess, sys
from extract import needed_siblings

CACHE = os.path.join(os.path.dirname(__file__), "cache")


def gh_raw(repo, path, sha):
    key = os.path.join(CACHE, repo.replace("/", "__"), sha, path)
    if os.path.exists(key):
        return open(key).read() or None
    r = subprocess.run(["gh", "api", f"repos/{repo}/contents/{path}?ref={sha}",
                        "-H", "Accept: application/vnd.github.raw"], capture_output=True, text=True)
    txt = r.stdout if r.returncode == 0 else ""
    os.makedirs(os.path.dirname(key), exist_ok=True)
    open(key, "w").write(txt)
    return txt or None


def fetch_closure(repo, sha, path, root_pkg, depth=2, src_prefix=""):
    """Return {path: source} for `path` and the real-needed siblings, depth-limited."""
    files, frontier = {}, [(path, 0)]
    while frontier:
        p, d = frontier.pop()
        if p in files:
            continue
        src = gh_raw(repo, src_prefix + p, sha)
        if src is None:
            continue
        files[p] = src
        if d >= depth:
            continue
        mod = p[:-3].replace("/", ".").removesuffix(".__init__")
        for m in needed_siblings(src, mod, p.endswith("__init__.py")):
            if m.split(".")[0] != root_pkg:
                continue
            mp = m.replace(".", "/")
            frontier += [(mp + ".py", d + 1), (mp + "/__init__.py", d + 1)]
    return files


if __name__ == "__main__":
    repo, sha, path, root = sys.argv[1:5]
    print(json.dumps(sorted(fetch_closure(repo, sha, path, root))))
