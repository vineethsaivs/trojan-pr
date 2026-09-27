"""Oracle replay for ray-project/ray #65790 (PBT quantile_fraction=0 selects everyone).

Fetches python/ray/tune/schedulers/pbt.py at each SHA (read-only gh api GET), extracts
PopulationBasedTraining._quantiles with ast, runs it on stub trials, and checks the spec:

  Docstring: "Parameters are transferred from the top quantile_fraction fraction of trials to
  the bottom quantile_fraction fraction ... Setting it to 0 essentially implies doing no
  exploitation at all."

Invariants (restated from the math, not from either implementation), for n live scored trials
and fraction q in [0, 0.5]:
  I1 q == 0            -> lower == upper == [] (documented "no exploitation")
  I2 |lower| == |upper| (bottom and top quantiles are mirror images)
  I3 lower and upper are disjoint
  I4 lower = the |lower| lowest-scored trials, upper = the |upper| highest-scored trials
  I5 floor(n*q) <= |upper| <= ceil(n*q), except capped at n//2 so I3 can hold
  I6 metamorphic: |upper(q)| is non-decreasing in q (asking for less never returns more)
Tolerance: exact. Outputs are integer counts and set membership; scores are distinct floats
0.0..n-1 compared only for ordering, so no rounding enters.

Usage: python oracle.py   (needs `gh` authenticated; caches sources in ./src)
"""
import ast, logging, math, os, random, subprocess, sys
from typing import List, Tuple

REPO, PATH = "ray-project/ray", "python/ray/tune/schedulers/pbt.py"
SHAS = [
    ("pre-intro b674c4a (parent of #4912)", "b674c4a5ba6cc62d4535e2e016b248edc28b53e5"),
    ("intro     c2253d2 (#4912 merge)", "c2253d2313f5a43c20658319063e1713ad695e67"),
    ("fix-parent 0dc4886 (ray master before fix)", "0dc4886da85a47ea236b45e7976e3ed930268ffa"),
    ("fix       8c9e87c (#65790 merge)", "8c9e87cd2d77af4eb2a6fdda8fc82a9d948bffad"),
]
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(sha):
    p = os.path.join(HERE, "src", f"pbt_{sha[:7]}.py")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        raw = subprocess.check_output(
            ["gh", "api", f"repos/{REPO}/contents/{PATH}?ref={sha}",
             "-H", "Accept: application/vnd.github.raw"])
        open(p, "wb").write(raw)
    return open(p).read()


class Trial:  # stub: only what _quantiles touches
    def __init__(s, name): s.name = name
    def is_finished(s): return False
    def __repr__(s): return s.name


class State:
    def __init__(s, score): s.last_score = score


def extract(src):
    """Return (_quantiles function, fixed q or None). Pre-#4912 code reads a module constant."""
    tree = ast.parse(src)
    fixed_q = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PBT_QUANTILE" for t in node.targets):
            fixed_q = ast.literal_eval(node.value)
    for c in tree.body:
        if isinstance(c, ast.ClassDef) and c.name == "PopulationBasedTraining":
            for f in c.body:
                if isinstance(f, ast.FunctionDef) and f.name == "_quantiles":
                    f.decorator_list = []
                    ns = {"math": math, "logger": logging.getLogger("oracle"), "Tuple": Tuple,
                          "List": List, "Trial": Trial, "PBT_QUANTILE": fixed_q}
                    exec(compile(ast.Module([f], []), "pbt.py", "exec"), ns)
                    uses_attr = "_quantile_fraction" in ast.unparse(f)
                    return ns["_quantiles"], (None if uses_attr else fixed_q)
    raise LookupError("_quantiles not found")


def call(fn, n, q, seed=0):
    self = type("PBT", (), {})()
    self._quantile_fraction = q
    trials = [Trial(f"t{i}") for i in range(n)]  # t_i has score i, so rank == index
    order = trials[:]
    random.Random(seed).shuffle(order)  # insertion order must not matter
    self._trial_state = {t: State(float(int(t.name[1:]))) for t in order}
    lo, hi = fn(self)
    return trials, list(lo), list(hi)


def check(fn, qs, ns=(2, 3, 4, 5, 8, 16)):
    fails, demo = [], {}
    for n in ns:
        uppers = []
        for q in qs:
            ranked, lo, hi = call(fn, n, q)
            a, b = len(lo), len(hi)
            cap = n // 2
            bad = []
            if q == 0 and (lo or hi): bad.append("I1 q=0 not empty")
            if a != b: bad.append("I2 |lower|!=|upper|")
            if set(lo) & set(hi): bad.append("I3 overlap")
            if lo != ranked[:a] or sorted(hi, key=ranked.index) != ranked[n - b:]: bad.append("I4 wrong members")
            if not (min(math.floor(n * q), cap) <= b <= min(math.ceil(n * q), cap)): bad.append("I5 size")
            if bad: fails.append(f"n={n} q={q}: lower={a} upper={b} of {n} [{', '.join(bad)}]")
            uppers.append(b)
            if n == 4: demo[q] = (a, b)
        if any(x > y for x, y in zip(uppers, uppers[1:])):
            fails.append(f"n={n}: I6 |upper| not monotone in q {list(zip(qs, uppers))}")
    return fails, demo


if __name__ == "__main__":
    verdicts = []
    for label, sha in SHAS:
        fn, fixed_q = extract(fetch(sha))
        qs = [fixed_q] if fixed_q is not None else [0.0, 0.1, 0.25, 0.4, 0.5]
        fails, demo = check(fn, qs)
        v = "PASS" if not fails else "FAIL"
        verdicts.append((label, v))
        note = f" (q hardcoded to PBT_QUANTILE={fixed_q}; q=0 not expressible here)" if fixed_q is not None else ""
        print(f"{v}  {label}{note}")
        print("      n=4 (lower, upper) by q: " + ", ".join(f"q={q}: {lu}" for q, lu in demo.items()))
        for f in fails[:6]: print("      " + f)
        if len(fails) > 6: print(f"      ... {len(fails) - 6} more")
    ok = [v for _, v in verdicts] == ["PASS", "FAIL", "FAIL", "PASS"]
    print("\nexpected pattern PASS/FAIL/FAIL/PASS:", "MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
