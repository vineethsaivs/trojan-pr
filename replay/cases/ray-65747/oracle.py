"""Oracle for ray-65747: Ray Tune PBT integer perturbation.

Math (restated, not copied from any version of pbt.py):
  PBT perturbs a hyperparameter x by a multiplicative factor f in {4/5, 6/5}.
  For an integer hyperparameter the result y must be an int and the nearest
  integer to f*x:           |y - f*x| <= 1/2
  and move with the factor: f > 1 -> y >= x,  f < 1 -> y <= x.
  Metamorphic check: a symmetric random walk of 50 perturbations from x=32 must
  never be absorbed at 0 (nearest-integer of 0.8*x is >= 1 for every x >= 1).

Tolerance: the reference f*x is computed with exact Fractions (6/5, 4/5), so
there is no float error at all. 1/2 is the definition of nearest-integer
rounding. Ties cannot happen: 6x/5 = k + 1/2 needs 12x = 10k + 5 (even = odd),
same for 4x/5 (8x = 10k + 5), so the nearest integer is unique.

Code under test is function-extracted from pbt.py at each SHA (files fetched
with `gh api repos/ray-project/ray/contents/<path>?ref=<sha>`), exec'd with
stdlib-only stubs. The factor choice inside the code is forced by a stub
`random` module so every (x, f) pair is tested deterministically.
"""
import ast, contextlib, copy, io, math, random, sys, typing
from fractions import Fraction

HERE = __file__.rsplit("/", 1)[0]
FACTORS = {1.2: Fraction(6, 5), 0.8: Fraction(4, 5)}


class ForcedRandom:
    """Stands in for the `random` module: never resample, always pick factor f."""
    f = None

    def random(self):  # resample check `< 0.0` is False; 2018 branch `> 0.5` picks 1.2
        return 0.9 if self.f > 1 else 0.1

    def choice(self, seq):  # prototype: random.choice([0.8, 1.2]); modern: perturbation_factors
        return self.f


def load(tag, name):
    src = open(f"{HERE}/pbt_{tag}.py").read()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)
    rnd = ForcedRandom()

    class Domain:  # stub, never instantiated here
        pass

    ns = {"copy": copy, "math": math, "random": rnd, "Domain": Domain, "Callable": typing.Callable,
          "Dict": dict, "Optional": typing.Optional, "Tuple": tuple, "List": list}
    exec(compile(ast.get_source_segment(src, fn), f"pbt_{tag}.{name}", "exec"), ns)
    return ns[name], rnd


def make_perturb(tag):
    if tag == "a936468f":  # prototype method _explore(self, hyperparams, mutations, best_trial)
        fn, rnd = load(tag, "_explore")
        best = type("T", (), {"config": {"env": "e"}})()
        call = lambda x: fn(None, {"v": x, "env": "e"}, None, best)["v"]
    elif tag == "b9484055":  # 2018: explore(config, mutations, resample_probability, custom_explore_fn)
        fn, rnd = load(tag, "explore")
        call = lambda x: fn({"v": x}, {"v": lambda c: x}, 0.0, None)["v"]
    else:  # modern: _explore(config, mutations, p, perturbation_factors, custom_explore_fn)
        fn, rnd = load(tag, "_explore")
        call = lambda x: fn({"v": x}, {"v": lambda: x}, 0.0, perturbation_factors=(rnd.f, rnd.f),
                            custom_explore_fn=None)[0]["v"]

    def perturb(x, f):
        rnd.f = f
        with contextlib.redirect_stdout(io.StringIO()):  # 2018/prototype code prints
            return call(x)
    return perturb


def violates(x, f, y):
    fx = FACTORS[f] * x
    return not (type(y) is int and abs(y - fx) <= Fraction(1, 2) and (y >= x if f > 1 else y <= x))


def walk_zero_frac(perturb, start=32, steps=50, seeds=500):
    zeros = 0
    for s in range(seeds):
        rng, v = random.Random(s), start
        for _ in range(steps):
            v = perturb(v, rng.choice((1.2, 0.8)))
        zeros += v == 0
    return zeros / seeds


SHAS = [  # (tag, role, expected verdict)
    ("a936468f", "pre-intro prototype (not a clean control, ceil-to-even rule)", None),
    ("b9484055", "introducing #1478 b948405 (2018-02-03)", "FAIL"),
    ("8c9e87cd", "fix parent 8c9e87c", "FAIL"),
    ("e9456df3", "fix #65747 e9456df", "PASS"),
]

if __name__ == "__main__":
    ok = True
    for tag, role, want in SHAS:
        p = make_perturb(tag)
        viol = [(x, f, p(x, f)) for x in range(1, 51) for f in (1.2, 0.8) if violates(x, f, p(x, f))]
        z = walk_zero_frac(p)
        verdict = "FAIL" if viol or z > 0 else "PASS"
        ctrl = [(x, f, p(x, f)) for x in (10, 100) for f in (1.2, 0.8)]
        ctrl_ok = not any(violates(*c) for c in ctrl)
        print(f"{verdict}  {role}\n      violations={len(viol)}/100  walk P(final==0)={z:.0%}"
              f"  x=4*1.2->{p(4, 1.2)} (want 5)  x=1*0.8->{p(1, 0.8)} (want 1)"
              f"  controls x=10,100 {'pass' if ctrl_ok else 'FAIL'} {ctrl}\n      first={viol[:5]}")
        if want and verdict != want:
            ok = False
    print("ORACLE BEHAVES AS EXPECTED" if ok else "UNEXPECTED VERDICT")
    sys.exit(0 if ok else 1)
