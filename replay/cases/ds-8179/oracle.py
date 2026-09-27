"""ds-8179 oracle: OneCycle stair counts must produce a staircase (deepspeedai/DeepSpeed #8179).

Math restated (not copied from either implementation):
  A 1Cycle half spans N batches and moves lr from lo to hi (momentum from hi to lo).
  With stair count k > 0 the value only takes the k+1 levels
      lo + (hi - lo) * j / k,  j = 0..k
  and moves monotonically through them. With 0 < k < N that is a different schedule
  from k = 0 (continuous, one new value per batch).
Checks per half (rise, fall) for lr and momentum:
  1. every value is on the k-level grid   2. monotone in the right direction
  3. schedule differs from the k = 0 run (metamorphic)
Controls: pre-introduction commit (function absent, reported N/A) and the k = 0 schedule,
which must be byte-identical between the fix parent and the fix (fix must not touch it).

Sources are fetched read-only with `gh api .../contents/PATH?ref=SHA` and exec'd with stubs
for torch.optim.Optimizer and deepspeed.utils.logger. Stdlib only.
Run: python3 oracle.py
"""
import base64, json, logging, os, subprocess, sys, types

REPO = "deepspeedai/DeepSpeed"
OLD, NEW = "deepspeed/pt/deepspeed_lr_schedules.py", "deepspeed/runtime/lr_schedules.py"
VERSIONS = [  # (label, sha, path)
    ("pre-intro dc226fdb", "dc226fdb7aae074ce16aa68d088dfde1be654892", OLD),
    ("intro     7a9fbe67", "7a9fbe67adc431a8d74d930585ee6264660fcb8d", OLD),
    ("fixparent 6c58fb75", "6c58fb75fd9ddfa8d7b63fa35848d4f69beb8789", NEW),
    ("fix       3169c2dc", "3169c2dcf89b16b96d0ec76b57a22f0f1fbdb5ea", NEW),
]
LO_LR, HI_LR, LO_MOM, HI_MOM = 0.001, 0.01, 0.8, 0.9
N = 8
# Tolerance from arithmetic: a grid level is a + (b - a) * j / k in float64, a few ops on
# values <= 0.9, each off by <= 2^-53 relative, so |error| < 1e-15. 1e-12 is 1000x that and
# ~1e9x smaller than the finest grid spacing here ((0.01 - 0.001) / 4 = 2.25e-3), so float
# noise can neither fake nor hide a stair. The failure itself is structural (8 levels vs <= k+1).
TOL = 1e-12


class Optimizer:  # stub for torch.optim.Optimizer: param_groups and defaults with betas
    def __init__(self):
        self.param_groups = [{"params": [], "lr": LO_LR, "betas": (HI_MOM, 0.99)}]
        self.defaults = {"lr": LO_LR, "betas": (HI_MOM, 0.99)}


def install_stubs():
    torch = types.ModuleType("torch")
    torch.tensor, torch.is_tensor = (lambda *a, **k: a[0]), (lambda x: False)
    optim = types.ModuleType("torch.optim")
    optim.Optimizer = Optimizer
    torch.optim = optim
    utils = types.ModuleType("deepspeed.utils")
    utils.logger = logging.getLogger("stub")
    sys.modules.update({"torch": torch, "torch.optim": optim, "deepspeed": types.ModuleType("deepspeed"),
                        "deepspeed.utils": utils, "deepspeed.pt": types.ModuleType("deepspeed.pt"),
                        # the 2020 file star-imports this; OneCycle uses nothing from it
                        "deepspeed.pt.deepspeed_constants": types.ModuleType("deepspeed.pt.deepspeed_constants")})


def fetch(sha, path):
    """Return source text at sha, or None if the file does not exist there (HTTP 404)."""
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", f"{sha[:8]}.py")
    if os.path.exists(cache):
        return open(cache).read()
    r = subprocess.run(["gh", "api", f"repos/{REPO}/contents/{path}?ref={sha}"], capture_output=True, text=True)
    if r.returncode != 0:
        if "404" in r.stdout + r.stderr:
            return None
        raise RuntimeError(r.stderr)
    src = base64.b64decode(json.loads(r.stdout)["content"]).decode()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    open(cache, "w").write(src)
    return src


def run(OneCycle, k):
    """Full cycle (2N step() calls). Returns lists of lr and beta1 after each step."""
    opt = Optimizer()
    s = OneCycle(opt, cycle_min_lr=LO_LR, cycle_max_lr=HI_LR, cycle_min_mom=LO_MOM, cycle_max_mom=HI_MOM,
                 cycle_first_step_size=N, cycle_second_step_size=N,
                 cycle_first_stair_count=k, cycle_second_stair_count=k)
    lr, mom = [], []
    for _ in range(2 * N):
        s.step()
        lr.append(opt.param_groups[0]["lr"])
        mom.append(opt.param_groups[0]["betas"][0])
    return lr, mom


def on_grid(vals, lo, hi, k):
    return all(abs((v - lo) / (hi - lo) * k - round((v - lo) / (hi - lo) * k)) * (hi - lo) / k < TOL
               and -TOL <= v - lo <= hi - lo + TOL for v in vals)


def monotone(vals, up):
    return all((b - a) * (1 if up else -1) >= -TOL for a, b in zip(vals, vals[1:]))


def check(OneCycle):
    ok = True
    base = run(OneCycle, 0)
    for k in (2, 4):
        got = run(OneCycle, k)
        for name, vals, ref, lo, hi, rise_up in (("lr ", got[0], base[0], LO_LR, HI_LR, True),
                                                 ("mom", got[1], base[1], LO_MOM, HI_MOM, False)):
            # ponytail: halves split at N; a version that indexes one step earlier only moves
            # the peak across the split, which leaves all three checks valid.
            for half, seg, up in (("rise", vals[:N], rise_up), ("fall", vals[N:], not rise_up)):
                levels = len({round(v, 12) for v in seg})
                g, m, c = on_grid(seg, lo, hi, k), monotone(seg, up), seg != (ref[:N] if half == "rise" else ref[N:])
                passed = g and m and c
                ok &= passed
                print(f"    k={k} {name} {half}: levels={levels} (max {k + 1}) on_grid={g} monotone={m} "
                      f"changed_vs_k0={c} -> {'PASS' if passed else 'FAIL'}")
        print(f"    k={k} lr rise: {[round(v, 6) for v in got[0][:N]]}")
    print(f"    k=0 lr rise: {[round(v, 6) for v in base[0][:N]]}")
    return ok, base


if __name__ == "__main__":
    install_stubs()
    verdict, k0 = {}, {}
    for label, sha, path in VERSIONS:
        print(f"{label}  {path}")
        src = fetch(sha, path)
        if src is None:
            verdict[label] = "N/A (file absent: OneCycle did not exist yet)"
            print("    file not present at this SHA")
            continue
        ns = {"__name__": "lr_schedules"}
        exec(compile(src, f"{sha[:8]}:{path}", "exec"), ns)
        ok, k0[label] = check(ns["OneCycle"])
        verdict[label] = "PASS" if ok else "FAIL"
    same = k0["fixparent 6c58fb75"] == k0["fix       3169c2dc"]
    print(f"benign control: k=0 schedule identical in fix parent and fix: {same}")
    print("VERDICT:", json.dumps(verdict, indent=1))
    expected = verdict["intro     7a9fbe67"] == "FAIL" and verdict["fixparent 6c58fb75"] == "FAIL" \
        and verdict["fix       3169c2dc"] == "PASS" and same
    print("ORACLE BEHAVES AS EXPECTED (fail on bug, pass on fix, k=0 unchanged):", expected)
    sys.exit(0 if expected else 1)
