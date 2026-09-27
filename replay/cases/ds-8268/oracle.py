"""ds-8268 oracle: DeepSpeed CLI helper must carry WarmupCosineLR's ratios.

Fetches deepspeed/runtime/lr_schedules.py at each SHA (gh api, read-only), loads it
with real CPU torch, and stubs only deepspeed.utils.logger (the file's one other import).

Oracle restates the contract, not either implementation:
  1. Conservation: flags the user typed for schedule S reach config['params'] unchanged,
     and no param that S does not accept appears (checked by constructing S from them).
  2. Closed form of "warmup from L*wmin to L, then cosine down to L*cmin":
       lr(first step) = L * wmin          (warmup term is 0 at step 0)
       lr(last step)  = L * cmin          (cos(pi) = -1)
       every lr in [L*min(wmin,cmin), L]; non-decreasing in warmup, non-increasing after.
  3. get_lr_from_config must not report an absolute lr for a ratio-only schedule,
     and must not crash on a hand-written WarmupCosineLR config.

Tolerance: dict values are exact (same decimal string -> same float). lr checks use
rel_tol 1e-9: the closed form is a handful of IEEE double ops (cos(pi) == -1.0 exactly,
log(1) == 0), so honest error is a few ulp (~1e-16 relative). Real failures are a
missing key or an exception, orders of magnitude outside any tolerance.
"""
import argparse, base64, importlib.util, json, logging, math, os, subprocess, sys, types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO, PATH = "deepspeedai/DeepSpeed", "deepspeed/runtime/lr_schedules.py"
SHAS = [
    ("pre-intro (parent of #4563)", "da652d0e0b4ff0b7c870e74730b9d3d8f51bf8aa"),
    ("intro #4563", "4388a605f854db91302c4f89053ee861eb31bacd"),
    ("fix parent", "3bdabae087fedb510253535675f98a8663cce454"),
    ("fix #8268", "cdd206ff416e9ec5dbcf3845068efb13c9224ac2"),
]
L, T, W, WMIN, CMIN = 1e-3, 2000, 1000, 0.1, 0.05
ARGV = ["--lr_schedule", "WarmupCosineLR", "--warmup_min_ratio", str(WMIN), "--cos_min_ratio", str(CMIN),
        "--warmup_num_steps", str(W), "--warmup_type", "linear"]
TOL = 1e-9

ds = types.ModuleType("deepspeed"); dsu = types.ModuleType("deepspeed.utils")
dsu.logger = logging.getLogger("ds"); dsu.logger.setLevel(logging.ERROR); ds.utils = dsu
sys.modules.update({"deepspeed": ds, "deepspeed.utils": dsu})


def fetch(sha):
    path = os.path.join(HERE, f"src_{sha[:10]}.py")
    if not os.path.exists(path):
        out = subprocess.run(["gh", "api", f"repos/{REPO}/contents/{PATH}?ref={sha}"], check=True,
                             capture_output=True, text=True).stdout
        open(path, "wb").write(base64.b64decode(json.loads(out)["content"]))
    spec = importlib.util.spec_from_file_location(f"lrs_{sha[:10]}", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def adam():
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=L)


def oracle(m):
    if "WarmupCosineLR" not in getattr(m, "VALID_LR_SCHEDULES", []):
        return "N/A", "WarmupCosineLR not in VALID_LR_SCHEDULES yet"
    fails = []
    cfg, err = m.get_config_from_args(m.add_tuning_arguments(argparse.ArgumentParser()).parse_args(ARGV))
    p = cfg["params"]
    got = (p.get("warmup_min_ratio"), p.get("cos_min_ratio"))
    if got != (WMIN, CMIN):
        fails.append(f"[1] ratios typed ({WMIN}, {CMIN}) came back {got}; params={p}")
    try:
        lr_rep = m.get_lr_from_config(cfg)[0]
        if lr_rep is not None:
            fails.append(f"[3] get_lr_from_config(CLI cfg) reports lr={lr_rep}")
    except Exception as e:
        fails.append(f"[3] get_lr_from_config(CLI cfg) raises {type(e).__name__}: {e}")
    try:
        hand = {"type": "WarmupCosineLR", "params": {"warmup_min_ratio": WMIN, "cos_min_ratio": CMIN}}
        lr_rep = m.get_lr_from_config(hand)[0]
        if lr_rep is not None:
            fails.append(f"[3] get_lr_from_config(hand cfg) reports lr={lr_rep}")
    except Exception as e:
        fails.append(f"[3] get_lr_from_config(hand cfg) raises {type(e).__name__}: {e!s}")
    try:
        opt = adam(); s = m.WarmupCosineLR(opt, total_num_steps=T, **p)
    except TypeError as e:
        fails.append(f"[1] WarmupCosineLR(**params) raises TypeError: {e}")
        return "FAIL", " | ".join(fails)
    lrs = []
    for _ in range(T):
        s.step(); lrs.append(opt.param_groups[0]["lr"])
    lo, warm, decay = L * min(WMIN, CMIN), lrs[:W], lrs[W:]
    if not math.isclose(lrs[0], L * WMIN, rel_tol=TOL):
        fails.append(f"[2] lr(0)={lrs[0]:.6g} want {L * WMIN:.6g}")
    if not math.isclose(lrs[-1], L * CMIN, rel_tol=TOL):
        fails.append(f"[2] lr(T-1)={lrs[-1]:.6g} want {L * CMIN:.6g}")
    if not all(lo * (1 - TOL) <= x <= L * (1 + TOL) for x in lrs):
        fails.append(f"[2] lr left [{lo:.3g}, {L:.3g}]: min={min(lrs):.3g} max={max(lrs):.3g}")
    if any(b < a * (1 - TOL) for a, b in zip(warm, warm[1:])) or any(b > a * (1 + TOL) for a, b in zip(decay, decay[1:])):
        fails.append("[2] trajectory not rise-then-fall")
    nums = f"params={p} lr(0)={lrs[0]:.6g} peak={max(lrs):.6g} lr(T-1)={lrs[-1]:.6g}"
    return ("FAIL", " | ".join(fails)) if fails else ("PASS", nums)


def control(m):
    # Unaffected schedule: WarmupLR's flag must round-trip at every SHA.
    a = m.add_tuning_arguments(argparse.ArgumentParser()).parse_args(["--lr_schedule", "WarmupLR", "--warmup_max_lr", "0.003"])
    cfg, _ = m.get_config_from_args(a)
    ok = cfg["params"].get("warmup_max_lr") == 0.003 and m.get_lr_from_config(cfg)[0] == 0.003
    return "PASS" if ok else f"FAIL {cfg}"


if __name__ == "__main__":
    print(f"torch {torch.__version__}  L={L} T={T} W={W} wmin={WMIN} cmin={CMIN}")
    verdicts = {}
    for label, sha in SHAS:
        m = fetch(sha)
        v, detail = oracle(m)
        verdicts[label] = v
        print(f"{sha[:10]} {label:28s} oracle: {v}  {detail}\n{'':39s} control(WarmupLR): {control(m)}")
    expect = {"pre-intro (parent of #4563)": "N/A", "intro #4563": "FAIL", "fix parent": "FAIL", "fix #8268": "PASS"}
    print("EXPECTED PATTERN HOLDS" if verdicts == expect else f"UNEXPECTED: {verdicts}")
