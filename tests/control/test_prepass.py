"""PLAN 8: proto/intro/ds-8313.diff -> clip_grad_norm_, power_root_order, decomposition (laptop; needs the
gitignored upstream snapshots in proto/intro/). Run: python3 tests/control/test_prepass.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from plumbline.prepass import prepass
d = os.path.join(ROOT, "proto", "intro")
P = "deepspeed/runtime/utils.py"
r = prepass(open(f"{d}/ds-8313.diff").read(), {P: open(f"{d}/ds-8313.head.py").read()}, {P: open(f"{d}/ds-8313.base.py").read()})
f = [x for x in r if x["file"] == P][0]
assert "clip_grad_norm_" in f["symbols"], f["symbols"]
assert "power_root_order" in [h["pattern"] for h in f["patterns"]], f["patterns"]
assert "decomposition" in f["families"] and "reference" in f["families"], f["families"]
print("test_prepass: #4915 -> clip_grad_norm_, power_root_order, decomposition + reference: PASS")
