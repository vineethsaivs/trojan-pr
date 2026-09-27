# ds-8199 oracle: DeepSpeed CommsLogger straggler + no-mutation, replayed at real SHAs.
# Fetches deepspeed/utils/{timer,comms_logging}.py and deepspeed/comm/constants.py at each SHA
# (read-only gh api), stubs deepspeed.comm -> torch.distributed and runs 2 real gloo ranks on CPU.
#
# Oracles restate the math, never the implementation:
#   O1 no-mutation: log_all() is a read. The stored records must be exactly as recorded afterwards
#      (exact deep equality, no tolerance).
#   O2 straggler: for index-aligned latencies x[r][i] (i = same collective on every rank),
#      straggler_r = sum_i x[r][i] - sum_i min_r' x[r'][i].
#      Tolerance: only a few float adds and mins, exact to ~1e-12. Pre-#7404 code only prints a
#      .2f table, so we allow 5e-3 (half a printed ulp). Bug-vs-truth gaps here are >= 0.4.
# usage: python oracle.py            (runs every SHA below, exit 0 iff every verdict is as expected)
import base64, contextlib, copy, inspect, io, json, os, socket, subprocess, sys

import torch.distributed as dist
import torch.multiprocessing as mp

REPO = "deepspeedai/DeepSpeed"
SHAS = [  # (label, sha, expected overall verdict)
    ("pre-intro #3579^", "fd1d2c64472c1a3061a05eb3b56a3f882199cfba", "FAIL"),  # no show_straggler yet; O1 already broken since #2012
    ("intro #3579", "5d1124f2aaff1af3dd1fab8f142ae09cc25cc228", "FAIL"),
    ("fix parent", "e1d6b4fe43626ca74df3defab06d747e0b499995", "FAIL"),
    ("fix #8199", "b8b448096524f7bfe81a709a14b57a1da83dfe75", "PASS"),
]
FILES = ["deepspeed/utils/timer.py", "deepspeed/utils/comms_logging.py", "deepspeed/comm/constants.py"]
STUBS = {
    "deepspeed/__init__.py": "",
    "deepspeed/utils/__init__.py": "def log_dist(*a, **k):\n    pass\n",
    "deepspeed/utils/logging.py": "def log_dist(*a, **k):\n    pass\n\n\ndef print_dist(*a, **k):\n    pass\n",
    "deepspeed/comm/__init__.py": "from torch.distributed import all_reduce, get_world_size, get_rank, is_initialized\n",
    "deepspeed/comm/reduce_op.py": "from torch.distributed import ReduceOp\n",
    "deepspeed/accelerator/__init__.py": (
        "class _A:\n    Event = object\n\n    def current_device_name(self):\n        return 'cpu'\n\n\n"
        "def get_accelerator():\n    return _A()\n"),
}
CASES = {
    # per-stage allreduce times from the #3579 author's worked example in review (discussion_r1247360222)
    "review_example": [[0.5, 0.1], [0.1, 0.9]],
    # 4-op case from the #8199 PR body
    "pr_body_4op": [[2.30, 1.60, 3.60, 1.29], [3.14, 2.46, 1.23, 3.03]],
}
MSG = 1024
HERE = os.path.dirname(os.path.abspath(__file__))


def materialize(sha):
    root = os.path.join(HERE, "src", sha[:8])
    for rel, text in STUBS.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)
    for rel in FILES:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            b64 = subprocess.check_output(["gh", "api", f"repos/{REPO}/contents/{rel}?ref={sha}", "--jq", ".content"])
            open(p, "wb").write(base64.b64decode(b64))
    return root


def true_straggler(per_rank, r):  # the oracle: own total minus sum of per-collective minimums
    return sum(per_rank[r]) - sum(min(col) for col in zip(*per_rank))


def rank_main(rank, root, port, q):
    sys.path.insert(0, root)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from deepspeed.utils.comms_logging import CommsLogger

    params = inspect.signature(CommsLogger.log_all).parameters
    out = {}
    for name, per_rank in CASES.items():
        lats = per_rank[rank]
        algbw = [10.0 / x for x in lats]  # algbw derives from latency (calc_bw_log), so lat*algbw is constant
        busbw = [2 * a for a in algbw]
        cl = CommsLogger()
        cl.comms_dict = {"all_reduce": {MSG: [len(lats), list(lats), list(algbw), list(busbw)]}}
        before = copy.deepcopy(cl.comms_dict)
        got = None
        if "show_straggler" not in params:  # pre-#3579: plain summary, O2 not applicable
            with contextlib.redirect_stdout(io.StringIO()):
                cl.log_all()
        elif "return_dict" in params:
            res = cl.log_all(print_log=False, show_straggler=True, return_dict=True)
            got = res["straggler_analysis"]["all_reduce"][MSG]["total_straggler_ms"]
        else:  # pre-#7404: parse the fixed-width (<20) printed straggler table, 5th column
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cl.log_all(print_log=True, show_straggler=True)
            row = buf.getvalue().split("Breakdown with straggler effect")[1].splitlines()[-1]
            got = float(row[80:100])
        out[name] = dict(got=got, want=true_straggler(per_rank, rank), mutated=cl.comms_dict != before,
                         after_lat=cl.comms_dict["all_reduce"][MSG][1])
    q.put((rank, out))
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    all_as_expected = True
    for label, sha, expected in SHAS:
        root = materialize(sha)
        q = mp.get_context("spawn").SimpleQueue()
        mp.spawn(rank_main, args=(root, free_port(), q), nprocs=2, join=True)
        res = dict(q.get() for _ in range(2))
        ok = True
        for name in CASES:
            for r in (0, 1):
                o = res[r][name]
                o1 = not o["mutated"]
                o2 = None if o["got"] is None else abs(o["got"] - o["want"]) <= 5e-3
                ok &= o1 and o2 is not False
                s2 = "n/a (no show_straggler)" if o2 is None else \
                    f"got={o['got']:.4f} want={o['want']:.4f} {'PASS' if o2 else 'FAIL'}"
                print(f"  {sha[:8]} {name:14} rank{r}  O2 straggler {s2:34} "
                      f"O1 no-mutation {'PASS' if o1 else 'FAIL'} (latencies after: {o['after_lat']})")
        verdict = "PASS" if ok else "FAIL"
        all_as_expected &= verdict == expected
        print(f"{label:18} {sha[:8]}: {verdict} (expected {expected})\n")
    print("ALL AS EXPECTED" if all_as_expected else "UNEXPECTED VERDICT")
    sys.exit(0 if all_as_expected else 1)
