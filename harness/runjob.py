"""Runs ONLY inside gVisor. Reads /job/job.json, prints one JSON line per event (PLAN 5.11):
{"ev":"started"} {"ev":"check",...} {"ev":"done","n","degraded"}. Per-check alarm; timeout = ERROR.
Per-check loop ported from proto/run_cases.py."""
import json, os, signal, sys, time
import torch
from extract import Loader
from shims import SHIMS
import oracles as O
import gold


# Events go to a private dup of stdout; fd 1 is pointed at stderr so prints from the code under
# test never reach the result stream. ponytail: in-process harness, a deliberate attacker can still
# find the fd; sandboxd + judge treat duplicate check ids as tampering (5.12).
_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def emit(**k):
    _OUT.write(json.dumps(k, default=str) + "\n")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def load_module(job):
    L = Loader(job["files"], job["root_pkg"], SHIMS.get(job["root_pkg"], {}))
    try:
        return L.load(job["module"]), L
    finally:
        L.uninstall()


def run_check(chk, mod):
    """One check spec -> list of Verdicts. Only gold plans (selected by case id) in block C."""
    if "gold" in chk:
        return gold.PLANS[chk["gold"]](mod)
    raise O.HarnessError(f"unsupported check spec: {sorted(chk)}")


def main(path):
    job = json.load(open(path))
    emit(ev="started", job_id=job["job_id"], torch=torch.__version__)
    torch.manual_seed(job.get("seed", 0))
    signal.signal(signal.SIGALRM, _alarm)
    n, degraded = 0, []
    try:
        mod, L = load_module(job)
        degraded = L.degraded[:5]
    except Exception as e:
        for chk in job["checks"]:
            emit(ev="check", id=chk["id"], family="harness", status="ERROR", metric=None, threshold=None,
                 detail=f"load: {type(e).__name__}: {e}"[:300], witness={}, ms=0)
            n += 1
        emit(ev="done", n=n, degraded=degraded)
        return
    for chk in job["checks"]:
        t0 = time.time()
        signal.alarm(int(job.get("per_check_timeout_s", 20)))
        try:
            vs = run_check(chk, mod)
        except _Timeout:
            vs = [O.Verdict("harness", "ERROR", detail="timeout")]
        except O.HarnessError as e:
            vs = [O.Verdict("harness", "ERROR", detail=str(e))]
        except Exception as e:
            vs = [O.Verdict("harness", "ERROR", detail=f"{type(e).__name__}: {e}")]
        finally:
            signal.alarm(0)
        ms = int((time.time() - t0) * 1000)
        for i, v in enumerate(vs):
            emit(ev="check", id=f"{chk['id']}.{i}" if len(vs) > 1 else chk["id"], ms=ms,
                 **{k: (v.detail[:300] if k == "detail" else val) for k, val in v.to_json().items()})
            n += 1
    emit(ev="done", n=n, degraded=degraded)


if __name__ == "__main__":
    main(sys.argv[1])
