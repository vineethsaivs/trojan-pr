"""Runs ONLY inside gVisor. Reads /job/job.json, prints one JSON line per event (PLAN 5.11):
{"ev":"started"} {"ev":"check",...} {"ev":"done","n","degraded"}. Per-check alarm; timeout = ERROR.
Per-check loop ported from proto/run_cases.py."""
import json, os, signal, sys, time
import torch
from extract import Loader
from shims import SHIMS
import oracles as O
import adapters
import gold


# Events go to a private dup of stdout; fd 1 is pointed at stderr so prints from the code under
# test never reach the result stream. ponytail: in-process harness, a deliberate attacker can still
# find the fd; sandboxd + judge treat duplicate check ids as tampering (5.12).
_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def emit(**k):
    _OUT.write(json.dumps(k, default=str) + "\n")


class _Timeout(BaseException):  # not Exception: oracles' _call must not turn a timeout into a FAIL
    pass


def _alarm(signum, frame):
    raise _Timeout()


def write_repo(job):
    """repo and ci modes: the PR's files are written to the tmpfs and imported normally."""
    root = "/work/repo" if os.path.isdir("/work") else os.path.join(os.environ.get("TMPDIR", "/tmp"), f"repo-{job['job_id']}")
    for rel, src in job["files"].items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(src)
    sys.path.insert(0, root)
    os.chdir(root)
    return root


def run_ci(job):
    import subprocess
    root = write_repo(job)
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=root,
                           capture_output=True, text=True, timeout=int(job.get("ci_timeout_s", 240)))
        code, out = r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired as e:
        code, out = "timeout", str(e.stdout or "")
    emit(ev="ci", exit=code, tail="\n".join(out.splitlines()[-60:]))
    emit(ev="done", n=0, degraded=[])


def load_module(job):
    if job.get("mode") == "repo":
        write_repo(job)
        return None, None
    L = Loader(job["files"], job["root_pkg"], SHIMS.get(job["root_pkg"], {}))
    try:
        return L.load(job["module"]), L
    finally:
        L.uninstall()


def run_check(chk, mod, job):
    """One check spec -> list of Verdicts: a gold plan by case id, or a typed plan check."""
    if "gold" in chk:
        return gold.PLANS[chk["gold"]](mod)
    return adapters.run_plan_check(chk, mod, job)


def main(path):
    job = json.load(open(path))
    emit(ev="started", job_id=job["job_id"], torch=torch.__version__)
    torch.manual_seed(job.get("seed", 0))
    signal.signal(signal.SIGALRM, _alarm)
    n, degraded = 0, []
    if job.get("mode") == "ci":
        return run_ci(job)
    try:
        mod, L = load_module(job)
        degraded = L.degraded[:5] if L else []
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
        fam = chk.get("family", "harness")
        try:
            vs = run_check(chk, mod, job)
        except _Timeout:
            vs = [O.Verdict(fam, "ERROR", detail="timeout", witness={"kind": "timeout"})]
        except O.HarnessError as e:
            vs = [O.Verdict(fam, "ERROR", detail=str(e), witness={"kind": getattr(e, "kind", "binding")})]
        except Exception as e:
            vs = [O.Verdict(fam, "ERROR", detail=f"{type(e).__name__}: {e}", witness={"kind": "harness"})]
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
