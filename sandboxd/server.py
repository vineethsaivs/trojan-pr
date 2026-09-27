"""sandboxd (VM2): runs one job per gVisor container with the PLAN 5.11 profile. Listens on the
NetBird IP only. Two routes. No secret ever reaches a container."""
import base64, hashlib, json, os, re, secrets, subprocess, threading, time
from pathlib import Path
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

TOKEN = os.environ["SANDBOXD_TOKEN"]
JOBS, OPT = Path("/srv/jobs"), Path("/opt/plumbline")
IMAGE = "pl/runner:cpu"
RUNTIME = os.environ.get("SANDBOXD_RUNTIME", "runsc-trace")
JOB_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
HARNESS = re.compile(r"^harness(-frozen-v\d+)?$")
SLOTS = threading.Semaphore(3)
app = FastAPI()


def _out(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()


RUNSC_VERSION = _out("runsc", "--version").splitlines()[0]
RUNTIMES = sorted(json.loads(_out("docker", "info", "--format", "{{json .Runtimes}}")))
IMAGE_ID = _out("docker", "image", "inspect", "-f", "{{.Id}}", IMAGE)


def harness_sha256(name):
    h = hashlib.sha256()
    for p in sorted((OPT / name).rglob("*.py")):
        h.update(p.relative_to(OPT / name).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


class Job(BaseModel):
    job_id: str
    harness: str = "harness"
    files: dict[str, str]          # relpath -> base64; must include job.json
    timeout_s: int = 120


@app.get("/healthz")
def healthz():
    return {"ok": True, "runsc_version": RUNSC_VERSION, "runtimes": RUNTIMES}


def auth(authorization: str = Header("")):  # a dependency, so it runs before body validation
    if not secrets.compare_digest(authorization, f"Bearer {TOKEN}"):
        raise HTTPException(401)


@app.post("/v1/jobs", dependencies=[Depends(auth)])
def run_job(job: Job):
    if not JOB_ID.match(job.job_id) or not HARNESS.match(job.harness) or not (OPT / job.harness).is_dir():
        raise HTTPException(400, "bad job_id or harness")
    if "job.json" not in job.files or not 1 <= job.timeout_s <= 900:
        raise HTTPException(400, "job.json missing or timeout out of range")
    d = JOBS / job.job_id
    if d.exists():
        raise HTTPException(409, "job_id already used")
    for rel, b64 in job.files.items():
        p = (d / rel).resolve()
        if not str(p).startswith(str(d.resolve()) + "/"):
            raise HTTPException(400, f"bad path {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(base64.b64decode(b64))
    name = f"pl-{job.job_id}"
    cmd = ["docker", "run", "--rm", "--name", name, f"--runtime={RUNTIME}", "--network", "none", "--read-only",
           "--tmpfs", "/work:rw,size=512m,mode=1777", "--tmpfs", "/tmp:rw,size=256m,mode=1777",
           "-v", f"{d}:/job:ro", "-v", f"{OPT / job.harness}:/harness:ro",
           "-v", f"{OPT / 'target/minigpt'}:/opt/minigpt:ro",
           "--cpus", "1", "--memory", "3g", "--pids-limit", "256", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges", "--user", "10001:10001",
           "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
           IMAGE, "python", "/harness/runjob.py", "/job/job.json"]
    with SLOTS:
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=job.timeout_s)
            exit_code, stdout, stderr = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "kill", name], capture_output=True)
            exit_code, stdout, stderr = "timeout", (e.stdout or b"").decode(), (e.stderr or b"").decode()
        wall_ms = int((time.time() - t0) * 1000)
    lines = []
    for line in stdout.splitlines():
        try:
            lines.append(json.loads(line))
        except ValueError:
            pass  # non-JSON prints from the code under test are dropped; stderr_tail keeps context
    return {"job_id": job.job_id, "exit_code": exit_code, "lines": lines, "stderr_tail": stderr[-2000:],
            "wall_ms": wall_ms, "runtime": RUNTIME, "image_id": IMAGE_ID,
            "harness_sha256": harness_sha256(job.harness)}
