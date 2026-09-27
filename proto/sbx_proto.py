"""sandboxd core mechanics, tested with runc locally (runsc on VM2 adds --runtime=runsc):
single-use warm containers, files in by tar stream (docker cp cannot write tmpfs),
JSONL events out on stdout, result by exit, container destroyed after one job."""
import asyncio, io, json, tarfile, time, uuid

IMAGE = "python:3.11-alpine"
RUNTIME = []  # VM2: ["--runtime=runsc-trace"]
WARM = r'''
import os, sys, time, json
# VM2 image: `import torch` here, before the job exists, so jobs skip the import cost
t0 = time.time()
while not os.path.exists("/work/GO"):
    time.sleep(0.005)
job = json.load(open("/work/job.json"))
print(json.dumps({"ev": "started", "waited_s": round(time.time() - t0, 3)}), flush=True)
sys.path.insert(0, "/work")
exec(compile(open("/work/" + job["entry"]).read(), job["entry"], "exec"), {"__name__": "__main__"})
'''
FLAGS = ["--network=none", "--read-only", "--tmpfs", "/work:rw,size=256m,mode=1777", "--tmpfs", "/tmp:rw,size=64m",
         "--cpus=1", "--memory=1g", "--pids-limit=128", "--cap-drop=ALL", "--security-opt=no-new-privileges",
         "--user=65534:65534", "--env", "OMP_NUM_THREADS=1"]


async def sh(*args, stdin=None):
    p = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE if stdin else None,
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await p.communicate(stdin)
    return p.returncode, out.decode(), err.decode()


async def warm_one():
    name = f"pl-{uuid.uuid4().hex[:8]}"
    rc, out, err = await sh("docker", "run", "-d", "--name", name, *RUNTIME, *FLAGS, IMAGE, "python", "-c", WARM)
    assert rc == 0, err
    return name


def tar_bytes(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for path, text in files.items():
            data = text.encode()
            ti = tarfile.TarInfo(path); ti.size = len(data); ti.mode = 0o644
            t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


async def run_job(name, files, entry, timeout=30):
    t = time.time()
    files = {**files, "job.json": json.dumps({"entry": entry})}
    rc, _, err = await sh("docker", "exec", "-i", name, "tar", "-x", "-C", "/work", stdin=tar_bytes(files))
    assert rc == 0, err
    await sh("docker", "exec", name, "touch", "/work/GO")
    try:
        rc, out, _ = await asyncio.wait_for(sh("docker", "wait", name), timeout)
        code = int(out.strip())
    except asyncio.TimeoutError:
        await sh("docker", "kill", name); code = 124
    _, logs, errs = await sh("docker", "logs", name)
    await sh("docker", "rm", "-f", name)
    events = [json.loads(l) for l in logs.splitlines() if l.startswith("{")]
    return {"exit": code, "wall_s": round(time.time() - t, 3), "events": events, "stderr_tail": errs[-300:]}


JOB = r'''
import json, os, socket
def ev(**k): print(json.dumps(k), flush=True)
try:
    open("/etc/passwd").read(); ev(kind="read_ok", path="/etc/passwd")
except Exception as e: ev(kind="read_fail", err=str(e))
try:
    socket.create_connection(("1.1.1.1", 443), timeout=2); ev(kind="egress_ok")
except Exception as e: ev(kind="egress_blocked", err=type(e).__name__ + ": " + str(e))
try:
    open("/usr/lib/evil.py", "w").write("x"); ev(kind="rootfs_write_ok")
except Exception as e: ev(kind="rootfs_write_blocked", err=type(e).__name__)
try:
    import subprocess; [subprocess.Popen(["sleep", "5"]) for _ in range(200)]; ev(kind="fork_bomb_ok")
except Exception as e: ev(kind="pids_capped", err=type(e).__name__)
ev(kind="result", status="PASS", uid=os.getuid())
'''


async def main():
    t = time.time()
    pool = await asyncio.gather(*[warm_one() for _ in range(3)])
    print("warm 3 containers:", round(time.time() - t, 2), "s")
    t = time.time()
    res = await asyncio.gather(*[run_job(n, {"job.py": JOB}, "job.py") for n in pool])
    print("3 jobs in parallel:", round(time.time() - t, 2), "s")
    for r in res[:1]:
        print(json.dumps(r, indent=1))
    print([r["wall_s"] for r in res])


if __name__ == "__main__":
    asyncio.run(main())
