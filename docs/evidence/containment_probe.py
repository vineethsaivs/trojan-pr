"""Containment probe: run INSIDE a job container with sandboxd's exact docker flags (see sandboxd/server.py).
Every line should say BLOCKED (or show the unprivileged identity)."""
import os, socket, urllib.request

def attempt(name, fn):
    try:
        fn()
        print(f"{name}: ALLOWED")
    except Exception as e:
        print(f"{name}: BLOCKED ({type(e).__name__})")

attempt("egress tcp 1.1.1.1:443", lambda: socket.create_connection(("1.1.1.1", 443), timeout=3))
attempt("dns resolve github.com", lambda: socket.getaddrinfo("github.com", 443))
attempt("cloud metadata 169.254.169.254", lambda: urllib.request.urlopen("http://169.254.169.254/v1.json", timeout=3))
attempt("VM1 NetBird IP :8080", lambda: socket.create_connection((os.environ.get("VM1_NB", "100.80.21.27"), 8080), timeout=3))
for path in ("/etc/probe", "/job/probe", "/harness/probe", "/usr/probe"):
    attempt(f"write {path}", lambda p=path: open(p, "w").write("x"))
attempt("write /work/probe (tmpfs, allowed by design)", lambda: open("/work/probe", "w").write("x"))
print("uid/gid:", os.getuid(), os.getgid())
print("kernel:", os.uname().release)
print("env keys:", sorted(k for k in os.environ if k not in ("PATH", "HOME", "HOSTNAME", "LANG", "PYTHONDONTWRITEBYTECODE", "PYTHON_VERSION", "PYTHON_SHA256", "VM1_NB")))
