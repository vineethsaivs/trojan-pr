"""Parse runsc --strace lines from the sandbox's *.boot log and raise containment events.
Line shape from gVisor source: task prefix "[% 4d:% 4d] " (kernel/task_log.go L199) then
"%s E %s(%s, ...)" on entry and "%s X %s(...) = %s" on exit (strace/strace.go L561, L598)."""
import re

LINE = re.compile(r"\[\s*(?P<pid>\d+)(?:\(\s*\d+\))?:\s*(?P<tid>\d+)(?:\(\s*\d+\))?\] (?P<comm>\S+) "
                  r"(?P<ph>[EX]) (?P<sc>\w+)\((?P<args>.*?)\)(?: = (?P<ret>.*))?$")
CANARY_PATHS = ("/home/runner/.aws/credentials", "/home/runner/.ssh/id_ed25519", "/home/runner/.netrc",
                "/work/.env", "/home/runner/.config/gh/hosts.yml")
SUSPECT_EXEC = ("curl", "wget", "nc", "ncat", "ssh", "scp", "socat", "python -c")


def events(lines, canary_tokens=()):
    out, counts = [], {}
    for line in lines:
        m = LINE.search(line)
        if not m or m["ph"] != "E":
            continue
        sc, args = m["sc"], m["args"]
        counts[sc] = counts.get(sc, 0) + 1
        if sc == "openat" and any(p in args for p in CANARY_PATHS):
            out.append({"kind": "canary_read", "pid": int(m["pid"]), "comm": m["comm"], "detail": args[:160]})
        elif sc in ("connect", "socket") and "AF_UNIX" not in args:
            out.append({"kind": "egress_attempt", "pid": int(m["pid"]), "comm": m["comm"], "detail": args[:160]})
        elif sc == "execve" and any(x in args for x in SUSPECT_EXEC):
            out.append({"kind": "suspicious_exec", "pid": int(m["pid"]), "comm": m["comm"], "detail": args[:160]})
        if any(t in line for t in canary_tokens):
            out.append({"kind": "canary_token_seen", "pid": int(m["pid"]), "detail": sc})
    return counts, out


if __name__ == "__main__":
    sample = [
        "I0926 16:00:00.000001       1 strace.go:567] [   7:   7] python3 E openat(AT_FDCWD /work/train.py, O_RDONLY|O_CLOEXEC, 0o0)",
        "I0926 16:00:00.000002       1 strace.go:567] [   7:   9] python3 E openat(AT_FDCWD /home/runner/.aws/credentials, O_RDONLY, 0o0)",
        "I0926 16:00:00.000003       1 strace.go:567] [   7:   9] python3 E connect(0x3 socket:[4], {Family: AF_INET, Addr: 1.2.3.4, Port: 443}, 0x10)",
        "I0926 16:00:00.000004       1 strace.go:604] [   7:   9] python3 X connect(0x3 socket:[4], {...}, 0x10) = 0 (0x0) errno=101 (network is unreachable)",
        "I0926 16:00:00.000005       1 strace.go:567] [   8:   8] sh E execve(/usr/bin/curl, [curl http://x], [])",
    ]
    c, ev = events(sample)
    print(c)
    for e in ev:
        print(e)
    assert [e["kind"] for e in ev] == ["canary_read", "egress_attempt", "suspicious_exec"], ev
