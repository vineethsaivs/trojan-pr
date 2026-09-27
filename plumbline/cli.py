"""python -m plumbline.cli keygen <private.pem>      (prints the public key; VM1)
python -m plumbline.cli hist <case> [intro|fix]       (historical Sentinel run; VM1)
python -m plumbline.cli verify <receipt.json> --pin <keys/plumbline.pub>   (anywhere; exit 0 = valid)
python -m plumbline.cli rerun <run_id> <harness>     (re-execute a stored plan, no planner call; amendment 5)
python -m plumbline.cli queue [harness-frozen-v1]    (Sentinel triple control: 12 intro runs, then 12 fix runs; resumable)"""
import base64, json, os, sys


def keygen(path):
    from plumbline import receipt
    priv, pub = receipt.keygen()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, priv)
    os.close(fd)
    print(base64.b64encode(pub).decode())


def hist(cid, which="intro"):
    from plumbline.sentinel import run_pr
    r = run_pr(cid, which)
    p = r["plan"]
    print(f"plan: source={p['source']} model={p['model']} latency={p['latency_ms']} ms mandatory={p['mandatory_ids']}")
    for c in p["checks"]:
        print(f"  {c['id']:3} {c['family']:15} {c['adapter']:9} {c['call']['target'].split(':')[-1]}  {c['params']}"[:170])
    if p["errors"]:
        print("  planner errors:", p["errors"][:3])
    for r_ in p.get("rejected", []):
        print(f"  rejected {r_['id']}: {'; '.join(r_['errors'])[:150]}")
    print("statuses:", " / ".join(f"{l} {r['statuses'][l]}" for l in r["labels"]))
    head = r["labels"][1]
    for k, cell in sorted(r["per"][head].items()):
        print(f"  {head} {k:6} {cell['status']:5} {r['decisions'].get(k, ''):12} {str(cell.get('detail'))[:110]}")
    print(f"VERDICT: {r['verdict']} ({r['reason']})")
    print(f"receipt: {r['receipt_path']}  wall: {r['wall_ms'] / 1000:.1f} s")


H = ["ds-8313", "ds-8533", "st-3921", "ds-8334", "ray-65747", "st-4019",
     "st-3868", "ray-65535", "ray-65790", "zoo-1286", "unsloth-11337", "unsloth-11470"]


def queue(harness="harness-frozen-v2"):
    """PLAN 4.9. Skips (case, which) pairs that already have a finished run on this harness."""
    import time
    from plumbline import db
    from plumbline.sentinel import run_pr
    conn = db.connect()
    for which in ("intro", "fix"):
        for cid in H:
            from plumbline.planner import PLANNER_VERSION
            rows = conn.execute("SELECT json_extract(plan, '$.errors') FROM runs WHERE case_id=? AND id LIKE ? "
                                "AND status='done' AND json_extract(subject, '$.harness')=? "
                                "AND json_extract(subject, '$.planner_version')=?",
                                (cid, f"{cid}-{which}-%", harness, PLANNER_VERSION)).fetchall()
            timeouts = sum(1 for (e,) in rows if e and "Timeout" in e)
            if rows and not (timeouts == len(rows) == 1):   # a planner-timeout-only run is rerun once (amendment 1)
                continue
            t = time.time()
            try:
                r = run_pr(cid, which, harness=harness, conn=conn, cap_s=180)   # batch cap (amendment 1)
                print(f"{cid} {which}: {r['verdict']} {r['statuses']} plan={r['plan']['source']} "
                      f"added={len(r['plan']['checks']) - len(r['plan']['mandatory_ids'])} {time.time() - t:.0f} s", flush=True)
            except Exception as e:
                print(f"{cid} {which}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)


def verify(path, pin_path):
    from plumbline import receipt
    ok, msg = receipt.verify(json.load(open(path)), open(pin_path).read().strip())
    print(("VALID: " if ok else "INVALID: ") + msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["keygen"]:
        keygen(a[1])
    elif a[:1] == ["hist"]:
        hist(a[1], a[2] if len(a) > 2 else "intro")
    elif a[:1] == ["rerun"]:
        from plumbline import db
        from plumbline.sentinel import run_pr
        r = db.connect().execute("SELECT case_id, subject, plan FROM runs WHERE id=?", (a[1],)).fetchone()
        plan = {**json.loads(r["plan"]), "from_run": a[1], "model_calls": []}   # no model call in a re-execution
        out = run_pr(r["case_id"], json.loads(r["subject"])["which"], harness=a[2], plan=plan)
        print(a[1], "->", out["run_id"] if "run_id" in out else "", out["verdict"], out["statuses"])
    elif a[:1] == ["queue"]:
        queue(*a[1:2])
    elif a[:1] == ["verify"] and "--pin" in a:
        verify(a[1], a[a.index("--pin") + 1])
    else:
        print(__doc__)
        sys.exit(2)
