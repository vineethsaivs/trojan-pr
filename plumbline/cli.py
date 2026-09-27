"""python -m plumbline.cli keygen <private.pem>      (prints the public key; VM1)
python -m plumbline.cli hist <case> [intro|fix]       (historical Sentinel run; VM1)
python -m plumbline.cli verify <receipt.json> --pin <keys/plumbline.pub>   (anywhere; exit 0 = valid)"""
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
    print("statuses:", " / ".join(f"{l} {r['statuses'][l]}" for l in r["labels"]))
    head = r["labels"][1]
    for k, cell in sorted(r["per"][head].items()):
        print(f"  {head} {k:6} {cell['status']:5} {r['decisions'].get(k, ''):12} {str(cell.get('detail'))[:110]}")
    print(f"VERDICT: {r['verdict']} ({r['reason']})")
    print(f"receipt: {r['receipt_path']}  wall: {r['wall_ms'] / 1000:.1f} s")


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
    elif a[:1] == ["verify"] and "--pin" in a:
        verify(a[1], a[a.index("--pin") + 1])
    else:
        print(__doc__)
        sys.exit(2)
