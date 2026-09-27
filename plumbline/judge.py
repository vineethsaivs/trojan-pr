"""Decision rule (PLAN 5.7), VM1. Recomputes every status from metric and threshold: the sandbox's
own PASS/FAIL is never trusted where a number exists. Duplicate or unknown check ids in a job's
output are treated as tampering (the check becomes ERROR)."""
import math

TABLE = {("PASS", "FAIL"): "detected", ("PASS", "PASS"): "clean", ("FAIL", "FAIL"): "preexisting",
         ("FAIL", "PASS"): "improved"}
BLOCKING = {"detected", "detected_new"}


def _num(x):
    if isinstance(x, str):          # receipts store non-finite metrics as strings
        try:
            return float(x)
        except ValueError:
            return None
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def recompute(line):
    """Status from the numbers when both exist; a non-finite metric is a FAIL."""
    if line.get("status") == "ERROR":
        return "ERROR"
    m, t = _num(line.get("metric")), _num(line.get("threshold"))
    if m is not None and t is not None:
        return "PASS" if math.isfinite(m) and m <= t else "FAIL"
    if m is not None and not math.isfinite(m):
        return "FAIL"
    return line.get("status") if line.get("status") in ("PASS", "FAIL") else "ERROR"


def cells(lines, plan_ids):
    """Job output lines -> {check_id: cell}. plan_ids: ids the plan scheduled (sub-ids c1.0 allowed)."""
    out, seen = {}, set()
    for l in lines:
        if l.get("ev") != "check":
            continue
        cid = str(l.get("id"))
        known = cid in plan_ids or cid.split(".")[0] in plan_ids
        if cid in seen or not known:
            out[cid] = {**l, "status": "ERROR", "kind": "tamper", "detail": "duplicate or unknown check id"}
            continue
        seen.add(cid)
        out[cid] = {**l, "status": recompute(l), "kind": (l.get("witness") or {}).get("kind")}
    return out


def decide(base, head):
    """base/head: cells (dicts with status, kind) or None when the job produced nothing."""
    if base is None or head is None:
        return "inconclusive"
    b, h = base["status"], head["status"]
    if b == "ERROR" and base.get("kind") == "missing_target":
        return {"FAIL": "detected_new", "PASS": "clean_new"}.get(h, "inconclusive")
    return TABLE.get((b, h), "inconclusive")


def verdict(decisions, targets, touched):
    """decisions {check_id: decision}; targets {check_id: 'path.py:Qual'}; touched: touched symbols
    ('path.py:Qual'). BLOCK if anything blocks; PASS only if every touched symbol has a clean check;
    else NOT COVERED (never green when nothing ran)."""
    if any(d in BLOCKING for d in decisions.values()):
        bad = sorted(c for c, d in decisions.items() if d in BLOCKING)
        return "BLOCK", f"invariant broken by the change: {', '.join(bad)}"
    clean = {targets[c.split('.')[0]] for c, d in decisions.items() if d in ("clean", "clean_new")}
    want = set(touched) or ({"*"} if not clean else set())
    covered = {t for t in want if any(c == t or c.startswith(t + ".") or t.startswith(c + ".") for c in clean)}
    if clean and covered == want:
        return "PASS", f"{len(clean)} clean target(s); nothing blocking"
    return "NOT COVERED", f"no clean check for: {', '.join(sorted(want - covered)) or 'any touched symbol'}"
