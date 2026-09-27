"""Trojan PR web app (VM1). One app, two listeners: PL_ROLE=judge on :8080 (behind the NetBird reverse
proxy with PIN), PL_ROLE=operator on :8081 (laptop over NetBird P2P). Binds the NetBird IP or loopback
only; the VM has zero inbound ports. Judges may run Sentinel on the hero cases only."""
import hashlib, json, os, threading, uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plumbline import db

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROLE = os.environ.get("PL_ROLE", "judge")   # least privilege if the unit forgets it
CHIP = {"judge": "judge · PIN · via NetBird", "operator": "operator · NetBird peer"}[ROLE]
LABELS = ["pre", "intro", "fixp", "fix"]
_sha = os.path.join(ROOT, "GIT_SHA")
VER = (open(_sha).read().split() or ["dev"])[0][:8] if os.path.exists(_sha) else "dev"   # static cache-bust
HEROES = {"ds-8313", "ds-8533", "st-3921"}          # judge-runnable (PLAN 8: judge on heroes only)
EVAL_HARNESS = "harness-frozen-v2"
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
T = Jinja2Templates(directory=os.path.join(HERE, "templates"))
_LIVE, _CUR = threading.BoundedSemaphore(1), {"id": None}   # one live run at a time: each costs model calls + 4 sandbox jobs


@app.middleware("http")
async def _no_framing(request, call_next):
    r = await call_next(request)
    r.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    return r


def _cases():
    return json.load(open(os.path.join(ROOT, "results", "cases.json")))


def _numbers():
    p = os.path.join(ROOT, "results", "numbers.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _j(v, default=None):
    try:
        return json.loads(v) if isinstance(v, str) else (v if v is not None else default)
    except ValueError:
        return default


def _ctx(request, **kw):
    return {"request": request, "role": ROLE, "chip": CHIP, "ver": VER, **kw}


def _eval_runs(conn):
    """Latest finished eval run per (case, which) on the frozen harness and current planner version."""
    rows = conn.execute("SELECT * FROM runs WHERE status='done' AND json_extract(subject, '$.harness')=? "
                        "ORDER BY created_at", (EVAL_HARNESS,)).fetchall()
    latest = {}
    for r in rows:
        s = _j(r["subject"], {})
        latest[(r["case_id"], s.get("which"))] = r
    return latest


def _statuses(conn, run_id):
    st = {}
    for c in conn.execute("SELECT commit_label, status FROM cells WHERE run_id=?", (run_id,)):
        cur = st.get(c["commit_label"])
        st[c["commit_label"]] = "FAIL" if "FAIL" in (cur, c["status"]) else "PASS" if "PASS" in (cur, c["status"]) else "ERROR"
    return st


@app.get("/")
def index(request: Request):
    conn = db.connect()
    runs, nums = _eval_runs(conn), _numbers()
    ph = (((nums.get("sentinel") or {}).get("posthoc_a5") or {}).get("runs") or {}).get("intro") or {}
    rows = []
    for c in _cases():
        intro, fix = runs.get((c["id"], "intro")), runs.get((c["id"], "fix"))
        rows.append({**c, "sentinel": {"run_id": intro["id"], "verdict": intro["verdict"], "statuses": _statuses(conn, intro["id"])} if intro else None,
                     "sentinel_fix": {"run_id": fix["id"], "verdict": fix["verdict"]} if fix else None,
                     "reviewer": (nums.get("reviewer_by_case") or {}).get(c["id"]),
                     "posthoc": ph.get(c["id"]) if intro and ph.get(c["id"]) and ph[c["id"]]["run_id"] != intro["id"] else None})
    ev = [r for r in rows if r["split"] in ("dev", "held-out")]
    totals = {"n": len(ev), "ran": sum(1 for r in ev if r["sentinel"]),
              "blocked": sum(1 for r in ev if r["sentinel"] and r["sentinel"]["verdict"] == "BLOCK"),
              "fix_ran": sum(1 for r in ev if r["sentinel_fix"]),
              "fix_blocked": sum(1 for r in ev if r["sentinel_fix"] and r["sentinel_fix"]["verdict"] == "BLOCK")}
    return T.TemplateResponse(request, "index.html", _ctx(request, rows=rows, totals=totals, numbers=nums))


def _system():
    """Live status for the strip on /checks and for /healthz: sandbox VM reachable, model, signing key."""
    import httpx
    st = {"sandbox": None, "runsc": None, "model": os.environ.get("PLANNER_MODEL", "deepseek-v4-flash-0731"), "key_id": None}
    try:
        h = httpx.get(os.environ.get("SANDBOXD_URL", "") + "/healthz", timeout=2).json()
        st["sandbox"], st["runsc"] = bool(h.get("ok")), (h.get("runsc_version") or "").replace("runsc version ", "")
    except Exception:
        st["sandbox"] = False
    try:
        conn = db.connect()
        r = conn.execute("SELECT body FROM receipts ORDER BY rowid DESC LIMIT 1").fetchone()
        st["key_id"] = ((_j(r["body"], {}) or {}).get("signature") or {}).get("key_id") if r else None
        m = conn.execute("SELECT json_extract(plan, '$.model') AS m FROM runs WHERE json_extract(plan, '$.model') IS NOT NULL "
                         "ORDER BY created_at DESC LIMIT 1").fetchone()
        if m and m["m"]:
            st["model"] = m["m"]   # the model the planner actually used last, not the configured default
    except Exception:
        pass
    return st


@app.get("/healthz")
def healthz():
    st = _system()
    return JSONResponse({"ok": bool(st["sandbox"]), **st}, status_code=200 if st["sandbox"] else 503)


@app.get("/checks")
def checks(request: Request):
    conn = db.connect()
    rows, seen_pr = [], set()
    for r in conn.execute("SELECT id, kind, case_id, subject, status, verdict, plan, wall_ms, created_at FROM runs "
                          "ORDER BY created_at DESC LIMIT 600"):
        s, p = _j(r["subject"], {}), _j(r["plan"], {}) or {}
        key = (r["case_id"], s.get("which"), r["kind"])
        if key in seen_pr or len(rows) >= 30:   # latest check per distinct PR, so the variety of rules is visible
            continue
        seen_pr.add(key)
        mand = set(p.get("mandatory_ids") or [])
        fams, seen = [], set()
        for c in p.get("checks") or []:
            k = (c.get("family"), c.get("id") in mand)
            if k not in seen:
                seen.add(k)
                fams.append({"family": c.get("family"), "builtin": c.get("id") in mand})
        rows.append({"id": r["id"], "kind": r["kind"], "case_id": r["case_id"], "s": s, "status": r["status"],
                     "verdict": r["verdict"], "fams": fams, "wall_ms": r["wall_ms"], "created_at": r["created_at"],
                     "model": p.get("model")})
    return T.TemplateResponse(request, "checks.html", _ctx(request, rows=rows, sys=_system()))


@app.get("/cases/{cid}")
def case(request: Request, cid: str):
    c = next((c for c in _cases() if c["id"] == cid), None)
    if not c:
        raise HTTPException(404)
    conn = db.connect()
    runs = [dict(r) for r in conn.execute("SELECT id, status, verdict, created_at, wall_ms, subject FROM runs "
                                          "WHERE case_id=? ORDER BY created_at DESC LIMIT 5", (cid,))]
    for r in runs:
        r["subject"] = _j(r["subject"], {})
    runnable = os.path.exists(os.path.join(os.environ.get("PLUMBLINE_INPUTS", "/var/lib/plumbline/inputs"), cid, "commits.json"))
    can_run = runnable and (ROLE == "operator" or cid in HEROES)
    return T.TemplateResponse(request, "case.html", _ctx(request, c=c, runs=runs, can_run=can_run))


def _live(cid, run_id):
    from plumbline.sentinel import run_pr
    try:
        run_pr(cid, "intro", run_id=run_id, cap_s=45)
    finally:
        _CUR["id"] = None
        _LIVE.release()


@app.post("/cases/{cid}/run")
def run_case(cid: str, request: Request):
    if request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
        raise HTTPException(403, "cross-site run requests are refused")
    if ROLE != "operator" and cid not in HEROES:
        raise HTTPException(403, "judges can run the hero cases only")
    if not _LIVE.acquire(blocking=False):
        if _CUR["id"]:
            return RedirectResponse(f"/runs/{_CUR['id']}", status_code=303)   # join the run in progress
        raise HTTPException(429, "a run is starting, try again in a few seconds")
    run_id = _CUR["id"] = f"{cid}-live-{uuid.uuid4().hex[:6]}"
    threading.Thread(target=_live, args=(cid, run_id), daemon=True).start()
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


def _run_view(conn, run_id):
    r = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not r:
        return None
    run = dict(r)
    run["subject"], run["plan"] = _j(r["subject"], {}), _j(r["plan"], {})
    which = run["subject"].get("which", "intro")
    labels = LABELS if which == "intro" else ["base", "head"] if which == "repo" else ["fixp", "fix"]
    head_l = "intro" if which == "intro" else labels[-1]
    mand = set(run["plan"].get("mandatory_ids") or [])
    checks = {c["id"]: c for c in run["plan"].get("checks") or []}
    grid = {}
    for c in conn.execute("SELECT * FROM cells WHERE run_id=? ORDER BY check_id", (run_id,)):
        row = grid.setdefault(c["check_id"], {"check_id": c["check_id"], "family": c["family"],
                                              "target": (c["target"] or "").split(":")[-1], "cells": {}, "decision": None,
                                              "mandatory": c["check_id"].split(".")[0] in mand,
                                              "why": (checks.get(c["check_id"].split(".")[0]) or {}).get("why")})
        row["cells"][c["commit_label"]] = {"status": c["status"], "detail": c["detail"], "metric": _j(c["metric"]),
                                           "threshold": _j(c["threshold"]), "witness": _j(c["witness"], {}), "sha": c["sha"]}
        if c["decision"]:
            row["decision"] = c["decision"]
    headline = None
    for row in grid.values():
        w = row["cells"].get(head_l, {}).get("witness") or {}
        if row["decision"] in ("detected", "detected_new") and "got" in w and "want" in w:
            headline = {"got": f"{w['got']:.6g}", "must": f"{w['want']:.6g}", "check_id": row["check_id"],
                        "context": (row["cells"][head_l]["detail"] or "").split(":")[0]}
            break
    rec = conn.execute("SELECT body FROM receipts WHERE run_id=?", (run_id,)).fetchone()
    receipt = _j(rec["body"]) if rec else None
    return {"run": run, "labels": labels, "grid": list(grid.values()), "headline": headline,
            "receipt": {"sha256": hashlib.sha256(rec["body"].encode()).hexdigest(),
                        "key_id": receipt["signature"]["key_id"], "jobs": receipt["execution"]["jobs"],
                        "runtime": receipt["execution"]["runtime"]} if receipt else None,
            "done": run["status"] == "done"}


@app.get("/runs/{run_id}")
def run_page(request: Request, run_id: str):
    v = _run_view(db.connect(), run_id)
    return T.TemplateResponse(request, "run.html", _ctx(request, run_id=run_id, v=v))


@app.get("/runs/{run_id}/body")
def run_body(request: Request, run_id: str):
    """htmx polls this every second until the run is done."""
    v = _run_view(db.connect(), run_id)
    return T.TemplateResponse(request, "_run_body.html", _ctx(request, run_id=run_id, v=v))


@app.get("/runs/{run_id}/receipt.json")
def receipt_json(run_id: str):
    rec = db.connect().execute("SELECT body FROM receipts WHERE run_id=?", (run_id,)).fetchone()
    if not rec:
        raise HTTPException(404)
    return JSONResponse(json.loads(rec["body"]))


def _pdt(iso):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        return datetime.fromisoformat(iso).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%a %H:%M PDT")
    except (TypeError, ValueError):
        return None


@app.get("/arena")
def arena(request: Request):
    """PLAN 6.6: the stored round from numbers.json; verdicts are the stored Sentinel runs, revealed by R."""
    nums, conn = _numbers(), db.connect()
    game = nums.get("game") or {}
    diffs = {}
    hr = os.path.join(ROOT, "eval", "hand_round.jsonl")
    if os.path.exists(hr):
        for line in open(hr):
            r = json.loads(line)
            diffs.setdefault(r["id"], r.get("diff"))
    cards = []
    for c in (game.get("round") or {}).get("cards") or []:
        c = dict(c)
        run = conn.execute("SELECT plan FROM runs WHERE id=?", (c.get("run_id"),)).fetchone()
        rec = conn.execute("SELECT body FROM receipts WHERE run_id=?", (c.get("run_id"),)).fetchone()
        checks = (_j(run["plan"], {}) if run else {}).get("checks") or []
        c.update(families=sorted({x.get("family") for x in checks if x.get("family")}), n_checks=len(checks),
                 receipt=hashlib.sha256(rec["body"].encode()).hexdigest() if rec else None,
                 ran=_pdt(c.get("ran_at")), diff=diffs.get(c["id"]))
        cards.append(c)
    return T.TemplateResponse(request, "arena.html", _ctx(request, rnd=game.get("round") or {}, cards=cards,
                                                          footer=game.get("footer") or {}))


@app.get("/replay")
def replay():
    """Key P on stage: the latest finished hero run, shown with a REPLAY banner (PLAN 0.2 fallback)."""
    r = db.connect().execute("SELECT id FROM runs WHERE case_id='ds-8313' AND status='done' AND verdict='BLOCK' "
                             "ORDER BY created_at DESC LIMIT 1").fetchone()
    if not r:
        raise HTTPException(404, "no finished hero run to replay")
    return RedirectResponse(f"/runs/{r['id']}?replay=1", status_code=303)
