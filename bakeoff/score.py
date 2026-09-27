"""Bake-off scoring, exactly the pre-registered rubric (PROTOCOL.md, PLAN 4.8). Text only: nothing executes.

  python bakeoff/score.py --llm-judge   same model, same rubric, every adjudication item -> bakeoff/labels_llm.jsonl
  python bakeoff/score.py               results.csv + scores.json (labels.jsonl wins, LLM label for leftovers;
                                        labels.jsonl rows carry "by": human, or panel = 3 blind agents, majority, amendment 6)

Mechanical, per output:
  L = a finding names the scope file (basename ok) and the function (FUNC below, the rubric's function
      column as identifiers) or a line range within 3 lines of that function in the scope file at the PR head.
  A = verdict REQUEST_CHANGES, or an L-passing (intro) / any (fix) finding with severity high or medium.
Human (bakeoff/adjudicate.py) or LLM judge for leftovers:
  intro items (L-passing outputs): M = states the mechanism class, T = names a trigger from the set.
  fix items (actionable outputs): F = claims a numerical defect the adjudicator rejects (false alarm).
CAUGHT = L and M and A. PARTIAL = L and (M or T), not CAUGHT. MISSED = the rest (errors stay in the denominator).
Best cell = highest mean CAUGHT; ties go to fewer false alarms, then p1."""
import ast, csv, glob, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B = os.path.join(ROOT, "bakeoff")
H = ["ds-8313", "ds-8533", "st-3921", "ds-8334", "ray-65747", "st-4019",
     "st-3868", "ray-65535", "ray-65790", "zoo-1286", "unsloth-11337", "unsloth-11470"]
FUNC = {"ds-8313": ["clip_grad_norm_"], "ds-8533": ["zeropower_via_gram_newtonschulz"], "st-3921": ["compute_gor"],
        "ds-8334": ["fixed_linear", "fixed_root"], "ray-65747": ["explore"],
        "st-4019": ["batch_hard_triplet_soft_margin_loss"], "st-3868": ["__init__", "similarity_fct"],
        "ray-65535": ["Quantized"], "ray-65790": ["_quantiles"], "zoo-1286": ["grpo_compute_loss"],
        "unsloth-11337": ["compute_loss"], "unsloth-11470": ["_create_unsloth_optimizer"]}
HUMAN, LLM = os.path.join(B, "labels.jsonl"), os.path.join(B, "labels_llm.jsonl")


def cases():
    import yaml
    return {c["id"]: c for c in yaml.safe_load(open(os.path.join(B, "cases.yaml")))["cases"]}


def ranges(case, which):
    """Line ranges (+-3) of every def/class in the scope file at the PR head whose name holds a FUNC token."""
    src = open(os.path.join(B, "inputs", case, which, "scope.py")).read()
    return [(n.lineno - 3, n.end_lineno + 3) for n in ast.walk(ast.parse(src))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and any(t in n.name for t in FUNC[case])]


def located(f, case, path, rng):
    if os.path.basename(str(f.get("file") or "")) != os.path.basename(path):
        return False
    if any(t in str(f.get("function") or "") for t in FUNC[case]):
        return True
    try:
        a, b = int(f.get("line_start")), int(f.get("line_end") or f.get("line_start"))
    except (TypeError, ValueError):
        return False
    return any(a <= hi and b >= lo for lo, hi in rng)


def salvage(content):
    """Malformed top-level JSON (8 of 144 outputs; run.py's first-object parse then returned one finding):
    keep the verdict string and every well-formed finding object after the last </think>. Favours the reviewer."""
    c = content.split("</think>")[-1]
    v = re.search(r'"verdict"\s*:\s*"(APPROVE|REQUEST_CHANGES)"', c)
    fs = []
    for m in re.finditer(r'\{\s*"file"', c):
        try:
            fs.append(json.JSONDecoder().raw_decode(c[m.start():])[0])
        except ValueError:
            pass
    return {"verdict": v and v[1], "findings": fs}


def outputs():
    """Every raw output with its mechanical L and A, and the findings an adjudicator must see."""
    cs, out = cases(), []
    for fn in sorted(glob.glob(os.path.join(B, "raw", "*", "*", "*", "r*.json"))):
        r = json.load(open(fn))
        case, which = r["case"], r["which"]
        c = cs[case]
        path = c["intro_path" if which == "intro" else "fix_path"]
        p = r.get("parsed") if isinstance(r.get("parsed"), dict) else {}
        salvaged = "verdict" not in p
        if salvaged:
            p = salvage(r.get("content") or "")
        fs = [f for f in (p.get("findings") or []) if isinstance(f, dict)]
        rc = p.get("verdict") == "REQUEST_CHANGES"
        rng = ranges(case, which)
        lf = [f for f in fs if located(f, case, path, rng)]
        sev = lambda f: str(f.get("severity", "")).lower() in ("high", "medium")
        show = lf if which == "intro" else [f for f in fs if rc or sev(f)]
        out.append({"key": f"{case}/{which}/{r['model']}/{r['prompt']}/r{r['run']}", "case": case, "which": which,
                    "model": r["model"], "prompt": r["prompt"], "run": int(r["run"]), "status": r["status"],
                    "L": bool(lf), "A": rc or any(sev(f) for f in (lf if which == "intro" else fs)),
                    "show": show, "repo": c["repo"], "path": path, "function": c["function"],
                    "mechanism_class": c["mechanism_class"], "trigger_set": c["trigger_set"],
                    "split": c["split"], "verdict": p.get("verdict"), "salvaged": salvaged})
    return out


def items():
    """Adjudication queue: L-passing intro outputs and actionable fix outputs, shuffled (seed fixed)."""
    q = [o for o in outputs() if o["status"] == "ok" and o["show"] and (o["L"] if o["which"] == "intro" else o["A"])]
    random.Random(20260927).shuffle(q)
    return q


def load(fn):
    d = {}
    if os.path.exists(fn):
        for line in open(fn):
            if line.strip():
                r = json.loads(line)
                d[r["key"]] = r
    return d


JUDGE = """You adjudicate one AI code review against a fixed rubric. Answer with one JSON object only.
Repository: {repo}. Scope file: {path}. Function: {function}.
{task}
Review findings (JSON):
{findings}"""
T_INTRO = """This PR introduced a known numerical bug. Mechanism class: "{mechanism_class}". Trigger set: "{trigger_set}".
mechanism = true only if a finding states this mechanism class (the same wrong computation), not just a
nearby concern. trigger = true only if a finding names a trigger from the set.
Answer: {{"mechanism": true|false, "trigger": true|false, "why": "<one sentence>"}}"""
T_FIX = """This PR is a merged fix (for: "{mechanism_class}") and is the benign control. Its diff:
<diff>{diff}</diff>
false_alarm = true if a finding claims this change introduces a numerical defect (wrong values, NaN or inf,
wrong scaling or rounding) that is not real. A finding that names a real remaining bug, or claims only
style, typing, performance or a crash on invalid input, is not a false alarm.
Answer: {{"false_alarm": true|false, "why": "<one sentence>"}}"""


def judge_one(o):
    from bakeoff.run import first_json
    from plumbline import llm
    diff = open(os.path.join(B, "inputs", o["case"], "fix", "g2.diff")).read()[:12000] if o["which"] == "fix" else ""
    task = (T_INTRO if o["which"] == "intro" else T_FIX).format(diff=diff, **o)
    msg = JUDGE.format(task=task, findings=json.dumps(o["show"], indent=1)[:12000], **o)
    model = json.load(open(os.path.join(B, "models.json")))["headline_reviewer"][0]
    for _ in range(2):
        m, rec = llm.chat("judge", model, [{"role": "user", "content": msg}], max_tokens=6000, temperature=0, timeout=180)
        j = first_json(m.content)
        if isinstance(j, dict):
            lab = {"m": bool(j.get("mechanism")), "t": bool(j.get("trigger"))} if o["which"] == "intro" \
                else {"f": bool(j.get("false_alarm"))}
            return {"key": o["key"], **lab, "why": str(j.get("why", ""))[:300], "by": "llm", "call": rec}
    return None


def llm_judge():
    done = load(LLM)
    todo = [o for o in items() if o["key"] not in done]
    with ThreadPoolExecutor(8) as ex, open(LLM, "a") as fh:
        for r in ex.map(judge_one, todo):
            if r:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
    print(f"llm labels: {len(load(LLM))} of {len(items())} items")


def agreement(hum, llm_):
    both = [k for k in hum if k in llm_]
    agree = sum(all(hum[k].get(x) == llm_[k].get(x) for x in ("m", "t", "f") if x in hum[k]) for k in both)
    return {"n_overlap": len(both), "agree": agree}


def score():
    hum, llm_ = load(HUMAN), load(LLM)
    rows = []
    for o in outputs():
        lab = hum.get(o["key"]) or llm_.get(o["key"]) or {}
        src = hum[o["key"]].get("by", "human") if o["key"] in hum else "llm" if o["key"] in llm_ else "none"
        if o["which"] == "intro":
            m, t = bool(lab.get("m")) and o["L"], bool(lab.get("t")) and o["L"]
            res = "CAUGHT" if o["L"] and m and o["A"] else "PARTIAL" if o["L"] and (m or t) else "MISSED"
        else:
            m = t = False
            res = "FALSE_ALARM" if o["A"] and lab.get("f") else "CLEAN"
        rows.append({k: o[k] for k in ("case", "which", "split", "model", "prompt", "run", "status", "salvaged", "L", "A")}
                    | {"M": m, "T": t, "label": src, "result": res})
    with open(os.path.join(B, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cells = {}
    for model in sorted({r["model"] for r in rows}):
        for p in ("p0", "p1"):
            per = lambda which, res: [sum(1 for r in rows if r["model"] == model and r["prompt"] == p and r["run"] == n
                                          and r["which"] == which and r["result"] == res) for n in (1, 2, 3)]
            st = lambda v: {"per_run": v, "mean": round(sum(v) / 3, 2), "min": min(v), "max": max(v)}
            case_k = {c: sum(1 for r in rows if r["model"] == model and r["prompt"] == p and r["case"] == c
                             and r["which"] == "intro" and r["result"] == "CAUGHT") for c in H}
            cells[f"{model}/{p}"] = {"caught": st(per("intro", "CAUGHT")), "partial": st(per("intro", "PARTIAL")),
                                     "false_alarms": st(per("fix", "FALSE_ALARM")), "caught_runs_by_case": case_k,
                                     "n_intro": 12, "n_fix": 12, "runs": 3}
    best = max(cells, key=lambda k: (cells[k]["caught"]["mean"], -cells[k]["false_alarms"]["mean"], k.endswith("p1")))
    labels = {s: sum(1 for r in rows if r["label"] == s) for s in ("human", "panel", "llm", "none")}
    out = {"cells": cells, "best_cell": best, "labels": labels, "agreement": agreement(hum, llm_),
           "n_outputs": len(rows), "n_errors": sum(1 for r in rows if r["status"] != "ok"),
           "n_salvaged": sum(1 for r in rows if r["salvaged"]),
           "n_items": len(items()), "cmd": "python bakeoff/score.py"}
    json.dump(out, open(os.path.join(B, "scores.json"), "w"), indent=1)
    for k, v in cells.items():
        print(f"{k}: caught {v['caught']['mean']} of 12 [{v['caught']['min']}, {v['caught']['max']}], "
              f"partial {v['partial']['mean']}, false alarms {v['false_alarms']['mean']} of 12 "
              f"[{v['false_alarms']['min']}, {v['false_alarms']['max']}]")
    print(f"best {best}; labels {labels}; agreement {out['agreement']}")
    return out


if __name__ == "__main__":
    llm_judge() if sys.argv[1:] == ["--llm-judge"] else score()
