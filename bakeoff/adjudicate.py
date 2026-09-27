"""Blind adjudication (PROTOCOL.md, PLAN 4.8). Run in your own terminal: .venv/bin/python bakeoff/adjudicate.py
Shows L-passing intro outputs and actionable fix outputs, shuffled, model/prompt/run hidden.
Keys: intro m (states the mechanism class), t (names a trigger), n (neither), e.g. "mt".
      fix  f (false alarm: claims a numerical defect you reject), n (not a false alarm).
q quits; rerun to resume. Hard cap 30 minutes in total; leftovers keep the LLM judge's label."""
import json, os, sys, textwrap, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bakeoff.score import HUMAN, items, load  # noqa: E402

CAP = 30 * 60
W = min(os.get_terminal_size().columns if sys.stdout.isatty() else 100, 110)


def show(i, n, o, left):
    print("\n" + "=" * W)
    print(f"item {i}/{n}   time left {int(left // 60)}:{int(left % 60):02d}   {o['case']} ({o['repo']})")
    if o["which"] == "intro":
        print("PR THAT INTRODUCED THE BUG. Rubric:")
        print(f"  function:  {o['function']}\n  mechanism: {o['mechanism_class']}\n  triggers:  {o['trigger_set']}")
    else:
        print(f"MERGED FIX PR (benign control; it fixed: {o['mechanism_class']}). Is any flag below a false alarm?")
    seen = set()
    for f in o["show"]:
        k = str(f.get("mechanism"))[:200]
        if k in seen:
            continue
        seen.add(k)
        print("-" * W + f"\n{f.get('file', '').split('/')[-1]}:{f.get('function')} L{f.get('line_start')}-{f.get('line_end')}"
              f" severity={f.get('severity')}")
        for lab in ("mechanism", "trigger", "consequence"):
            print(textwrap.fill(f"{lab}: {f.get(lab)}", W, initial_indent="  ", subsequent_indent="    "))
        if len(seen) == 6:
            print(f"  ... {len(o['show']) - 6} more findings not shown")
            break


def main():
    done = load(HUMAN)
    used = sum(r.get("secs", 0) for r in done.values())
    todo = [o for o in items() if o["key"] not in done]
    n = len(todo) + len(done)
    with open(HUMAN, "a") as fh:
        for i, o in enumerate(todo, len(done) + 1):
            left = CAP - used
            if left <= 0:
                print("\n30-minute cap reached. Leftovers keep the LLM judge's label.")
                break
            show(i, n, o, left)
            ok = "mtn" if o["which"] == "intro" else "fn"
            t0 = time.time()
            while True:
                a = input(f"[{'/'.join(ok)}, q quit] > ").strip().lower()
                if a == "q" or (a and set(a) <= set(ok) and not ("n" in a and len(a) > 1)):
                    break
            if a == "q":
                break
            secs = round(time.time() - t0, 1)
            used += secs
            lab = {"m": "m" in a, "t": "t" in a} if o["which"] == "intro" else {"f": "f" in a}
            fh.write(json.dumps({"key": o["key"], **lab, "by": "human", "secs": secs}) + "\n")
            fh.flush()
    print(f"labelled {len(load(HUMAN))} of {n} items; used {int(used // 60)} min. Next: python bakeoff/score.py")


if __name__ == "__main__":
    main()
