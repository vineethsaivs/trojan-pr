"""Generated data for the UI and README. `python -m plumbline.numbers cases` (laptop: needs the private
research/cases_facts.json) writes results/cases.json from whitelisted public facts."""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = ["id", "tier", "repo", "file", "function", "bug_plain", "pattern", "intro", "pre", "fixp", "fix",
          "review_on_intro", "ci_at_merge", "days_alive", "releases", "sibling", "tests_note", "oracle",
          "minimal_input", "two_numbers", "replay", "static", "stage_safe_claim", "caveats", "downstream"]
SPLIT = {"dev": ["ds-8313", "ds-8533", "st-3921", "ds-8334", "ray-65747", "st-4019"],
         "held-out": ["st-3868", "ray-65535", "ray-65790", "zoo-1286", "unsloth-11337", "unsloth-11470"],
         "index": ["unsloth-10842", "zoo-1369", "unsloth-10681", "zoo-1240", "st-4012", "ds-8324", "ds-8199", "ds-8179", "st-4020"],
         "dropped": ["ds-8268", "ds-8227"]}


def headline(two):
    """'p=1: returns 53.0, must be 9.0 (...)' -> {'context': 'p=1', 'got': '53', 'must': '9'}."""
    m = re.match(r"\s*(?P<ctx>[^:]*):\s*(?:\w+\s*)?(?:returns|=|gives)?\s*(?P<got>[^,]+?),\s*must be\s*(?P<must>[^(;,]+)", two or "")
    if not m:   # "4 x 1.2 gives 4, must be 5"
        m = re.match(r"\s*(?P<ctx>.+?)\s+gives\s+(?P<got>[^,]+?),\s*must be\s*(?P<must>[^(;,]+)", two or "")
    if not m:
        return None
    t = lambda s: re.sub(r"\.0\b", "", s.strip())
    return {"context": m["ctx"].strip(), "got": t(m["got"].split("=")[-1]), "must": t(m["must"])}


def cases():
    facts = {c["id"]: c for c in json.load(open(os.path.join(ROOT, "research", "cases_facts.json")))["cases"]}
    out = []
    for split, ids in SPLIT.items():
        for cid in ids:
            c = {k: facts[cid][k] for k in PUBLIC if k in facts[cid]}
            c["split"], c["headline"] = split, headline(c.get("two_numbers"))
            out.append(c)
    missing = set(facts) - {c["id"] for c in out}
    assert not missing, f"cases not placed in a split: {missing}"
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "cases.json"), "w"), indent=1)
    return out


if __name__ == "__main__" and sys.argv[1:] == ["cases"]:
    cs = cases()
    print(len(cs), "cases;", sum(1 for c in cs if c["headline"]), "with a parsed headline")
    for c in cs[:6] + cs[6:12]:
        print(f"  {c['id']:14} {c['split']:9} {c['headline']}")
