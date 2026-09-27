"""Static pre-pass: diff -> touched functions + taxonomy pattern hits -> candidate families.
Deterministic, no LLM. Its output bounds what the planner may choose and fixes the
mandatory checks, so the planner can add evidence but never remove it."""
import ast, re, sys

# patterns.md grep heuristics, applied to ADDED and REMOVED lines of the hunk
PATTERNS = {
    "zero_denominator":   (r"/\s*\(|/\s*len\(|/\s*\(?\s*\w+\s*-\s*\w+|\.mean\(\)", ["edge_sweep", "finite_extremes"]),
    "power_root_order":   (r"\*\*\s*2|\.pow\(2\)|\.square\(\)|norm_type|1\.?0?\s*/\s*norm_type|\.norm\(", ["decomposition", "reference"]),
    "overflow":           (r"torch\.exp\(|\.exp\(\)|log1p\(|logsumexp|log\(1\s*\+", ["finite_extremes", "dtype_shadow"]),
    "narrowing_cast":     (r"\bint\(|\.half\(\)|float16|bfloat16|\.long\(\)|//|\.to\(\s*(torch\.)?\w*(16|8)", ["dtype_shadow"]),
    "sentinel_survives":  (r"float\(['\"]inf|math\.inf|=\s*-1\b|= None\b", ["edge_sweep"]),
    "one_sided_clamp":    (r"\bmin\(|\bmax\(|clamp\(|math\.floor|%\s*\w+", ["bounds"]),
    "shape_from_index0":  (r"shape\[0\]|size\(0\)|\.shape\[i\]|zip\(.*shape", ["reference"]),
    "in_place_mutation":  (r"\.sort\(\)|\w_\(|\.update\(|\[:\]\s*=|del \w+\[", ["no_mutation"]),
    "empty_reduction":    (r"masked_fill|torch\.where|\.min\(|\.max\(|topk\(|ignore_index", ["edge_sweep", "reference"]),
    "flag_before_assign": (r"self\.\w*(precision|dtype)\w*", ["config_audit"]),
    "grad_accum":         (r"accum|gradient_accumulation|num_items_in_batch|micro_?batch", ["ga_invariance"]),
    "param_groups":       (r"param_groups|weight_decay|no_decay|optim_groups", ["config_audit"]),
    "lr_schedule":        (r"warmup|cosine|get_lr|lr_decay|min_lr", ["bounds", "monotonic", "edge_sweep"]),
    "defeat_device":      (r"inspect\.stack|sys\._getframe|sys\.modules\[|PYTEST|oracle|sentinel|os\.environ", []),
}


def parse_diff(diff):
    """-> {path: {"new": set(lines), "old": set(lines), "added": [...], "removed": [...]}}"""
    files, cur, o, n = {}, None, 0, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            p = line[4:].removeprefix("b/")
            cur = files.setdefault(p, {"new": set(), "old": set(), "added": [], "removed": []}) if p != "/dev/null" else None
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line)
            o, n = int(m[1]), int(m[2])
        elif cur is not None and line[:1] in "+- " and not line.startswith(("---", "+++")):
            if line[:1] == "+":
                cur["new"].add(n); cur["added"].append((n, line[1:])); n += 1
            elif line[:1] == "-":
                cur["old"].add(o); cur["removed"].append((o, line[1:])); o += 1
            else:
                o += 1; n += 1
    return files


def touched_symbols(src, lines):
    """Qualified names of the innermost functions containing any changed line."""
    out = set()
    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}{ch.name}"
                if any(ch.lineno <= l <= ch.end_lineno for l in lines):
                    inner = len(out)
                    walk(ch, q + ".")
                    if len(out) == inner and not isinstance(ch, ast.ClassDef):
                        out.add(q)
            else:
                walk(ch, prefix)
    walk(ast.parse(src), "")
    return sorted(out)


def prepass(diff, head_src=None, base_src=None, surface=None):
    res = []
    for path, h in parse_diff(diff).items():
        if not path.endswith(".py"):
            res.append({"file": path, "symbols": [], "patterns": [], "families": [], "note": "non-python"})
            continue
        syms = set()
        if head_src and path in head_src:
            syms |= set(touched_symbols(head_src[path], h["new"]))
        if base_src and path in base_src:
            syms |= set(touched_symbols(base_src[path], h["old"]))
        hits, fams = [], set()
        for pid, (rx, fam) in PATTERNS.items():
            for ln, text in h["added"] + h["removed"]:
                if re.search(rx, text):
                    hits.append({"pattern": pid, "line": ln, "text": text.strip()[:100]})
                    fams |= set(fam)
                    break
        for s in syms:
            fams |= set((surface or {}).get(f"{path}:{s}", []))
        res.append({"file": path, "symbols": sorted(syms), "patterns": hits, "families": sorted(fams)})
    return res


if __name__ == "__main__":
    import json
    diff = open(sys.argv[1]).read()
    head = {sys.argv[2]: open(sys.argv[3]).read()} if len(sys.argv) > 3 else None
    base = {sys.argv[2]: open(sys.argv[4]).read()} if len(sys.argv) > 4 else None
    print(json.dumps(prepass(diff, head, base), indent=1))
