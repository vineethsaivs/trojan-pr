# Oracle for unsloth-10681: LongRope must use short_factor for every L <= original_max.
#
# Relation (restated from the LongRope / Phi-3 definition, not from unsloth code):
#   f   = long_factor if L > original_max else short_factor
#   w_i = 1 / (f_i * base^(2i/d)),  i in [0, d/2)
#   a   = sqrt(1 + ln(max_pos/original_max) / ln(original_max)) if max_pos > original_max else 1
#   cos[p, j] = a * cos(p * w_(j mod d/2)), sin likewise, for p in [0, L)
# Sources: transformers v4.44.0 phi3 `if seq_len > self.original_max_position_embeddings`,
# microsoft/Phi-3.5-mini-instruct modeling_phi3.py.
#
# Probes: L in {1, orig-1, orig, orig+1}, each on a fresh module (cold) and after a
# 4*orig sequence already ran in the process (warm). Checks forward() and get_cached().
# A raise or None counts as FAIL.
#
# Tolerance: the cache is stored in fp16 (is_bfloat16_supported stubbed False). |value| <= a,
# a = 1.19 (Phi-3.5) or 1.32 (toy); fp16 spacing near 1.2 is 2^-10 ~ 9.8e-4, so storage
# rounding is <= 4.9e-4. Angles p*w are built in fp32 with p <= 4097: fp32 ulp at 4096 is
# 4.9e-4, so angle error ~2.4e-4, times a ~ 3e-4. Total <= ~8e-4; abs tol 4e-3 gives ~5x
# margin. The bug's error (wrong factor table) is O(a), i.e. > 1, so the verdict does not
# depend on the tolerance.
import warnings; warnings.filterwarnings("ignore", category=SyntaxWarning)
import ast, base64, json, math, pathlib, subprocess, types
import torch

HERE = pathlib.Path(__file__).parent
SRC = HERE / "src"; SRC.mkdir(exist_ok=True)
TOL = 4e-3
REPO, PATH = "unslothai/unsloth", "unsloth/models/llama.py"
SHAS = [
    ("pre-intro (parent of #940)", "0927c343920f2e7d394df1b2da14e8ccd4e3328c"),
    ("INTRO #940 fb60340", "fb60340a909c0e08901ab493051cc5e23ba92011"),
    ("tag November-2024", "387dd13e31361c02e7c26a6d0e7636eda453fe32"),
    ("#2919 merge 33b02d4", "33b02d4a80682f3323c6b06542e19b57cb651ec3"),
    ("fix parent 7434717", "7434717466d09c5d936e4348267feca4bb4621d3"),
    ("FIX #10681 9b3b2b1", "9b3b2b1c825f02bf74d566f2e61995aa742ff575"),
]
phi = json.load(open(HERE / "phi35_config.json"))
CONFIGS = {
    "phi3.5": dict(dim=phi["hidden_size"] // phi["num_attention_heads"],
                   orig=phi["original_max_position_embeddings"], maxp=phi["max_position_embeddings"],
                   base=phi.get("rope_theta", 10000.0),
                   short=phi["rope_scaling"]["short_factor"], long=phi["rope_scaling"]["long_factor"]),
    "toy": dict(dim=8, orig=16, maxp=128, base=10000.0, short=[1.0] * 4, long=[2.0] * 4),
}


def fetch(sha):
    f = SRC / f"llama_{sha[:10]}.py"
    if not f.exists():
        b64 = subprocess.check_output(["gh", "api", f"repos/{REPO}/contents/{PATH}?ref={sha}", "--jq", ".content"])
        f.write_bytes(base64.b64decode(b64))
    return f.read_text()


class CPUTorch:  # routes every device the class asks for to CPU
    cuda = types.SimpleNamespace(current_device=lambda: 0, device_count=lambda: 1)
    def __getattr__(self, n): return getattr(torch, n)
    def device(self, spec, index=None):
        if isinstance(spec, int) and not isinstance(spec, bool): return torch.device("cpu", spec)
        return torch.device(spec) if index is None else torch.device(spec, index)
    def empty(self, *a, **k): k["device"] = "cpu"; return torch.empty(*a, **k)


def extract(src):
    tree = ast.parse(src.replace('"cuda:0"', '"cpu"').replace('"cuda"', '"cpu"'))
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "LongRopeRotaryEmbedding"), None)
    if cls is None: return None
    ns = {"torch": CPUTorch(), "math": math, "DEVICE_COUNT": 1, "DEVICE_TYPE_TORCH": "cpu", "DEVICE_TYPE": "cpu",
          "get_current_device": lambda: 0, "is_bfloat16_supported": lambda: False,
          "_get_rope_theta": lambda c, default=10000: default}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "llama.py", "exec"), ns)
    return ns["LongRopeRotaryEmbedding"]


def reference(c, L):
    d = c["dim"]
    f = torch.tensor(c["long"] if L > c["orig"] else c["short"], dtype=torch.float64)
    w = 1.0 / (f * c["base"] ** (2 * torch.arange(d // 2, dtype=torch.float64) / d))
    a = math.sqrt(1 + math.log(c["maxp"] / c["orig"]) / math.log(c["orig"])) if c["maxp"] > c["orig"] else 1.0
    ang = torch.arange(L, dtype=torch.float64)[:, None] * w[torch.arange(d) % (d // 2)][None, :]
    return a * ang.cos(), a * ang.sin()


def probe(Cls, c, L, warm, via):
    R = Cls(dim=c["dim"], max_position_embeddings=c["maxp"], original_max_position_embeddings=c["orig"],
            base=c["base"], short_factor=c["short"], long_factor=c["long"])
    x = types.SimpleNamespace(device=torch.device("cpu", 0), dtype=torch.float16)
    try:
        if warm:
            R.extend_rope_embedding(x, 4 * c["orig"]); R.forward(x, seq_len=4 * c["orig"])
        R.extend_rope_embedding(x, L)
        cos, sin = R.forward(x, seq_len=L) if via == "forward" else R.get_cached(seq_len=L)
        if cos is None: return None, "returned None"
        cos, sin = cos[:L], sin[:L]
    except Exception as e:
        return None, f"raised {type(e).__name__}: {str(e)[:60]}"
    rc, rs = reference(c, L)
    return max((cos.double() - rc).abs().max().item(), (sin.double() - rs).abs().max().item()), ""


if __name__ == "__main__":
    for label, sha in SHAS:
        Cls = extract(fetch(sha))
        print(f"== {label} {sha[:10]}")
        if Cls is None:
            print("   N/A: LongRopeRotaryEmbedding does not exist at this SHA (no pre-intro control possible)")
            continue
        bad_boundary = bad_other = 0
        for name, c in CONFIGS.items():
            o = c["orig"]
            for L in (1, o - 1, o, o + 1):
                for warm in (False, True):
                    for via in ("forward", "get_cached"):
                        err, msg = probe(Cls, c, L, warm, via)
                        ok = err is not None and err <= TOL
                        if not ok: bad_boundary += L == o; bad_other += L != o
                        if not ok or L == o:
                            tag = "orig" if L == o else f"{L}"
                            print(f"   {name:6} L={tag:5} {'warm' if warm else 'cold'} {via:10} "
                                  f"{'PASS' if ok else 'FAIL'} " + (f"max_abs_err={err:.3g}" if err is not None else msg))
        verdict = "FAIL" if bad_boundary or bad_other else "PASS"
        print(f"   VERDICT {verdict}: boundary failures={bad_boundary}, non-boundary failures={bad_other} (tol {TOL})")
