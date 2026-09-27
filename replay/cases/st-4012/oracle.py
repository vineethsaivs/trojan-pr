"""Oracle replay for sentence-transformers #4012 (ListMLE exp overflow / underflow).

Runs the ListMLE loss (PListMLELoss with lambda_weight=None, which is exactly what
ListMLELoss passes to super().__init__) from each SHA's source, with the two package
imports stubbed, and checks it against the math, not against either implementation:

  ListMLE = Plackett-Luce NLL:  L(s) = sum_i [ log sum_{j>=i} e^{s_j} - s_i ]
  dL/ds_k = sum_{i<=k} e^{s_k - lse(s[i:])} - 1
  Metamorphic: L(s + c) = L(s) for any constant c (softmax ignores a shared offset).

The reference is pure-Python float64 with the max subtracted before exp.
Every case is one query with no padding, so the separate #3827 padding bug is not hit.
"""
import math, sys, types, torch

# Stub the only package imports the loss files need.
ce = types.ModuleType("sentence_transformers.cross_encoder"); ce.CrossEncoder = object
ut = types.ModuleType("sentence_transformers.util")
ut.batch_to_device = lambda x, d: x
ut.fullname = lambda o: type(o).__name__
sys.modules.update({"sentence_transformers": types.ModuleType("sentence_transformers"),
                    "sentence_transformers.cross_encoder": ce, "sentence_transformers.util": ut})

SHAS = [  # (label, source file fetched with gh api .../contents/PATH?ref=SHA)
    ("introducing 8d0b2d3 (PListMLELoss.py)", "PListMLELoss_intro_8d0b2d3.py"),
    ("fix parent  0e6ef65 (plist_mle.py)   ", "plist_mle_buggy_0e6ef65.py"),
    ("fix         94d68cf (plist_mle.py)   ", "plist_mle_fixed_94d68cf.py"),
]
BASE4 = [3.0, 1.0, 2.0, 0.0]
CASES = [  # (name, scores in input order); controls first
    ("control [1,0]", [1.0, 0.0]),
    ("control base4", BASE4),
    ("[1000,999]", [1000.0, 999.0]),
    ("[-1000,-1001]", [-1000.0, -1001.0]),
    ("base4 + 89", [x + 89 for x in BASE4]),
    ("base4 - 25", [x - 25 for x in BASE4]),
]
EPS32 = 2.0 ** -23


def ref(s):
    """float64 Plackett-Luce NLL and gradient, from the formula above."""
    lse = []
    for i in range(len(s)):
        m = max(s[i:])
        lse.append(m + math.log(math.fsum(math.exp(v - m) for v in s[i:])))
    loss = math.fsum(lse[i] - s[i] for i in range(len(s)))
    grad = [math.fsum(math.exp(s[k] - lse[i]) for i in range(k + 1)) - 1 for k in range(len(s))]
    return loss, grad


def tol(s):
    # Scores are exact in fp32. Each suffix log-normalizer sits near max|s| and is rounded
    # to fp32, so it carries about eps32 * max|s| absolute error (ulp(1000) = 6.1e-5).
    # n suffix terms add up; 8x covers the few ulps from exp/log/subtract. At [1000, 999]
    # this is 1.9e-3; the correct fp32 fix is off by 3e-5 there, the bug by inf or ~2e3.
    return 8 * len(s) * EPS32 * max(1.0, max(abs(v) for v in s))


class ScoreModel(torch.nn.Module):
    """Returns fixed learnable scores. Speaks both CrossEncoder APIs:
    8d0b2d3: model.tokenizer(pairs, ...).to(dev); model(**tokens)[0]
    0e6ef65+: model.preprocess(pairs); model(tokens)["scores"]"""
    num_labels = 1

    def __init__(self, scores):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor(scores, dtype=torch.float32))

    @property
    def device(self):
        return self.scores.device

    def preprocess(self, pairs, **kw):
        return {"indices": torch.tensor([int(d) for _, d in pairs])}

    def tokenizer(self, pairs, **kw):
        class Toks(dict):
            def to(self, dev):
                return self
        return Toks(self.preprocess(pairs))

    def forward(self, inputs=None, indices=None):
        if indices is not None:
            return (self.scores[indices].unsqueeze(-1),)
        return {"scores": self.scores[inputs["indices"]].unsqueeze(-1)}


def run(cls, s):
    model = ScoreModel(s)
    n = len(s)
    inputs = (["q"], [[str(i) for i in range(n)]])
    labels = [torch.arange(n, 0, -1, dtype=torch.float32)]  # descending, matches input order
    loss = cls(model, lambda_weight=None, respect_input_order=True)(inputs, labels)
    loss.backward()
    return loss.item(), model.scores.grad.tolist()


def load(path):
    ns = {}
    exec(compile(open(path).read(), path, "exec"), ns)
    return ns["PListMLELoss"]


if __name__ == "__main__":
    print(f"torch {torch.__version__}, loss in fp32, reference in fp64")
    verdicts = {}
    for label, path in SHAS:
        cls = load(path)
        ok_all = True
        print(f"\n== {label}")
        for name, s in CASES:
            want, gwant = ref(s)
            got, g = run(cls, s)
            t = tol(s)
            finite = math.isfinite(got) and all(math.isfinite(x) for x in g)
            gerr = max(abs(a - b) for a, b in zip(g, gwant)) if finite else float("inf")
            ok = finite and abs(got - want) <= t and gerr <= t
            ok_all &= ok
            if name.startswith("control"):
                assert ok, f"false alarm on normal scores: {label} {name}"
            print(f"  {name:15s} loss={got:12.6f} ref={want:9.6f} |dL|={abs(got - want):9.2e} "
                  f"max|dgrad|={gerr:8.2e} tol={t:7.1e} {'PASS' if ok else 'FAIL'}")
        verdicts[path] = ok_all
        print(f"  VERDICT {'PASS' if ok_all else 'FAIL'}")

    # Self-check: fails on introducing and fix-parent, passes on fix.
    assert not verdicts[SHAS[0][1]] and not verdicts[SHAS[1][1]] and verdicts[SHAS[2][1]]
    print("\nself-check OK: FAIL on 8d0b2d3 and 0e6ef65, PASS on 94d68cf, no control false alarms")
    print("pre-introduction bd86a28: not applicable, ListMLE/PListMLE files do not exist there")
