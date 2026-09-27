"""st-3868 oracle: SparseCoSENTLoss must compute the published CoSENT loss.

For each SHA: fetch the real source files from GitHub, exec them with only the
package imports stubbed (torch is real), build the loss with its OWN default
arguments, and compare against CoSENT restated from the paper
(Su 2022, https://kexue.fm/archives/8847), not from either implementation.

Run: ../../venv/bin/python oracle.py
"""
from __future__ import annotations

import ast
import math
import pathlib
import subprocess
import sys
import types
import warnings

import numpy as np
import torch

REPO = "huggingface/sentence-transformers"
HERE = pathlib.Path(__file__).parent
SHAS = [
    # (label, sha, expect, sparse loss path, dense CoSENT path, util sources)
    ("pre-intro  ddcb056 (main before #3401 merge)", "ddcb0569b78a24b823803543082663132bb7b25a", "PASS",
     None, "sentence_transformers/losses/CoSENTLoss.py", ["sentence_transformers/util.py"]),
    ("introduced 14afc4b (#3401 merge)", "14afc4b6681f0b83bded05fe91a8fd3320d453f9", "FAIL",
     "sentence_transformers/sparse_encoder/losses/SparseCoSENTLoss.py", "sentence_transformers/losses/CoSENTLoss.py",
     ["sentence_transformers/util.py"]),
    ("fix-parent 756a160 (last buggy main)", "756a16019a1206c5687ee43edc9cd190b2d1f193", "FAIL",
     "sentence_transformers/sparse_encoder/losses/sparse_cosent.py",
     "sentence_transformers/sentence_transformer/losses/cosent.py",
     ["sentence_transformers/util/tensor.py", "sentence_transformers/util/similarity.py"]),
    ("fixed      4713cf1 (#3868)", "4713cf11b0dce44a16d437ea0e76fc80c6791425", "PASS",
     "sentence_transformers/sparse_encoder/losses/sparse_cosent.py",
     "sentence_transformers/sentence_transformer/losses/cosent.py",
     ["sentence_transformers/util/tensor.py", "sentence_transformers/util/similarity.py"]),
]

# Tolerance from arithmetic: fp32 cosines carry ~1e-7 relative error; scale=20
# amplifies score differences to ~2e-6 absolute inside logsumexp over <= 28 terms,
# so a correct fp32 loss lands within ~1e-5 of the float64 oracle. atol=rtol=1e-4
# leaves 10x headroom. The bug moves the loss by >= 1 (20.0 on the minimal case).
ATOL = RTOL = 1e-4
SCALE = 20.0
warnings.filterwarnings("ignore", message="Sparse")  # torch beta-sparse notices


def fetch(sha: str, path: str) -> str:
    cache = HERE / "src" / sha[:7] / path
    if not cache.exists():
        out = subprocess.run(["gh", "api", f"repos/{REPO}/contents/{path}?ref={sha}", "-H",
                              "Accept: application/vnd.github.raw"], capture_output=True, text=True, check=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(out.stdout)
    return cache.read_text()


def build_util(sha: str, paths: list[str]) -> types.ModuleType:
    """Every top-level undecorated def from the real util sources, verbatim."""
    mod = types.ModuleType("sentence_transformers.util")
    mod.__dict__.update(torch=torch, np=np, Tensor=torch.Tensor, logger=types.SimpleNamespace(warning=print))
    exec("from __future__ import annotations", mod.__dict__)
    for p in paths:
        src = fetch(sha, p)
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and not node.decorator_list:
                exec(compile("from __future__ import annotations\n" + ast.get_source_segment(src, node), p, "exec"),
                     mod.__dict__)
    return mod


class _Stub(types.ModuleType):
    def __getattr__(self, name):  # SentenceTransformer, SparseEncoder, ... are only type hints here
        return type(name, (), {})


def load_losses(sha: str, sparse_path: str | None, dense_path: str, util_paths: list[str]):
    for k in [k for k in sys.modules if k.startswith("sentence_transformers")]:
        del sys.modules[k]
    util = build_util(sha, util_paths)

    def mod(name):
        m = _Stub(name)
        m.__path__ = []
        sys.modules[name] = m
        return m

    for name in ["sentence_transformers", "sentence_transformers.SentenceTransformer",
                 "sentence_transformers.sentence_transformer", "sentence_transformers.sentence_transformer.model",
                 "sentence_transformers.sentence_transformer.losses", "sentence_transformers.losses",
                 "sentence_transformers.sparse_encoder", "sentence_transformers.sparse_encoder.SparseEncoder",
                 "sentence_transformers.sparse_encoder.model"]:
        mod(name)
    sys.modules["sentence_transformers"].util = util
    sys.modules["sentence_transformers.util"] = util

    dense_mod = mod(dense_path[:-3].replace("/", "."))
    exec(compile(fetch(sha, dense_path), dense_path, "exec"), dense_mod.__dict__)
    sparse_cls = None
    if sparse_path:
        sparse_mod = mod(sparse_path[:-3].replace("/", "."))
        exec(compile(fetch(sha, sparse_path), sparse_path, "exec"), sparse_mod.__dict__)
        sparse_cls = sparse_mod.SparseCoSENTLoss
    return sparse_cls, dense_mod.CoSENTLoss


def oracle(u: torch.Tensor, v: torch.Tensor, y: torch.Tensor, scale: float = SCALE) -> float:
    """CoSENT, restated: s_k = cos(u_k, v_k), one cosine per pair.
    L = log(1 + sum_{(i,j): y_i < y_j} exp(scale * (s_i - s_j))), in float64."""
    u = (u.to_dense() if u.is_sparse else u).double()
    v = (v.to_dense() if v.is_sparse else v).double()
    s = [float(u[k] @ v[k]) / (float(u[k].norm()) * float(v[k].norm())) for k in range(len(y))]
    terms = [0.0] + [scale * (s[i] - s[j]) for i in range(len(y)) for j in range(len(y)) if y[i] < y[j]]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def cases():
    # 1) hand-checkable: perfectly ranked 2-pair batch, expected log(1 + e^-20) ~= 2.06e-9
    yield "minimal B=2", torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]), torch.tensor([[1.0, 0, 0], [0, 0, 1.0]]), \
        torch.tensor([1.0, 0.0])
    # 2) random non-negative (SPLADE-like) batches
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        yield f"random seed={seed} B=8", torch.randn(8, 32, generator=g).relu(), \
            torch.randn(8, 32, generator=g).relu(), torch.rand(8, generator=g)


def metamorphic(loss_fn) -> tuple[float, float]:
    """Rotating both vectors of pair 0 by the same orthogonal Q keeps every own-pair
    cosine, so CoSENT must not change."""
    g = torch.Generator().manual_seed(0)
    u, v, y = torch.randn(6, 16, generator=g).relu(), torch.randn(6, 16, generator=g).relu(), torch.rand(6, generator=g)
    Q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g))
    u2, v2 = u.clone(), v.clone()
    u2[0], v2[0] = Q @ u[0], Q @ v[0]
    return loss_fn(u, v, y), loss_fn(u2, v2, y)


def check(name, obj, sparse_inputs: bool) -> bool:
    def loss_fn(u, v, y):
        if sparse_inputs:
            u, v = u.to_sparse(), v.to_sparse()
        if hasattr(obj, "compute_loss_from_embeddings"):
            return float(obj.compute_loss_from_embeddings([u, v], y))
        # pre-#3401 CoSENTLoss only has forward(); feed embeddings through a passthrough model
        obj.model = lambda feats: {"sentence_embedding": feats["emb"]}
        return float(obj([{"emb": u}, {"emb": v}], y))

    worst, first = 0.0, None
    for label, u, v, y in cases():
        got, want = loss_fn(u, v, y), oracle(u, v, y)
        err = abs(got - want)
        if first is None:
            first = (label, got, want)
        worst = max(worst, err / (ATOL + RTOL * abs(want)))
    a, b = metamorphic(loss_fn)
    meta_ok = math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    ok = worst <= 1.0 and meta_ok
    print(f"    {name}: default similarity_fct={obj.similarity_fct.__name__}")
    print(f"      {first[0]}: loss={first[1]:.6g} oracle={first[2]:.6g}")
    print(f"      21 batches: worst |loss-oracle| / tol = {worst:.3g}   (<= 1 passes)")
    print(f"      metamorphic rotate-pair-0: {a:.6f} -> {b:.6f} ({'invariant' if meta_ok else 'CHANGED'})")
    print(f"      => {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    mismatches = 0
    for label, sha, expect, sparse_path, dense_path, util_paths in SHAS:
        print(f"[{label}] expect {expect}")
        sparse_cls, dense_cls = load_losses(sha, sparse_path, dense_path, util_paths)
        model = types.SimpleNamespace()  # no weights: the loss only needs embeddings
        if sparse_cls is None:
            print("    SparseCoSENTLoss: absent at this SHA (new file in #3401); sibling control below")
            ok = check("CoSENTLoss (dense parent class)", dense_cls(model), sparse_inputs=False)
        else:
            ok = check("SparseCoSENTLoss", sparse_cls(model), sparse_inputs=True)
        got = "PASS" if ok else "FAIL"
        mismatches += got != expect
        print(f"    verdict {got}, expected {expect}: {'as expected' if got == expect else 'UNEXPECTED'}\n")
    print("ALL AS EXPECTED" if mismatches == 0 else f"{mismatches} UNEXPECTED")
    return int(mismatches != 0)


if __name__ == "__main__":
    sys.exit(main())
