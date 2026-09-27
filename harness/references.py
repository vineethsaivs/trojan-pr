"""Fixed float64 references the planner picks by name (PLAN 5.2). Each takes the same positional
args and kwargs as the target call, promoted to float64. Never the code under test."""
import math
import torch
import torch.nn.functional as F


def _d(x):
    return x.detach().double() if isinstance(x, torch.Tensor) else x


def clip_grad_norm(grads, norm_type=2.0, max_norm=1.0, **_):
    """torch.nn.utils.clip_grad_norm_ on float64 copies; returns the total norm."""
    ps = []
    for g in grads:
        p = torch.nn.Parameter(torch.zeros_like(_d(g)))
        p.grad = _d(g).clone()
        ps.append(p)
    return torch.nn.utils.clip_grad_norm_(ps, max_norm=float(max_norm), norm_type=float(norm_type))


def broadcast_numel(a, b, *_, **__):
    return torch.tensor(float(math.prod(torch.broadcast_shapes(a.shape, b.shape))), dtype=torch.float64)


def cross_entropy_ignore_index(logits, labels, *_, ignore_index=-100, **__):
    return F.cross_entropy(_d(logits).reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=ignore_index)


def k3_kl_float64(ref_logp, logp, *_, **__):
    d = _d(ref_logp) - _d(logp)
    return torch.expm1(d) - d


def softplus_float64(x, *_, beta=1.0, **__):
    return F.softplus(_d(x), beta=beta)


def logsumexp_float64(x, *_, dim=-1, **__):
    return torch.logsumexp(_d(x), dim=dim)


# cosine_lr and adamw_step need the minigpt adapters (block G); the call adapter reports ERROR.
REFS = {"broadcast_numel": broadcast_numel, "cross_entropy_ignore_index": cross_entropy_ignore_index,
        "k3_kl_float64": k3_kl_float64, "softplus_float64": softplus_float64,
        "logsumexp_float64": logsumexp_float64}
