import math

import torch

from config import Config
from data import VOCAB, get_batch
from model import GPT, configure_optimizers
from train import clip_grads, train_step


def test_config_defaults():
    cfg = Config()
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size) == (2, 2, 64, 16)
    assert (cfg.batch_size, cfg.grad_accum, cfg.lr, cfg.weight_decay) == (64, 1, 3e-3, 0.1)
    assert (cfg.grad_clip, cfg.clip_norm_type, cfg.seed) == (1.0, 2.0, 0)


def test_shapes():
    cfg = Config()
    X, Y = get_batch(4, torch.Generator().manual_seed(0))
    assert X.shape == Y.shape == (4, cfg.block_size)
    assert X.dtype == Y.dtype == torch.long
    assert ((Y != -1).sum(1) == 4).all()
    logits, loss = GPT(cfg)(X)
    assert logits.shape == (4, cfg.block_size, len(VOCAB)) and loss is None


def test_smoke_20_steps():
    cfg = Config()
    rng = torch.Generator().manual_seed(cfg.seed)
    model = GPT(cfg)
    opt = configure_optimizers(model, cfg)
    losses = [train_step(model, opt, [get_batch(cfg.batch_size, rng)], cfg) for _ in range(20)]
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0]


def test_clip_grads_matches_torch():
    torch.manual_seed(0)
    a = [torch.zeros(5, 3, requires_grad=True), torch.zeros(7, requires_grad=True)]
    b = [torch.zeros(5, 3, requires_grad=True), torch.zeros(7, requires_grad=True)]
    for p, q in zip(a, b):
        p.grad = torch.randn(p.shape)
        q.grad = p.grad.clone()
    ours = clip_grads(a, 0.5, 2.0)
    ref = torch.nn.utils.clip_grad_norm_(b, 0.5, 2.0)
    assert math.isclose(ours, ref.item(), rel_tol=1e-5)
    for p, q in zip(a, b):
        torch.testing.assert_close(p.grad, q.grad)
