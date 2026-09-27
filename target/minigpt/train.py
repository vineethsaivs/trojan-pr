import torch

from config import Config
from data import get_batch
from model import GPT, configure_optimizers


def clip_grads(params, max_norm, norm_type):
    """Scale grads in place so their global norm_type-norm is at most max_norm.

    Returns the global norm before clipping, as a float.
    """
    grads = [p.grad for p in params if p.grad is not None]
    total = 0.0
    for g in grads:
        total += g.abs().pow(norm_type).sum().item()
    total = total ** (1.0 / norm_type)
    if total > max_norm:
        for g in grads:
            g.mul_(max_norm / (total + 1e-6))
    return total


def train_step(model, opt, batches, cfg):
    """One optimizer step over a list of (X, Y) micro-batches (grad accumulation)."""
    for X, Y in batches:
        loss = model(X, Y)[1] / len(batches)
        loss.backward()
    clip_grads(model.parameters(), cfg.grad_clip, cfg.clip_norm_type)
    opt.step()
    opt.zero_grad()
    return loss.item() * len(batches)


def main(steps=300):
    cfg = Config()
    rng = torch.Generator().manual_seed(cfg.seed)
    model = GPT(cfg)
    opt = configure_optimizers(model, cfg)
    micro = cfg.batch_size // cfg.grad_accum
    for step in range(1, steps + 1):
        batches = [get_batch(micro, rng) for _ in range(cfg.grad_accum)]
        loss = train_step(model, opt, batches, cfg)
        if step == 1 or step % 50 == 0:
            print(f"step {step} loss {loss:.4f}")


if __name__ == "__main__":
    main()
