import torch
import torch.nn as nn
import torch.nn.functional as F

from data import VOCAB


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E = cfg.n_embd
        self.ln1 = nn.LayerNorm(E)
        self.attn = nn.MultiheadAttention(E, cfg.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(nn.Linear(E, 4 * E), nn.GELU(), nn.Linear(4 * E, E))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.wte = nn.Embedding(len(VOCAB), cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, len(VOCAB), bias=False)

    def forward(self, X, Y=None):
        T = X.size(1)
        x = self.wte(X) + self.wpe(torch.arange(T))
        mask = torch.ones(T, T, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.ln_f(x))
        if Y is None:
            return logits, None
        V = logits.size(-1)
        loss = F.cross_entropy(logits.view(-1, V), Y.view(-1), ignore_index=-1)
        return logits, loss


def configure_optimizers(model, cfg):
    """AdamW: weight decay on matrices and embeddings (dim >= 2), none on biases and norms."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.dim() < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr)
