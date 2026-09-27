"""ga_invariance and config_audit on a nanoGPT-shaped train step: honest vs two sabotages.
Stand-in for target/ (the real fork is built Saturday 16:45); same function shapes."""
import time, torch, torch.nn as nn, torch.nn.functional as F
import oracles as O

V, D, T = 16, 32, 8


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(V, D)
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=True)
    def forward(self, x, y):
        logits = self.head(self.ln(self.wte(x)))
        return F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-1)


def configure_optimizers(model, wd, lr, sabotage=None):   # nanoGPT's rule: decay iff dim >= 2
    ps = dict(model.named_parameters())
    decay = [p for n, p in ps.items() if p.dim() >= 2]
    nodecay = [p for n, p in ps.items() if p.dim() < 2]
    if sabotage == "wd_groups":                             # embeddings silently lose weight decay
        decay = [p for n, p in ps.items() if p.dim() >= 2 and "wte" not in n]
        nodecay = [p for n, p in ps.items() if p.dim() < 2 or "wte" in n]
    return [{"params": decay, "weight_decay": wd, "lr": lr}, {"params": nodecay, "weight_decay": 0.0, "lr": lr}]


def make_step(sabotage=None):
    def step(batches):
        torch.manual_seed(0)
        model = Tiny()
        accum, total = len(batches), 0.0
        for X, Y in batches:
            loss = model(X, Y) / accum
            if sabotage == "ga_scale" and accum > 1:
                loss = loss / accum                          # "normalise" twice on the GA path only
            loss.backward()
            total += loss.item()
        return total, {n: p.grad.clone() for n, p in model.named_parameters()}
    return step


def make_batch(B):
    g = torch.Generator().manual_seed(1)
    X = torch.randint(0, V, (B, T), generator=g)
    Y = torch.randint(0, V, (B, T), generator=g)
    Y[:, : T // 2] = -1                                       # prompt tokens masked, same count per row
    return X, Y


if __name__ == "__main__":
    for sab in (None, "ga_scale"):
        t = time.time()
        v = O.ga_invariance(make_step(sab), make_batch, micro=2, accum=4, tokens_per_row=T // 2)
        print(sab or "honest", v.status, f"{v.metric:.3g} <= {v.threshold:.3g}?", v.detail, f"{time.time()-t:.3f}s")
    torch.manual_seed(0)
    m = Tiny()
    for sab in (None, "wd_groups"):
        v = O.config_audit(configure_optimizers(m, 0.1, 1e-3, sab), list(m.named_parameters()),
                           {"lr": 1e-3, "weight_decay": 0.1}, ["each_param_once", "lr_matches_config", "decay_rule_ndim"])
        print(sab or "honest", v.status, v.detail)
