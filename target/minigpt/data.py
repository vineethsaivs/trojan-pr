"""3-digit addition samples like "123+456=0579" (answer zero-padded to 4 digits)."""
import torch

from config import Config

VOCAB = "0123456789+=."  # "." pads each sample to block_size
PLUS, EQ, PAD = 10, 11, 12


def digits(x, k):
    return torch.stack([x // 10**i % 10 for i in reversed(range(k))], dim=1)


def get_batch(n, rng):
    """Return (X, Y), both LongTensors of shape (n, block_size).

    rng is a torch.Generator. Y is the next token on the 4 answer positions and -1
    everywhere else, so only the answer is trained.
    """
    a = torch.randint(0, 1000, (n,), generator=rng)
    b = torch.randint(0, 1000, (n,), generator=rng)
    plus, eq = torch.full((n, 1), PLUS), torch.full((n, 1), EQ)
    s = torch.cat([digits(a, 3), plus, digits(b, 3), eq, digits(a + b, 4)], dim=1)
    X = torch.full((n, Config.block_size), PAD)
    Y = torch.full((n, Config.block_size), -1)
    X[:, :11] = s[:, :-1]
    Y[:, 7:11] = s[:, 8:]
    return X, Y
