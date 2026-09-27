from dataclasses import dataclass


@dataclass
class Config:
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 16
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 3e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    clip_norm_type: float = 2.0
    seed: int = 0
