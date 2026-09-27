"""Single-rank CPU shims for repo internals that function-extract must not stub blindly.
Each shim states world_size=1 semantics. Oracles that need >1 rank are out of scope."""
import logging, types
import torch
import torch.nn.functional as F

_log = logging.getLogger("replay")


class _CpuAccel:
    def current_device_name(self): return "cpu"
    def device_name(self, i=None): return "cpu"
    def FloatTensor(self, x): return torch.tensor(x, dtype=torch.float32)
    def is_available(self): return True
    def is_bf16_supported(self): return True
    def is_fp16_supported(self): return True
    def synchronize(self, *a): return None
    def communication_backend_name(self): return "gloo"
    def __getattr__(self, name):          # anything else (Event, memory_reserved...) is a stub
        from extract import _stub_class
        return _stub_class(name)


_acc = _CpuAccel()
_reduce = types.SimpleNamespace(SUM="sum", MAX="max", MIN="min", AVG="avg")

DEEPSPEED = {
    "deepspeed.accelerator": {"get_accelerator": lambda: _acc},
    "deepspeed.comm": {"get_world_size": lambda group=None: 1, "get_rank": lambda group=None: 0,
                       "all_reduce": lambda t, op=None, group=None, async_op=False: None,
                       "is_initialized": lambda: False, "ReduceOp": _reduce, "barrier": lambda *a, **k: None},
    "deepspeed.utils": {"logger": _log, "groups": types.SimpleNamespace(
        _get_data_parallel_group=lambda: None, _get_model_parallel_group=lambda: None)},
    "deepspeed.utils.logging": {"logger": _log},
}


def _cos_sim(a, b):
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T


SENTENCE_TRANSFORMERS = {
    "sentence_transformers.util": {"cos_sim": _cos_sim, "fullname": lambda o: type(o).__name__,
                                   "batch_to_device": lambda b, d: b},
}

RAY = {}
UNSLOTH = {}

SHIMS = {"deepspeed": DEEPSPEED, "sentence_transformers": SENTENCE_TRANSFORMERS,
         "ray": RAY, "unsloth": UNSLOTH, "unsloth_zoo": UNSLOTH}
