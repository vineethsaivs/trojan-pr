# Oracle for unslothai/unsloth#11470: embedding_learning_rate silently zeroed weight decay.
#
# Math restated (not copied from unsloth or transformers):
#   AdamW, fresh state, gradient exactly 0 -> the Adam term is 0/(sqrt(0)+eps) = 0, so one step
#   moves a parameter only by decoupled decay:  p_new = p * (1 - lr_g * wd_g).
#   wd_g = args.weight_decay for matrices (Linear weight, Embedding weight), 0 for biases and
#   norm gains (the standard AdamW convention). lr_g = embedding_learning_rate for the
#   modules_to_save embedding, learning_rate for everything else.
# Tolerance: rel 1e-6. The smallest expected shift is EMB_LR*WD = 1e-5, 10x the tolerance and
#   ~80x fp32 eps (1.19e-7); the largest is 1e-4. One fp32 multiply errs by <= 1 ulp (~6e-8 at 1.0).
#
# Harness: for each SHA, `git show SHA:unsloth/trainer.py`, ast-extract the real
# UnslothTrainer.create_optimizer plus the module-level helpers it calls, and drive it with a fake
# self holding real TrainingArguments. SFTTrainer is bound to transformers.Trainer (it only uses
# the inherited get_optimizer_cls_and_kwargs). No unsloth import, no GPU.
import ast, functools, subprocess, sys, types, weakref
import torch, torch.nn as nn
from transformers import Trainer, TrainingArguments

REPO = "repo"  # blobless clone of unslothai/unsloth (read-only use)
HELPERS = {"_create_unsloth_optimizer", "_LEGACY_ROLE_ORDER", "_migrate_legacy_optimizer_state",
           "_unsloth_base_optimizer", "_install_legacy_scheduler_resume", "_install_legacy_resume"}
LR, EMB_LR, WD, RTOL = 1e-3, 1e-4, 0.1, 1e-6
EMB = "embed.modules_to_save.default.weight"
DECAY = {"proj.weight", EMB}  # matrices decay; proj.bias, norm.weight, norm.bias do not


class Tiny(nn.Module):  # names mimic a PEFT model whose embedding is in modules_to_save
    def __init__(s):
        super().__init__()
        s.proj, s.norm = nn.Linear(4, 4), nn.LayerNorm(4)
        s.embed = nn.ModuleDict({"modules_to_save": nn.ModuleDict({"default": nn.Embedding(8, 4)})})


def make_args(emb_lr):
    a = TrainingArguments(output_dir="hf_out", learning_rate=LR, weight_decay=WD,
                          optim="adamw_torch", report_to=[], use_cpu=True)
    a.embedding_learning_rate = emb_lr
    return a


def load_create_optimizer(sha):
    src = subprocess.check_output(["git", "-C", REPO, "show", f"{sha}:unsloth/trainer.py"], text=True)
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in HELPERS)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in HELPERS for t in n.targets))]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnslothTrainer")
    keep.append(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "create_optimizer"))
    ns = dict(torch=torch, types=types, functools=functools, weakref=weakref, wraps=functools.wraps,
              SFTTrainer=Trainer, __builtins__=__builtins__)
    exec(compile(ast.Module(keep, []), f"{sha}:unsloth/trainer.py", "exec"), ns)
    return ns["create_optimizer"]


def check(label, build_optimizer, emb_lr):
    torch.manual_seed(0); m = Tiny()
    for p in m.parameters(): nn.init.constant_(p, 1.0)
    opt = build_optimizer(m, make_args(emb_lr))
    for p in m.parameters(): p.grad = torch.zeros_like(p)
    opt.step()
    bad = 0
    print(label)
    for n, p in m.named_parameters():
        lr_g = emb_lr if (n == EMB and emb_lr is not None) else LR
        want = 1.0 - lr_g * (WD if n in DECAY else 0.0)
        got = p.detach().double()
        ok = bool(((got - want).abs() <= RTOL * abs(want)).all())
        bad += not ok
        print(f"  {n:38s} got {got.mean().item():.8f} want {want:.8f} {'ok' if ok else 'FAIL'}")
    print("  param_group weight_decay:", [g["weight_decay"] for g in opt.param_groups])
    print("  VERDICT:", "FAIL" if bad else "PASS")
    return bad


def unsloth_path(sha):
    f = load_create_optimizer(sha)
    def build(m, args):
        self = types.SimpleNamespace(args=args, model=m, optimizer=None,
                                     get_decay_parameter_names=lambda mm: Trainer.get_decay_parameter_names(None, mm))
        return f(self)
    return build


def stock_hf(m, args):  # control: the path taken when embedding_learning_rate is None
    return Trainer(model=m, args=args).create_optimizer()


if __name__ == "__main__":
    print("transformers optimizer_kwargs for adamw_torch:",
          sorted(Trainer.get_optimizer_cls_and_kwargs(make_args(None))[1]))
    print("pre-intro dde6a0f0: N/A, unsloth/trainer.py and embedding_learning_rate do not exist yet")
    r = {
        "intro": check("intro 72b19da6 (PR #506)", unsloth_path("72b19da6"), EMB_LR),
        "parent": check("fix parent bbfabb2e", unsloth_path("bbfabb2e"), EMB_LR),
        "fix": check("fix ed004ac7 (PR #11470)", unsloth_path("ed004ac7"), EMB_LR),
        "stock": check("control: stock HF Trainer.create_optimizer (no embedding lr)", stock_hf, None),
    }
    expected = {"intro": 1, "parent": 1, "fix": 0, "stock": 0}  # 1 = must fail
    ok = all(bool(r[k]) == bool(v) for k, v in expected.items())
    print("ORACLE SEPARATES BUGGY FROM FIXED:", ok)
    sys.exit(0 if ok else 1)
