"""Gold plans: the hand-written check set per corpus case, copied verbatim from Saturday's
proto/run_cases.py. Runs inside gVisor only. Selected by case id; nothing from a job is evaluated."""
import math, random
import torch
import oracles as O

def ds_8142(m):
    def sched(total, warm, cmin=0.1):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        return m.WarmupCosineLR(opt, total_num_steps=total, warmup_num_steps=warm, cos_min_ratio=cmin)
    def ratio_at(s, step):
        s.last_batch_iteration = step
        return s.get_lr_ratio()
    s1 = sched(100, 10)
    s2 = sched(100, 100)
    return [
        O.monotonic(lambda k: ratio_at(s1, k), list(range(10, 300)), "nonincreasing", atol=1e-12),
        O.edge_sweep(lambda: [ratio_at(s2, k) for k in range(0, 150)], {"warmup_eq_total": ()}),
        O.bounds(lambda k: ratio_at(s1, k), list(range(0, 300)), lo=0.0, hi=1.0),
    ]


def ds_8199(m):
    return [O.no_mutation(m.trim_mean, lambda: ([3.0, 1.0, 2.0, 5.0, 4.0], 0.2))]


def ds_8313(m):
    def total_norm(p):
        def fn(tensors):
            params = []
            for t in tensors:
                q = torch.nn.Parameter(torch.zeros_like(t)); q.grad = t.clone(); params.append(q)
            return m.clip_grad_norm_(params, max_norm=1e9, norm_type=p)
        return fn
    g = torch.tensor([3.0, -4.0, 2.0, 1.0, -0.5, 7.0])
    out = [O.decomposition(total_norm(p), g, pieces=3, combine="same", p=p) for p in (2, 1, 3)]
    for v, p in zip(out, (2, 1, 3)):
        v.detail = f"p={p}: " + v.detail
    return out


def ds_8324(m):
    shapes = [((2, 3, 4), (4,)), ((2, 3, 4), (3, 1)), ((4,), (2, 3, 4)), ((5, 5), (5, 5))]
    ref = lambda a, b: torch.tensor(float(math.prod(torch.broadcast_shapes(a.shape, b.shape))))
    fn = lambda a, b: torch.tensor(float(m._elementwise_flops_compute(a, b)[0]))
    mk = lambda a, b: (torch.zeros(a), torch.zeros(b))
    return [O.reference(fn, ref, mk, [{"a": a, "b": b} for a, b in shapes])]


def ds_8334(m):
    def sched(mn, step, kind="fixed_linear"):
        cfg = {"min_difficulty": mn, "max_difficulty": 1024, "schedule_type": kind,
               "schedule_config": {"total_curriculum_step": 100, "difficulty_step": step, "root_degree": 2}}
        return m.CurriculumScheduler(cfg)
    a, b = sched(8, 16), sched(64, 16)
    return [O.bounds(a.get_difficulty, list(range(0, 150)), lo=8, hi=1024),
            O.bounds(b.get_difficulty, list(range(0, 150)), lo=64, hi=1024)]


def ds_8533(m):
    torch.manual_seed(0)
    G = torch.randn(8, 32)
    fn = lambda g: m.zeropower_via_gram_newtonschulz(g, 5)
    # fp16 iteration: output is an approximate polar factor; allow fp16-level drift
    return [O.finite_extremes(fn, lambda s: (G * s,), [1.0, 1e3, 1e5], expect="scale_invariant", rtol=5e-2)]


def ray_65747(m):
    def fn(config):
        random.seed(0)
        new, _ = m._explore(config, {"batch_size": lambda: 4}, 0.0, (1.2,), None)
        return float(new["batch_size"])
    return [O.dtype_shadow(fn, lambda dt: ({"batch_size": 4.0 if dt == "wide" else 4},), target_dtype="int")]


def st_3921(m):
    loss = m.GlobalOrthogonalRegularizationLoss(model=None)
    torch.manual_seed(0)
    return [O.edge_sweep(lambda e: loss.compute_gor(e), {"batch_1": (torch.randn(1, 8),), "batch_4": (torch.randn(4, 8),)})]


def st_4012(m):
    class Fake:
        device, num_labels = "cpu", 1
        def __init__(self): self.logits = None
        def preprocess(self, pairs, **k): return {}
        def __call__(self, tokens): return {"scores": self.logits}
    fake = Fake()
    loss = m.PListMLELoss(model=fake, mini_batch_size=4)
    base = torch.tensor([2.0, 1.0, 0.5, -1.0])
    def run(scale):
        fake.logits = base * scale
        return loss([["q"], [["d1", "d2", "d3", "d4"]]], [torch.tensor([3.0, 2.0, 1.0, 0.0])])
    return [O.finite_extremes(run, lambda s: (s,), [1.0, 10.0, 60.0, 100.0], expect="finite")]


def st_4019(m):
    loss = m.BatchHardSoftMarginTripletLoss(model=None)
    labels = torch.tensor([0, 0, 1, 1])
    def mk(s):
        e = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.01], [1.0, 0.01]]) * s
        return (labels, e)
    return [O.finite_extremes(loss.batch_hard_triplet_soft_margin_loss, mk, [1.0, 10.0, 100.0, 1000.0], expect="finite")]


def unsloth_7872(m):
    pre = m.TextPreprocessor.__new__(m.TextPreprocessor)
    return [O.edge_sweep(pre.validate_dataset, {"all_empty": ({"text": ["", "   "]},),
                                                "one_sample": ({"text": ["hello world"]},)})]


def zoo_1240(m):
    def kl(d):
        new = torch.full((1, 1), -1.0)
        ref = new + d
        out = m.grpo_compute_loss(ref, new, new, None, torch.zeros_like(new, dtype=torch.long),
                                  torch.ones_like(new), 1.0, torch.zeros(1))
        return out[2]
    ds = [s * 10.0 ** -e for e in (3, 3.5, 4) for s in (1, -1)]
    def k3(d):  # exact difference of the two fp32 inputs, then the maths in float64
        dd = torch.tensor(float(torch.tensor(-1.0) + d) + 1.0, dtype=torch.float64)
        return torch.expm1(dd) - dd
    return [O.bounds(kl, [k * 1e-5 for k in range(-300, 301, 7)], lo=0.0, hi=1.0),
            O.reference(kl, k3, lambda d: (d,), [{"d": d} for d in ds],
                        rtol=8 * 1.19e-7 / 1e-4, why="best fp32 form expm1(d)-d is accurate to ~eps/|d|, |d|>=1e-4, x8")]


PLANS = {"ds-8142": ds_8142, "ds-8199": ds_8199, "ds-8313": ds_8313, "ds-8324": ds_8324,
         "ds-8334": ds_8334, "ds-8533": ds_8533, "ray-65747": ray_65747, "st-3921": st_3921,
         "st-4012": st_4012, "st-4019": st_4019, "unsloth-7872": unsloth_7872, "zoo-1240": zoo_1240}
