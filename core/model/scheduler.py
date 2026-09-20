import math
from torch.optim import Optimizer

from ..misc.config import TrainerConfig


class LRScheduler:
    def __init__(self, optimizer: Optimizer, cfg: TrainerConfig):
        self.cfg = cfg
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

        tps = cfg.batch_size * cfg.context_length
        self.decay_steps = math.ceil(cfg.decay_tokens / tps) if cfg.decay_tokens else 0
        self.decay_start = cfg.num_train_steps - self.decay_steps

        if cfg.schedule == "wsd" and self.decay_steps <=0:
            raise ValueError("WSD needs decay_tokens > 0, or behaves like constant.")
        if cfg.num_train_steps <= cfg.warmup_steps:
            raise ValueError("num_train_steps must > warmup_steps, or never decays.")

    def step(self, step: int):
        f = self.factor(step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * f

    def factor(self, step: int) -> float:
        if self.cfg.schedule == "constant":
            decay = 1.0
        elif self.cfg.schedule == "cosine":
            decay = self.cosine_decay(
                step,
                self.cfg.warmup_steps,
                self.cfg.num_train_steps - self.cfg.warmup_steps,
                self.cfg.min_lr_ratio
            )
        elif self.cfg.schedule == "wsd":
            decay = self.cosine_decay(
                step,
                self.decay_start,
                self.decay_steps,
                self.cfg.min_lr_ratio
            )
        else:
            raise ValueError(f"Why gave me a {self.cfg.schedule} as scheduler?")
        
        return self.warmup_factor(step, self.cfg.warmup_steps) * decay

    @staticmethod
    def warmup_factor(step: int, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        else:
            return min(1.0, step / warmup_steps)
        
    @staticmethod
    def cosine_decay(step: int, decay_start: int, decay_steps: int, min_lr_ratio: float) -> float:
        if decay_steps <= 0:
            return 1.0
        
        u = (step - decay_start) / decay_steps
        u = min(max(u, 0.0), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * u))
