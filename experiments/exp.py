import math

from core.train import TrainerConfig, Trainer


def build_cfg(
    *,
    token_budget=1_000_000,
    batch_size=4,
    context_length=128,
    warmup_ratio=0.02,
    decay_ratio=0.15,
    **overrides,
):
    tokens_per_step = batch_size * context_length
    steps = math.ceil(token_budget / tokens_per_step)

    warmup_steps = max(25, round(warmup_ratio * steps))
    decay_tokens = round(decay_ratio * token_budget)

    base = TrainerConfig(
        num_train_steps=steps,
        warmup_steps=warmup_steps,
        decay_tokens=decay_tokens,
        **overrides
    )

    return base


def run_exps(exps):
    for exp in exps:
        run_exp(exp)


def run_exp(exp: TrainerConfig):
    trainer = Trainer(exp)
    did_train = trainer.train()
    if did_train:
        trainer.check_plz()
