import math
import dataclasses

from core.train import TrainerConfig, Trainer

TOKEN_BUDGET = 65_000
WARMUP_RATIO = 0.02
DECAY_RATIO = 0.15


def build_cfg(**overrides):
    batch_size, context_length = 4, 32
    tps = batch_size * context_length
    steps = math.ceil(TOKEN_BUDGET / tps)

    base = TrainerConfig(
        name="Antigone",

        dataset="antigone",
        batch_size=batch_size,
        context_length=context_length,

        num_layers=2,
        num_heads=4,
        d_model=24,
        d_ff=64,
        theta=10000.0,

        betas=(0.9, 0.999),
        weight_decay=0.1,
        eps=1e-8,
        lr=3e-4,
        schedule="wsd",
        min_lr_ratio=0.1,

        num_train_steps=steps,
        warmup_steps=max(25, round(WARMUP_RATIO * steps)),
        decay_tokens=round(DECAY_RATIO * TOKEN_BUDGET),

        num_eval_steps=100,
        log_every=25,
        eval_every=50,
        checkpoint_every=250,
        seed=42,
    )

    return dataclasses.replace(base, **overrides) if overrides else base


def run_exps(exps):
    for exp in exps:
        run_exp(exp)


def run_exp(exp: TrainerConfig):
    trainer = Trainer(exp)
    trainer.train()
    trainer.check_plz()
