import math
from pathlib import Path

from core.training.train import Trainer
from core.misc import *


def build_cfg(
    *,
    token_budget=1_000_000,
    batch_size=4,
    context_length=128,
    warmup_ratio=0.02,
    decay_ratio=0.15,
    **overrides,
):
    if token_budget <= 0 or batch_size <= 0 or context_length <= 0:
        raise ValueError(f"token_budget, batch_size and context_length must be positive")

    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")

    if not 0 <= decay_ratio <= 1:
        raise ValueError(f"decay ratio must be in [0, 1]")

    tokens_per_step = batch_size * context_length
    steps = math.ceil(token_budget / tokens_per_step)

    warmup_steps = round(warmup_ratio * steps)
    decay_tokens = round(decay_ratio * token_budget)

    if warmup_steps >= steps:
        raise ValueError(f"warmup_steps must be smaller than training steps")

    cfg = TrainerConfig(
        batch_size=batch_size,
        context_length=context_length,
        num_train_steps=steps,
        warmup_steps=warmup_steps,
        decay_tokens=decay_tokens,
        **overrides
    )

    if cfg.schedule == "wsd":
        decay_steps = math.ceil(decay_tokens / tokens_per_step)
        if decay_steps <= 0:
            raise ValueError(f"WSD requires a positive decay duration")
        if warmup_steps + decay_steps > steps:
            raise ValueError(f"WSD warmup and decay phases overlap")

    return cfg


def run_config(
    cfg: TrainerConfig,
    resume_from: str | None = None,
    force_new: bool = False,
):
    action, run_dir, checkpoint = resolve_run(cfg, resume_from, force_new)

    if action == "skip":
        print(f"[skip] {cfg.name}: completed in {run_dir}")
        return
    elif action == "resume":
        print(f"[resume] {cfg.name}: {checkpoint}")
    else:
        print(f"[new] {cfg.name}")

    trainer = Trainer(
        cfg=cfg,
        resume_from=run_dir,
        resume_checkpoint=checkpoint
    )
    trainer.train()
    trainer.check_plz()


def run_configs(exps):
    for exp in exps:
        run_config(exp)


def resolve_run(cfg: TrainerConfig, resume_from=None, force_new=False):
    """
    Decide if an experiment should skip, resume or new.
    """
    if force_new and resume_from is not None:
        raise ValueError(f"force_new and resume_from cannot be used together")

    if force_new:
        return "new", None, None

    if resume_from is not None:
        run_dirs = [Path(resume_from)]
    else:
        root = Path("results") / cfg.name
        run_dirs = sorted(
            path for path in root.glob("*") if path.is_dir()
        )

    matching_runs = []
    for run_dir in run_dirs:
        cfg_path = run_dir / "config.json"

        if not cfg_path.exists():
            if resume_from is not None:
                raise FileNotFoundError(cfg_path)

            print(f"[ignore] No config.json: {run_dir}")
            continue

        check_config(read_json(cfg_path), cfg)
        matching_runs.append(run_dir)

    # if already finished, then skip
    for run_dir in matching_runs:
        if is_finished(run_dir, cfg):
            return "skip", run_dir, None

    if len(matching_runs) > 1:
        candidates = "\n".join(str(path) for path in matching_runs)
        raise ValueError(
            "Multiple unfinished runs; choose resume_from manually:\n"
            + candidates
        )

    if matching_runs:
        run_dir = matching_runs[0]
        checkpoint = find_checkpoint(run_dir, cfg)

        if checkpoint is not None:
            return "resume", run_dir, checkpoint

        if resume_from is not None:
            raise FileNotFoundError(
                f"No checkpoint available for resuming: {run_dir}"
            )

        print(f"[restart] No checkpoint in {run_dir}")

    return "new", None, None
