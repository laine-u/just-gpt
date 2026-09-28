from .exp import build_cfg, run_configs


def build_scaling_configs(
    *,
    dataset="owt",
    data_dir="data/processed/owt-gpt2",
    batch_size=8,
    context_length=256,
):
    models = {
        "S": {"num_layers": 4, "d_model": 128, "num_heads": 4, "d_ff": 384, },
        "M": { "num_layers": 6, "d_model": 256, "num_heads": 8, "d_ff": 768,},
        "L": { "num_layers": 8, "d_model": 384, "num_heads": 12, "d_ff": 1152,},
    }

    token_budgets = (10_000_000, 30_000_000, 100_000_000, )
    seeds = (42, 43, 44)

    configs = []

    for model_name, model_config in models.items():
        for token_budget in token_budgets:
            for seed in seeds:
                cfg = build_cfg(
                    name=(
                        f"{dataset}-{model_name}"
                        f"-tokens{token_budget}-seed{seed}"
                    ),
                    dataset=dataset,
                    data_dir=data_dir,
                    token_budget=token_budget,
                    batch_size=batch_size,
                    context_length=context_length,
                    seed=seed,
                    eval_seed=2026,
                    lr=3e-4,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                    eps=1e-8,
                    schedule="wsd",
                    warmup_ratio=0.02,
                    decay_ratio=0.15,
                    min_lr_ratio=0.1,
                    num_eval_steps=128,
                    log_every=100,
                    eval_every=1000,
                    checkpoint_every=5000,
                    **model_config,
                )
                configs.append(cfg)

    return configs


def run_scaling_setup():
    configs = build_scaling_configs()

    for cfg in configs:
        print(
            cfg.name,
            f"steps={cfg.num_train_steps}",
            f"batch={cfg.batch_size}",
            f"context={cfg.context_length}",
        )

    run_configs(configs)


if __name__ == "__main__":
    run_scaling_setup()