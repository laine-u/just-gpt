from .exp import build_cfg, run_exps


def run_scaling_setup():
    models = {
        # model_name: [num_layers, d_model, d_ff]
        "tiny" : [2, 128, 512],
        "small": [6, 256, 1024],
        "medium": [12, 512, 2048],
    }

    exps = []
    for exp_name, (num_layers, d_model, d_ff) in models.items():
        cfg = build_cfg(
            name=exp_name,
            num_layers=num_layers,
            d_model=d_model,
            d_ff=d_ff,
        )
        exps.append(cfg)

    run_exps(exps)