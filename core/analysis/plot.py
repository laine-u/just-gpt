import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .summarize import read_json, group_runs


MODELS = ("S", "M", "L")
COLORS = {
    "S": "#0072B2",
    "M": "#D55E00",
    "L": "#009E73",
}



def mean_std(values):
    values = np.asarray(values, dtype=float)

    if values.shape[0] < 2:
        raise ValueError(f"At least two seeds are required for sample SD")
    
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def save_fig(fig, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved {path}")

    plt.close(fig)


def get_lr_stages(runs):
    stages = set()

    for run in runs:
        cfg = read_json(Path(run["run_dir"]) / "config.json")

        # 当前这批实验全部使用 WSD。
        if cfg["schedule"] != "wsd":
            raise ValueError("LR stage markers currently support WSD only")

        tokens_per_step = cfg["batch_size"] * cfg["context_length"]
        decay_steps = math.ceil(cfg["decay_tokens"] / tokens_per_step)

        warmup_end = cfg["warmup_steps"] * tokens_per_step
        decay_start = (
            cfg["num_train_steps"] - decay_steps
        ) * tokens_per_step
        train_end = cfg["num_train_steps"] * tokens_per_step

        stages.add((warmup_end, decay_start, train_end))

    # 同一子图包含多个模型和 seed，必须共享相同的阶段边界。
    if len(stages) != 1:
        raise ValueError("Runs in the same subplot have different LR stages")

    return tuple(value / 1e6 for value in stages.pop())


def mark_lr_stages(ax, warmup_end, decay_start):
    if warmup_end > 0:
        ax.axvline(
            warmup_end,
            color="0.45",
            linestyle=":",
            linewidth=1.2,
            label="Warmup end",
            zorder=1,
        )

    ax.axvline(
        decay_start,
        color="0.25",
        linestyle="--",
        linewidth=1.2,
        label="Decay start",
        zorder=1,
    )


def load_curves(runs, mode):
    token_axes = []
    values = []

    for run in runs:
        run_dir = Path(run["run_dir"])
        cfg =read_json(run_dir / "config.json")
        records = read_json(run_dir / "records.json")[mode]

        tokens_per_step = cfg["batch_size"] * cfg["context_length"]
        tokens = np.array(
            [record["step"] * tokens_per_step for record in records],
            dtype=np.int64,
        )

        token_axes.append(tokens)
        values.append([
            [record["loss"], record["acc"] * 100]
            for record in records
        ])

    if not all(
        np.array_equal(tokens, token_axes[0])
        for tokens in token_axes[1:]
    ):
        raise ValueError(
            f"Unaligned {mode} curves: "
            f"{[run['run_dir'] for run in runs]}"
        )

    return token_axes[0] / 1e6, np.asarray(values, dtype=float)


def plot_curves(groups, output_dir, mode):
    if mode not in ("train", "eval"):
        raise ValueError(f"Unknown mode: {mode}")
    
    budgets = sorted({budget for _, budget in groups})
    title = "Training" if mode == "train" else "Validation"

    fig, axes = plt.subplots(
        2,
        len(budgets),
        figsize=(4.5 * len(budgets), 6.5),
        sharex="col",
        sharey="row",
        squeeze=False,
        layout="constrained",
    )

    for col, budget in enumerate(budgets):
        for model in MODELS:
            x, values = load_curves(groups[(model, budget)], mode)
            mean, std = mean_std(values)

            for metric in range(2):
                ax = axes[metric, col]
                color = COLORS[model]

                ax.plot(
                    x,
                    mean[:, metric],
                    color=color,
                    label=model,
                    linewidth=1.7,
                    marker="o" if mode == "eval" else None,
                    markersize=3,
                )
                ax.fill_between(
                    x,
                    mean[:, metric] - std[:, metric],
                    mean[:, metric] + std[:, metric],
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )

        runs = [
            run
            for model in MODELS
            for run in groups[(model, budget)]
        ]
        warmup_end, decay_start, train_end = get_lr_stages(runs)

        for ax in axes[:, col]:
            mark_lr_stages(ax, warmup_end, decay_start)
            ax.grid(alpha=0.25)
            ax.set_xlim(0, train_end)

        axes[0, col].set_title(f"Budget: {budget / 1e6:g}M tokens")
        axes[0, col].legend(fontsize=8)
        axes[1, col].set_xlabel("Training tokens (M)")

    axes[0, 0].set_ylabel(f"{title} loss")
    axes[1, 0].set_ylabel(f"{title} accuracy (%)")
    fig.suptitle(f"{title} curves — mean ± 1 sample SD")

    save_fig(fig, output_dir, f"{mode}_curves")


def plot_scaling(groups, output_dir):
    budgets = sorted({budget for _, budget in groups})

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 4.5),
        sharey=True,
        layout="constrained",
    )

    # 左图：固定训练预算，比较模型参数量。
    for budget in budgets:
        x, means, stds = [], [], []

        for model in MODELS:
            runs = groups[(model, budget)]
            mean, std = mean_std([run["val_loss"] for run in runs])

            x.append(runs[0]["total_params"] / 1e6)
            means.append(mean)
            stds.append(std)

        axes[0].errorbar(
            x, means, yerr=stds,
            fmt="o-",
            capsize=4,
            label=f"{budget / 1e6:g}M tokens",
        )

    # 右图：固定模型，比较实际训练 token 数。
    for model in MODELS:
        x, means, stds = [], [], []

        for budget in budgets:
            runs = groups[(model, budget)]
            mean, std = mean_std([run["val_loss"] for run in runs])

            x.append(runs[0]["trained_tokens"] / 1e6)
            means.append(mean)
            stds.append(std)

        axes[1].errorbar(
            x, means, yerr=stds,
            fmt="o-",
            capsize=4,
            color=COLORS[model],
            label=model,
        )

    parameter_ticks = [
        groups[(model, budgets[0])][0]["total_params"] / 1e6
        for model in MODELS
    ]
    token_ticks = [
        groups[(MODELS[0], budget)][0]["trained_tokens"] / 1e6
        for budget in budgets
    ]

    for ax, ticks in zip(axes, (parameter_ticks, token_ticks)):
        ax.set_xscale("log")
        ax.set_xticks(ticks, labels=[f"{value:.3g}" for value in ticks])
        ax.minorticks_off()
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].set_title("Effect of model size")
    axes[0].set_xlabel("Total parameters (M, log scale)")
    axes[0].set_ylabel("Final validation loss")

    axes[1].set_title("Effect of training tokens")
    axes[1].set_xlabel("Training tokens (M, log scale)")

    fig.suptitle("Scaling results — mean ± 1 sample SD")
    save_fig(fig, output_dir, "scaling")


def plot_all(rows, output_dir):
    groups = group_runs(rows)

    plot_curves(groups, output_dir, mode="eval")
    plot_curves(groups, output_dir, mode="train")
    plot_scaling(groups, output_dir)
