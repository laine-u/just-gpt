import re
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def group_runs(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (row["model"], row["token_budget"])
        groups[key].append(row)

    for runs in groups.values():
        runs.sort(key=lambda r: r["seed"])

    return groups


def summarize_groups(rows):
    groups = group_runs(rows)
    model_order = {"S": 0, "M": 1, "L": 2}
    grouped_rows = []

    keys = sorted(
        groups,
        key=lambda key: (model_order[key[0]], key[1]),
    )

    for model, budget in keys:
        runs = groups[(model, budget)]

        if len(runs) < 2:
            raise ValueError("At least two seeds are required for sample SD")

        result = {
            "model": model,
            "token_budget": budget,
            "num_seeds": len(runs),
            "seeds": ",".join(str(run["seed"]) for run in runs),
        }

        for field in (
            "total_params",
            "body_params",
            "num_layers",
            "d_model",
            "final_step",
            "trained_tokens",
        ):
            values = {run[field] for run in runs}
            if len(values) != 1:
                raise ValueError(
                    f"Inconsistent {field}: model={model}, budget={budget}"
                )
            result[field] = runs[0][field]

        for metric in ("val_loss", "val_acc", "val_ppl"):
            values = [run[metric] for run in runs]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = stdev(values)

        grouped_rows.append(result)

    return grouped_rows


def summarize_run(run_dir):
    cfg = read_json(run_dir / "config.json")
    info = read_json(run_dir / "run_info.json")
    records = read_json(run_dir / "records.json")

    match = re.fullmatch(
        r"owt-(S|M|L)-tokens(\d+)-seed(\d+)",
        cfg["name"]
    )
    if match is None:
        raise ValueError(f"Unable to extract experiment name: {cfg['name']}")

    model, budget, seed = match.groups()
    if int(seed) != cfg['seed']:
        raise ValueError(f"Experiment name is mismatch with seed in config: {run_dir}")

    train = records["train"]
    valid = records["eval"]
    if not train or not valid:
        raise ValueError(f"Missing train/valid records: {run_dir}")

    final_train = train[-1]
    final_valid = valid[-1]
    expected_step = cfg['num_train_steps']
    if not (final_train["step"] == final_valid["step"] == expected_step):
        raise ValueError(f"Experiments is unfinished or missing final validation: {run_dir}")

    train_tokens = sum(item["num_tokens"] for item in train)
    toknes_per_step = cfg["batch_size"] * cfg["context_length"]
    if train_tokens != expected_step * toknes_per_step:
        raise ValueError(f"Training token count mismatch: {run_dir}")
    
    return {
        "model": model,
        "token_budget": int(budget),
        "seed": cfg["seed"],
        "total_params": info["parameters"]["total"],
        "body_params": info["parameters"]["body"],
        "num_layers": cfg["num_layers"],
        "d_model": cfg["d_model"],
        "final_step": final_valid["step"],
        "trained_tokens": train_tokens,
        "val_loss": final_valid["loss"],
        "val_acc": final_valid["acc"],
        "val_ppl": math.exp(final_valid["loss"]),
        "run_dir": str(run_dir)
    }


def collect_summary(results_dir: Path):
    rows = []
    seen = set()

    for path in sorted(results_dir.glob("owt-*/ */config.json".replace(" ", ""))):
        row = summarize_run(path.parent)
        key = (row["model"], row["token_budget"], row["seed"])

        if key in seen:
            raise ValueError(f"Configuration has multiple runs, need to manually select: {key}")
        
        seen.add(key)
        rows.append(row)

    expected = {
        (model, budget, seed)
        for model in ("S", "M", "L")
        for budget in (10_000_000, 30_000_000, 100_000_000)
        for seed in (42, 43, 44)
    }
    if seen != expected:
        raise ValueError(f"Experiment combination mismatch: missing {expected - seen}, extra {seen - expected}")

    model_order = {"S": 0, "M": 1, "L": 2}
    return sorted(
        rows,
        key=lambda r: (
            model_order[r["model"]],
            r["token_budget"],
            r["seed"]
        )
    )
