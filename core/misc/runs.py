import dataclasses
import json
import math
import pickle
from pathlib import Path

import torch

from .config import TrainerConfig


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_config(saved: str, cfg: TrainerConfig) -> None:
    expected = json.loads(json.dumps(dataclasses.asdict(cfg)))
    actual = json.loads(json.dumps(saved))

    changed = [
        key
        for key in sorted(set(expected) | set(actual))
        if key not in expected
        or key not in actual
        or expected[key] != actual[key]
    ]

    if changed:
        raise ValueError(f"Run config differs in: {', '.join(changed)}")


def is_finished(run_dir: str, cfg: TrainerConfig) -> bool:
    run_dir = Path(run_dir)

    try:
        records = read_json(run_dir / "records.json")
        info = read_json(run_dir / "run_info.json")
        train = records["train"]
        valid = records["eval"]

        if not info["parameters"] or not train or not valid:
            return False

        target = cfg.num_train_steps
        if train[-1]["step"] != target or valid[-1]["step"] != target:
            return False

        for items in (train, valid):
            steps = [item["step"] for item in items]
            if any(b <= a for a, b in zip(steps, steps[1:])):
                return False

            if not all(
                math.isfinite(item["loss"])
                and math.isfinite(item["acc"])
                for item in items
            ):
                return False

        expected_tokens = target * cfg.batch_size * cfg.context_length
        if sum(item["num_tokens"] for item in train) != expected_tokens:
            return False

        marker = run_dir / "completed.json"
        if marker.exists():
            if read_json(marker)["completed_step"] != target:
                return False

        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def find_checkpoint(run_dir: str, cfg: TrainerConfig):
    paths = sorted((Path(run_dir) / "checkpoints").glob("*.pt"))
    best_path = None
    best_key = None

    required = {
        "step",
        "best_step",
        "best_val_loss",
        "train_metrics",
        "records",
        "sampler",
        "model",
        "optimizer",
        "config",
    }

    for path in paths:
        try:
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
            print(f"[checkpoint] Cannot read {path}: {exc}")
            continue

        if not isinstance(data, dict) or not required.issubset(data):
            print(f"[checkpoint] Incomplete checkpoint: {path}")
            del data
            continue

        check_config(data["config"], cfg)

        step = data["step"]
        if type(step) is not int or not 0 <= step <= cfg.num_train_steps:
            raise ValueError(f"Invalid checkpoint step in {path}: {step}")

        priority = (
            2 if path.stem == "latest"
            else 1 if path.stem.isdigit()
            else 0
        )
        key = (step, priority)

        if best_key is None or key > best_key:
            best_key = key
            best_path = path

        del data

    if paths and best_path is None:
        raise RuntimeError(f"No readable checkpoint in {run_dir}; please check manually")

    return best_path