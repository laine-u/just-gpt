import os
import json

from pathlib import Path


class Record:
    def __init__(self):
        self.train_steps = []
        self.eval_steps = []

    def add_train_step(self, step, loss, acc, num_tokens, lr=0.0):
        self.train_steps.append({
            "step": step,
            "loss": loss,
            "acc": acc,
            "num_tokens": num_tokens,
            "lr": lr
        })

    def add_eval_step(self, step, loss, acc, num_tokens):
        self.eval_steps.append({
            "step": step,
            "loss": loss,
            "acc": acc,
            "num_tokens": num_tokens,
        })

    def save(self, path):
        path = Path(path)
        tmp_path = path.with_name(path.name + ".tmp")

        data = {
            "train": self.train_steps,
            "eval": self.eval_steps
        }

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        os.replace(tmp_path, path)