import json


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
        print(f"[train] {self.train_steps[-1]}")

    def add_eval_step(self, step, loss, acc, num_tokens):
        self.eval_steps.append({
            "step": step,
            "loss": loss,
            "acc": acc,
            "num_tokens": num_tokens,
        })
        print(f"[eval] {self.eval_steps[-1]}")

    def save(self, path):
        data = {
            "train": self.train_steps,
            "eval": self.eval_steps
        }

        with open(path, "w") as f:
            json.dump(data, f)

        print(f"Records saved to {path}")
