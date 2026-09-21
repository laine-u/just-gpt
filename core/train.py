import json, random, dataclasses
from pathlib import Path

import tiktoken
import torch
import numpy as np
import torch.nn.functional as F

from core.model.optimizer import AdamW
from core.model.transformer import TransformerLM
from core.model.scheduler import LRScheduler
from core.misc.utils import generate_timestamp
from core.misc.config import TrainerConfig
from core.misc.record import Record
from core.data.dataset import load_tokens, get_batch


class Trainer:
    def __init__(self, cfg: TrainerConfig, resume_from: str | None = None):
        self.cfg = cfg

        self.resume_from = resume_from
        if not resume_from:
            self.uuid = generate_timestamp()

        self.record = Record()
        self.setup()

    # ==================================================
    # Setup
    # ==================================================

    def setup(self):
        """
        Setup tokenizer, model, optimizer and schedulers.
        """
        print(f"==================================================")
        print(f"Setup")
        print(f"==================================================")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.setup_randomness()
        self.setup_datasets()
        self.setup_model()
        self.setup_scheduler()

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Parameters: {num_params / 1e6:.2f}M")

    def setup_randomness(self):
        seed = self.cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Setting up seeds to {seed}")

    def setup_datasets(self):
        data_dir = Path(self.cfg.data_dir)

        with (data_dir / "meta.json").open(encoding="utf-8") as f:
            meta = json.load(f)

        if (
            meta["tokenizer"] != self.tokenizer.name or
            meta["vocab_size"] != self.tokenizer.n_vocab or
            meta["eot_token"] != self.tokenizer.eot_token
        ):
            raise ValueError(f"Token files and model tokenizer do not match")

        def load_split(name):
            info = meta["splits"][name]
            return load_tokens(
                path=data_dir / info["file"],
                dtype=meta["dtype"],
                expected_tokens=info["num_tokens"]
            )

        self.train_toks = load_split("train")
        self.valid_toks = load_split("valid")

    def setup_model(self):
        self.model = TransformerLM(
            self.tokenizer.n_vocab,
            num_layers=self.cfg.num_layers,
            num_heads=self.cfg.num_heads,
            context_length=self.cfg.context_length,
            d_model=self.cfg.d_model,
            d_ff=self.cfg.d_ff,
            theta=self.cfg.theta,
        )
        self.model.to(self.device)

        self.optimizer: torch.optim.Optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            weight_decay=self.cfg.weight_decay,
            eps=self.cfg.eps,
        )

    def setup_scheduler(self):
        self.scheduler = LRScheduler(self.optimizer, self.cfg)

    # ==================================================
    # Training
    # ==================================================

    def train(self) -> bool:
        """
        Return:
            did_train: bool
        """
        print(f"==================================================")
        print(f"Training")
        print(f"==================================================")
        self.model.train()

        self.reset_metrics("train")
        self.reset_metrics("eval")
        self.best_step = 0
        self.best_val_loss = float("inf")
        self.train_rng = torch.Generator(device="cpu").manual_seed(self.cfg.seed)

        completed_step = 0
        if self.resume_from:
            next_step = self.load_check_point(self.resume_from)
            completed_step = next_step - 1
        else:
            self.save_config()

        if completed_step > self.cfg.num_train_steps:
            raise ValueError("Checkpoint 超出训练总步数！")
        if completed_step == self.cfg.num_train_steps:
            print(f"训练已经完成，step = {completed_step}")
            return False

        for step in range(completed_step + 1, self.cfg.num_train_steps + 1):
            self.train_step(step)
            completed_step = step

            if self.check_step(step, self.cfg.log_every):
                self.record_metrics(step, "train")

            if self.check_step(step, self.cfg.eval_every):
                _, is_best = self.evaluate_step(step)
                if is_best:
                    self.save_checkpoint(step, "best")

            if self.check_step(step, self.cfg.checkpoint_every):
                self.save_record()
                self.save_checkpoint(step)
                self.save_checkpoint(step, "latest")

        if self.training_count > 0:
            self.record_metrics(completed_step, "train")

        if not self.check_step(completed_step, self.cfg.eval_every):
            _, is_best = self.evaluate_step(completed_step)
            if is_best:
                self.save_checkpoint(completed_step, "best")
                self.save_checkpoint(completed_step, "latest")

        self.save_checkpoint(completed_step, "latest")
        return True

    def train_step(self, step):
        inputs, targets = get_batch(
            self.train_toks,
            self.cfg.batch_size,
            self.cfg.context_length,
            rng=self.train_rng,
        )
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)
        loss = lm_loss(outputs, targets)
        self.update_metrics("train", loss, outputs, targets)

        self.scheduler.step(step)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    @torch.no_grad
    def evaluate_step(self, step):
        self.model.eval()
        eval_rng = torch.Generator(device="cpu").manual_seed(self.cfg.seed)

        for _ in range(self.cfg.num_eval_steps):
            inputs, targets = get_batch(
                self.valid_toks,
                self.cfg.batch_size,
                self.cfg.context_length,
                rng=eval_rng
            )
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = lm_loss(outputs, targets)
            self.update_metrics("eval", loss, outputs, targets)

        val_loss, _ = self.record_metrics(step, "eval")
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_step = step

        self.model.train()
        return val_loss, is_best

    def record_metrics(self, step, mode):
        if mode == "train":
            loss = self.training_loss / self.training_count
            acc = self.training_correct / self.training_count
            self.record.add_train_step(
                step=step, loss=loss, acc=acc, num_tokens=self.training_count,
                lr=self.optimizer.param_groups[0]["lr"]
            )
        elif mode == "eval":
            loss = self.validating_loss / self.validating_count
            acc = self.validating_correct / self.validating_count
            self.record.add_eval_step(
                step=step, loss=loss, acc=acc,num_tokens=self.validating_count
            )
        else:
            raise ValueError(f"Unknown metrics mode: {mode}")

        self.reset_metrics(mode)
        return loss, acc

    def reset_metrics(self, mode):
        if mode == "train":
            self.training_loss = 0.0
            self.training_correct = 0
            self.training_count = 0
        elif mode == "eval":
            self.validating_loss = 0.0
            self.validating_correct = 0
            self.validating_count = 0
        else:
            raise ValueError(f"Unknown metrics mode: {mode}")

    def update_metrics(self, mode, loss, logits, targets):
        if mode == "train":
            self.training_loss += loss.item() * targets.numel()
            self.training_correct += (logits.argmax(-1) == targets).sum().item()
            self.training_count += targets.numel()
        elif mode == "eval":
            self.validating_loss += loss.item() * targets.numel()
            self.validating_correct += (logits.argmax(-1) == targets).sum().item()
            self.validating_count += targets.numel()
        else:
            raise ValueError(f"Unknown metrics mode: {mode}")

    @staticmethod
    def check_step(step, every):
        return (every > 0) and (step % every == 0)

    # ==================================================
    # Check
    # ==================================================

    def check_plz(self):
        print(f"==================================================")
        print(f"Checking out")
        print(f"==================================================")
        self.save_record()
        self.generate()

    def get_base_dir(self):
        if self.resume_from:
            base_dir = Path(self.resume_from)
        else:
            base_dir = Path(f"results/{self.cfg.name}") / self.uuid
            base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def save_config(self):
        base_dir = self.get_base_dir()
        config_path = base_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(self.cfg), f)

    def save_record(self):
        base_dir = self.get_base_dir()
        record_path = base_dir / "records.json"
        self.record.save(record_path)

    def save_checkpoint(self, step, name=None):
        base_dir = self.get_base_dir()
        if name is None:
            name = f"{step}"

        path = base_dir / "checkpoints" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": step,
            "best_step": self.best_step,
            "best_val_loss": self.best_val_loss,
            "train_metrics": {
                "loss_sum": self.training_loss,
                "correct": self.training_correct,
                "count": self.training_count,
            },
            "records": {
                "train": self.record.train_steps,
                "eval": self.record.eval_steps,
            },
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.train_rng.get_state(),

            "config": dataclasses.asdict(self.cfg),
        }, path)

    def load_check_point(self, resume_from):
        base_dir = Path(resume_from)

        path = base_dir / "checkpoints" / "latest.pt"
        data = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )
        self.best_step = data["best_step"]
        self.best_val_loss = data["best_val_loss"]

        self.training_loss = data["train_metrics"]["loss_sum"]
        self.training_correct = data["train_metrics"]["correct"]
        self.training_count = data["train_metrics"]["count"]
        self.record.train_steps = data["records"]["train"]
        self.record.eval_steps = data["records"]["eval"]

        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.train_rng.set_state(data["rng"].cpu())
        return data["step"] + 1

    @torch.no_grad
    def generate(self):
        self.model.eval()

        prompt = "The same"
        ids = self.tokenizer.encode_ordinary(prompt)
        if not ids:
            raise ValueError("Generation prompt must not be empty")

        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        for _ in range(500):
            logits = self.model(tokens[:, -self.model.context_length :])
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, next_id), dim=1)
            if next_id.item() == self.tokenizer.eot_token:
                break

        print(repr(self.tokenizer.decode(tokens[0].tolist())))


def lm_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
