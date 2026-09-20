import json
import dataclasses
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from core.model.optimizer import AdamW
from core.model.transformer import TransformerLM
from core.model.scheduler import LRScheduler
from core.misc.config import TrainerConfig
from core.misc.record import Record


def get_batch(tokens, batch_size, context_length, *, rng=None):
    assert tokens.numel() >= context_length

    starts = torch.randint(
        tokens.numel() - context_length,
        (batch_size,),
        device=tokens.device,
        generator=rng,
    )
    indices = starts[:, None] + torch.arange(context_length + 1, device=tokens.device)
    chunks = tokens[indices]
    return chunks[:, :-1], chunks[:, 1:]


def lm_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

class Trainer:
    def __init__(self, cfg: TrainerConfig, resume: bool = False):
        self.cfg = cfg
        self.resume = resume
        self.record = Record()

        self.reset_metrics("train")
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
        self.setup_datasets()
        self.setup_model()
        self.setup_scheduler()

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Parameters: {num_params / 1e6:.2f}M")

    def setup_datasets(self):
        if self.cfg.dataset == "antigone":
            text = Path("data/Antigone.txt").read_text(encoding="utf-8")
            split = int(len(text) * 0.9)
            train_text = text[:split]
            valid_text = text[split:]

            self.train_toks = torch.tensor(self.tokenizer.encode(train_text), dtype=torch.long)
            self.valid_toks = torch.tensor(self.tokenizer.encode(valid_text), dtype=torch.long)
        else:
            raise ValueError(f"WoW, could've fooled me, you {self.cfg.dataset}.")

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

    def train(self):
        print(f"==================================================")
        print(f"Training")
        print(f"==================================================")
        self.model.train()

        rng = torch.Generator(device=self.valid_toks.device).manual_seed(self.cfg.seed)
        if self.resume:
            start = self.load_check_point(rng)
        else:
            start = 1

        for step in range(start, self.cfg.num_train_steps + 1):
            self.train_step(step, rng)

            if self.check_step(step, self.cfg.log_every):
                self.record.add_train_step(
                    step=step,
                    loss=self.training_loss / self.training_count,
                    acc=self.training_correct / self.training_count,
                    num_tokens=self.training_count,
                    lr=self.optimizer.param_groups[0]["lr"]
                )
                self.reset_metrics("train")

            if self.check_step(step, self.cfg.eval_every):
                self.evaluate_step(step)

            if self.check_step(step, self.cfg.checkpoint_every):
                self.save_checkpoint(step, rng)
                self.save_checkpoint(step, rng, "latest")

        if self.training_count > 0:
            self.record.add_train_step(
                step=step,
                loss=self.training_loss / self.training_count,
                acc=self.training_correct / self.training_count,
                num_tokens=self.training_count,
                lr=self.optimizer.param_groups[0]["lr"]
            )

    def train_step(self, step, rng=None):
        inputs, targets = get_batch(
            self.train_toks,
            self.cfg.batch_size,
            self.cfg.context_length,
            rng=rng
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
        self.reset_metrics("eval")

        rng = torch.Generator(device=self.valid_toks.device).manual_seed(self.cfg.seed)

        for _ in range(self.cfg.num_eval_steps):
            inputs, targets = get_batch(
                self.valid_toks,
                self.cfg.batch_size,
                self.cfg.context_length,
                rng=rng
            )
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = lm_loss(outputs, targets)
            self.update_metrics("eval", loss, outputs, targets)

        self.record.add_eval_step(
            step=step,
            loss=self.validating_loss / self.validating_count,
            acc=self.validating_correct / self.validating_count,
            num_tokens=self.validating_count
        )
        self.model.train()

    def reset_metrics(self, mode):
        if mode == "train":
            self.training_loss = 0.0
            self.training_correct = 0
            self.training_count = 0
        else:
            self.validating_loss = 0.0
            self.validating_correct = 0
            self.validating_count = 0

    def update_metrics(self, mode, loss, logits, targets):
        if mode == "train":
            self.training_loss += loss.item() * targets.numel()
            self.training_correct += (logits.argmax(-1) == targets).sum().item()
            self.training_count += targets.numel()
        else:
            self.validating_loss += loss.item() * targets.numel()
            self.validating_correct += (logits.argmax(-1) == targets).sum().item()
            self.validating_count += targets.numel()

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
        base_dir = self.get_base_dir()

        config_path = base_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(self.cfg), f)

        record_path = base_dir / "records.json"
        self.record.save(record_path)
        self.generate()

    def get_base_dir(self):
        base_dir = Path(f"results/{self.cfg.name}")
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def save_checkpoint(self, step, rng, name=None):
        base_dir = self.get_base_dir()
        if name is None:
            name = f"{step}"

        path = base_dir / "checkpoints" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": rng.get_state(),
            "config": dataclasses.asdict(self.cfg),
        }, path)

    def load_check_point(self, rng):
        base_dir = self.get_base_dir()

        path = base_dir / "checkpoints" / "latest.pt"
        data = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        rng.set_state(data["rng"])
        return data["step"]

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
