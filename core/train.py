import os, json, random, dataclasses
from pathlib import Path
from tqdm import tqdm

import tiktoken
import torch
import numpy as np
import torch.nn.functional as F

from .model import AdamW, TransformerLM, LRScheduler
from .data import load_tokens, get_batch, get_random_batch, WindowSampler
from .misc import Record, TrainerConfig, generate_timestamp, check_config, find_checkpoint


class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        resume_from: str | None = None,
        resume_checkpoint: str | Path | None = None,
    ):
        self.cfg = cfg
        self.resume_from = resume_from
        self.resume_checkpoint = resume_checkpoint

        if resume_checkpoint is not None and resume_from is None:
            raise ValueError(f"resume_checkpoint requires resume_from")

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

        print(json.dumps(dataclasses.asdict(self.cfg), indent=2))
        counts = self._count_parametes(self.model)
        print(f"Total parameters: {counts['total'] / 1e6:.2f}M")
        print(f"  - Body:  {counts['body'] / 1e6:.2f}M")
        print(f"  - Vocab: {counts['vocab'] / 1e6:.2f}M")

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

        num_windows = (len(self.train_toks) - 1) // self.cfg.context_length
        self.sampler = WindowSampler(num_windows, self.cfg.batch_size, self.cfg.seed)

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

    @staticmethod
    def _count_parametes(model):
        total = sum(p.numel() for p in model.parameters())
        body = sum(p.numel() for p in model.transformer_blocks.parameters())
        body += sum(p.numel() for p in model.ln_final.parameters())
        return {
            "total": total,
            "body": body,
            "vocab": total - body
        }

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

        completed_step = 0
        if self.resume_from:
            next_step = self.load_check_point(self.resume_from)
            completed_step = next_step - 1
        else:
            self.save_config()
            self.save_run_info()

        if completed_step > self.cfg.num_train_steps:
            raise ValueError("Checkpoint exceeds total training steps")
        if completed_step == self.cfg.num_train_steps:
            print(f"Training was already completed, step = {completed_step}")
            self.finalize_training(completed_step)
            return False

        print(f"Training steps: {self.cfg.num_train_steps}, starts from step {completed_step + 1}")

        with tqdm(
            total=self.cfg.num_train_steps,
            initial=completed_step,
            desc="Train",
            unit="step",
            dynamic_ncols=True,
            mininterval=0.5
        ) as progress:
            self.update_progress(progress)

            for step in range(completed_step + 1, self.cfg.num_train_steps + 1):
                self.train_step(step)
                completed_step = step

                if self.check_step(step, self.cfg.log_every):
                    self.record_metrics(step, "train")
                    self.update_progress(progress)

                progress.update(1)

                if self.check_step(step, self.cfg.eval_every):
                    progress.set_description("Train / eval")
                    _, is_best = self.evaluate_step(step)
                    if is_best:
                        self.save_checkpoint(step, "best")
                    progress.set_description("Train")

                if self.check_step(step, self.cfg.checkpoint_every):
                    self.save_record()
                    self.save_checkpoint(step)
                    self.save_checkpoint(step, "latest")

            progress.set_description("Finalizing")
            self.finalize_training(completed_step)
            self.update_progress(progress)
            progress.refresh()

        return True

    def train_step(self, step):
        window_ids = self.sampler.next_batch()
        inputs, targets = get_batch(
            self.train_toks,
            window_ids,
            self.cfg.context_length,
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
        eval_rng = torch.Generator(device="cpu").manual_seed(self.cfg.eval_seed)

        for _ in range(self.cfg.num_eval_steps):
            inputs, targets = get_random_batch(
                self.valid_toks,
                self.cfg.batch_size,
                self.cfg.context_length,
                rng=eval_rng
            )
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = lm_loss(outputs, targets)
            self.update_metrics("eval", loss, outputs, targets)

        val_loss, val_acc = self.record_metrics(step, "eval")
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_step = step

        self.model.train()

        tqdm.write(
            f"[eval] step={step} "
            f"loss={val_loss:.4f} "
            f"acc={val_acc:.2%} "
            f"best={is_best}"
        )
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

    def update_progress(self, progress):
        if not self.record.train_steps:
            return

        metrics = self.record.train_steps[-1]
        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            acc=f"{metrics['acc']:.2%}",
            lr=f"{metrics['lr']:.2e}",
            refresh=False,
        )

    # ==================================================
    # Check
    # ==================================================

    def check_plz(self):
        print(f"==================================================")
        print(f"Checking out")
        print(f"==================================================")
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
        path = base_dir / "config.json"

        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-9") as f:
            json.dump(dataclasses.asdict(self.cfg), f)
        os.replace(tmp_path, path)

    def save_run_info(self):
        base_dir = self.get_base_dir()
        path = base_dir / "run_info.json"

        info = {
            "parameters": self._count_parametes(self.model),
            "parameter_count_def": {
                "body": "transformer_blocks + ln_final",
                "vocab": "tok_emb + lm_head",
            }
        }

        tmp_path =  path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def save_record(self):
        base_dir = self.get_base_dir()
        record_path = base_dir / "records.json"
        self.record.save(record_path)

    def save_checkpoint(self, step, name=None):
        base_dir = self.get_base_dir()
        if name is None:
            name = f"{step}"

        data = {
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
            "sampler": self.sampler.state_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": dataclasses.asdict(self.cfg),
        }

        path = base_dir / "checkpoints" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_name(path.name + ".tmp")
        torch.save(data, tmp_path)
        os.replace(tmp_path, path)

    def load_check_point(self, resume_from):
        base_dir = Path(resume_from)

        if self.resume_checkpoint is not None:
            path = Path(self.resume_checkpoint)
        else:
            path = find_checkpoint(base_dir, self.cfg)

        if path is None:
            raise FileNotFoundError(f"No checkpoint found in {base_dir}")

        data = torch.load(path, map_location=self.device, weights_only=True)
        check_config(data["config"], self.cfg)
        print(f"[resume] {path}, completed step={data['step']}")

        self.best_step = data["best_step"]
        self.best_val_loss = data["best_val_loss"]
        self.training_loss = data["train_metrics"]["loss_sum"]
        self.training_correct = data["train_metrics"]["correct"]
        self.training_count = data["train_metrics"]["count"]
        self.record.train_steps = data["records"]["train"]
        self.record.eval_steps = data["records"]["eval"]
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.sampler.load_state_dict(data["sampler"])
        return data["step"] + 1

    def finalize_training(self, completed_step):
        if completed_step != self.cfg.num_train_steps:
            raise ValueError(f"Cannot finalize unfinished training")

        if self.training_count > 0:
            self.record_metrics(completed_step, "train")

        has_final_eval = (
            self.record.eval_steps
            and self.record.eval_steps[-1]["step"] == completed_step
        )
        if not has_final_eval:
            self.reset_metrics("eval")
            self.evaluate_step(completed_step)

        if self.best_step == completed_step:
            self.save_checkpoint(completed_step, "best")

        self.save_checkpoint(completed_step, "latest")
        self.save_record()
        self.save_config()
        self.save_run_info()

        path = self.get_base_dir() / "completed.json"
        tmp_path = path.with_name(path.name + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump({"completed_step": completed_step}, f, indent=2)

        os.replace(tmp_path, path)

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
