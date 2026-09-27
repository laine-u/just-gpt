import dataclasses


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    name: str = "Antigone"
    dataset: str = "antigone"
    data_dir: str = "data/processed/tinystories-gpt2-debug"

    # ---------- Transformer ----------
    num_layers: int = 2
    num_heads: int = 6
    d_model: int = 12
    d_ff: int = 64
    theta: int = 10000.0

    # ---------- Optimizer ----------
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1e-8

    # ---------- LR Scheduler ----------
    schedule: str = "wsd"
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    decay_tokens: int = 0

    # ---------- Hyperparameters ----------
    batch_size: int = 4
    context_length: int = 32
    num_train_steps: int = 5000
    num_eval_steps: int = 200

    # ---------- Stuff ----------
    log_every: int = 100
    eval_every: int = 500
    checkpoint_every: int = 1000
    seed: int = 42
    eval_seed: int = 42

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0
        assert (self.d_model // self.num_heads) % 2 == 0