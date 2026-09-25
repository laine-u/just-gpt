import torch


class WindowSampler:
    def __init__(
        self,
        num_windows: int,
        batch_size: int,
        seed: int = 42
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive")
        
        if num_windows < batch_size:
            raise ValueError(f"Not enough windows for one full batch")
        
        self.num_windows = num_windows
        self.batch_size = batch_size
        self.seed = seed

        self.epoch = 0
        self.position = 0
        self.order = self._make_order()

    def _make_order(self) -> torch.Tensor:
        """
        Make order of windows based on seed and current epoch.
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)

        return torch.randperm(
            self.num_windows,
            generator=generator,
            device="cpu"
        )

    def next_batch(self) -> list[int]:
        """
        Get next batch of window_ids.
        """
        # starts a new round if remaining windows is less than one batch
        if self.position + self.batch_size > self.num_windows:
            self.epoch += 1
            self.position = 0
            self.order = self._make_order()

        end = self.position + self.batch_size
        window_ids = self.order[self.position:end].tolist()
        self.position = end
        return window_ids

    def state_dict(self) ->  dict:
        return {
            "num_windows": self.num_windows,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position
        }

    def load_state_dict(self, state: dict):
        for name in ("num_windows", "batch_size", "seed"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Sampler {name} does not match")
            
        epoch = state["epoch"]
        position = state["position"]

        usable_windows = (self.num_windows // self.batch_size) * self.batch_size

        if epoch < 0:
            raise ValueError(f"epoch must be non-negatve")

        if not (
            0 <= position <= usable_windows
            and position % self.batch_size == 0
        ):
            raise ValueError(f"Invalid sampler position")
        
        self.epoch = epoch
        self.position = position
        self.order = self._make_order()
