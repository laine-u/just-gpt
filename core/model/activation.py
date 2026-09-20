import torch
import torch.nn as nn
from einops import einsum


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    Combine SiLU and GLU -> SwiGLU.
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Args:
            d_model: Hidden dimention of the model.
            d_ff: Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.xavier_normal_(self.W3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        part1 = silu(einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff"))
        part2 = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        result = einsum(part1 * part2, self.W2,
            "... d_ff, d_model d_ff -> ... d_model"
        )
        return result