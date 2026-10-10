import torch
import torch.nn as nn
from einops import einsum, rearrange


def softmax(x: torch.Tensor, dim: int = -1):
    stable = x - x.max(dim=dim, keepdim=True).values
    numerator = stable.exp()
    denominator = numerator.sum(dim=dim, keepdim=True)
    return numerator / denominator


class Linear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        Perform a Linear transformation.

        Args:
            d_in: final dimention of the input
            d_out: final dimention of the output
            device: device to store parameters on
            dtype: data type of the parameters
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.zeros((d_out, d_in), device=device, dtype=dtype))
        self._init_parameters()

    def _init_parameters(self):
        sigma = (2 / (self.d_in + self.d_out))**0.5
        nn.init.trunc_normal_(self.W, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        Perform Embedding.

        Args:
            vocab_size: size of vocabulary 
            d_model: dimention of the embedding vectors
            device: device to store parameters on
            dtype: data type of the parameters
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.zeros(
            (vocab_size, d_model),
            device=device,
            dtype=dtype)
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.W, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W[x]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Root mean square normalization.

        Args:
            d_model: hidden dimention of the model
            eps: episilon value for numerical stability
            device: device to store parameters on
            dtype: data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.gain = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def set_weights(self, weight: torch.Tensor):
        self.gain.data.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(dtype=torch.float32)

        rms = (x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        rmsnorm = x.div(rms) * self.gain
        return rmsnorm.to(orig_dtype)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Rotary Positional Embedding.

        Args:
            theta: \\Theta value for the RoPE
            d_k: dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be input
            device: Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.d = d_k // 2
        self.max_seq_len = max_seq_len
        self.device = device

        self.register_buffer("cache_sin", torch.zeros((max_seq_len, self.d), device=device), persistent=False)
        self.register_buffer("cache_cos", torch.zeros((max_seq_len, self.d), device=device), persistent=False)

        self._init_parameters()

    def _init_parameters(self):
        pos = torch.arange(0, self.max_seq_len)
        theta = self.theta ** (-2 * torch.arange(0, self.d) / self.d_k)
        angles = torch.outer(pos, theta)

        self.cache_sin.data.copy_(torch.sin(angles))
        self.cache_cos.data.copy_(torch.cos(angles))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        sin = self.cache_sin[token_positions]
        cos = self.cache_cos[token_positions]

        pre_rotate = rearrange(x, "... seq_len (d_k_half two) -> ... seq_len d_k_half two", two=2)
        x_even = pre_rotate[..., 0]
        x_odd = pre_rotate[..., 1]

        x_prime_even = x_even * cos - x_odd * sin
        x_prime_odd = x_even * sin + x_odd * cos

        stacked = torch.stack([x_prime_even, x_prime_odd], dim=-1)
        rotated = rearrange(stacked, "... d_k_half two -> ... (d_k_half two)")
        return rotated