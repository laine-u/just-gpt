import torch
import torch.nn as nn
from einops import einsum, rearrange


from .neural import softmax, RoPE


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    d_k = Q.shape[-1]

    attn_scores = einsum(Q, K, "... n d_k,... m d_k -> ... n m") / (d_k**0.5)
    attn_scores.masked_fill_(~mask, -torch.inf)
    attn_weights = softmax(attn_scores)
    context_vec = einsum(attn_weights, V, "... n m, ... m d_v -> ... n d_v")
    return context_vec


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Parameters:
            d_model: Dimensionality of the Transformer block inputs.
            num_heads: Number of heads to use in multi-head self-attention
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model % num_heads should be 0, instead of {d_model % num_heads}"
        self.d_model = d_model
        self.h = num_heads

        self.head_dim = d_model // num_heads
        self.dk = d_model // num_heads
        self.dv = d_model // num_heads

        self.q_proj = nn.Parameter(torch.zeros((self.h * self.dk, d_model)))
        self.k_proj = nn.Parameter(torch.zeros((self.h * self.dk, d_model)))
        self.v_proj = nn.Parameter(torch.zeros((self.h * self.dv, d_model)))
        self.o_proj = nn.Parameter(torch.zeros((d_model, self.h * self.dv)))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_normal_(self.q_proj)
        nn.init.xavier_normal_(self.k_proj)
        nn.init.xavier_normal_(self.v_proj)
        nn.init.xavier_normal_(self.o_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape of (..., seq_len, d_model)
        """
        seq_len = x.shape[-2]

        Q = einsum(x, self.q_proj, "... seq d_model, hdk d_model -> ... seq hdk")
        K = einsum(x, self.k_proj, "... seq d_model, hdk d_model -> ... seq hdk")
        V = einsum(x, self.v_proj, "... seq d_model, hdv d_model -> ... seq hdv")

        Q = rearrange(Q, "... seq (h dk) -> ... h seq dk", h=self.h, dk=self.dk)
        K = rearrange(K, "... seq (h dk) -> ... h seq dk", h=self.h, dk=self.dk)
        V = rearrange(V, "... seq (h dv) -> ... h seq dv", h=self.h, dv=self.dv)

        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        context_vec = scaled_dot_product_attention(Q, K, V, mask)

        context_vec = rearrange(context_vec, "... h seq dv -> ... seq (h dv)", h=self.h, dv=self.dv)
        return einsum(context_vec, self.o_proj, "... seq hdv, d_model hdv -> ... seq d_model")
    

class MHAWithRoPE(nn.Module):
    """
    MultiHeadAttention with RoPE, also combined Q,K,V and calculate them all at once.
    """
    def __init__(self, d_model: int, num_heads: int, theta: int, max_seq_len: int, device: torch.device = None):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model % num_heads should be 0, instead of {d_model % num_heads}"
        self.d_model = d_model
        self.h = num_heads
        self.device = device
        
        self.head_dim = d_model // num_heads
        self.dk = d_model // num_heads
        self.dv = d_model // num_heads
        assert self.dk == self.dv, "d_k must be equal to d_v"

        self.W_qkv = nn.Parameter(torch.zeros((self.h * self.dk * 3, d_model), device=device))
        self.W_o = nn.Parameter(torch.zeros((d_model, self.h * self.dv), device=device))

        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device)

        self._init_parameters()

    def _init_parameters(self):
        for weight in self.W_qkv.chunk(3, dim=0):
            nn.init.xavier_normal_(weight)

        nn.init.xavier_normal_(self.W_o)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        seq_len = x.shape[-2]

        QKV = einsum(x, self.W_qkv, "... seq d_model, hdqkv d_model -> ... seq hdqkv")
        Q, K, V = torch.chunk(QKV, 3, dim=-1)

        Q = rearrange(Q, "... seq (h dk) -> ... h seq dk", h=self.h, dk=self.dk)
        K = rearrange(K, "... seq (h dk) -> ... h seq dk", h=self.h, dk=self.dk)
        V = rearrange(V, "... seq (h dv) -> ... h seq dv", h=self.h, dv=self.dv)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        context_vec = scaled_dot_product_attention(Q, K, V, mask)

        context_vec = rearrange(context_vec, "... h seq dv -> ... seq (h dv)", h=self.h, dv=self.dv)
        return einsum(context_vec, self.W_o, "... seq hdv, d_model hdv -> ... seq d_model")