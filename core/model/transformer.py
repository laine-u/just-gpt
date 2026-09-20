import torch
import torch.nn as nn


from .activation import SwiGLU
from .attention import MHA, MHAWithRoPE
from .neural import Linear, Embedding, RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int,
        max_seq_len: int,
        device: torch.device = None
    ):
        """
        Args:
            d_model: Dimensionality of the Transformer block inputs.
            num_heads: Number of heads to use in multi-head self-attention.
            d_ff: Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.rmsnorm1 = RMSNorm(d_model, device=device)
        self.mha_with_rope = MHAWithRoPE(d_model, num_heads, theta, max_seq_len, device=device)

        self.rmsnorm2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(0, x.shape[-2], device=x.device)

        # ----- layer 1 : attention -----
        residual_x = x
        x = residual_x + self.mha_with_rope(self.rmsnorm1(x), token_positions)

        # ----- layer 2 : SwiGLU -----
        residual_x = x
        x = x + self.ffn(self.rmsnorm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        d_ff: int,
        num_layers: int = 4,
        num_heads: int = 6,
        theta: float = 10000.0,
        device: torch.device = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.tok_emb = Embedding(vocab_size, d_model, device=device)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device)
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if x.shape[-1] > self.context_length:
            raise ValueError("Input exceeds context_length")
        out = self.tok_emb(x)

        for tbk in self.transformer_blocks:
            out = tbk(out)

        out = self.ln_final(out)
        out = self.lm_head(out)
        return out