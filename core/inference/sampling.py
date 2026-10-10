import math
import torch


def validate_sampling(
    temperature: float,
    top_k: int,
    top_p: float,
):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and > 0, but got {temperature}")
    
    if type(top_k) is not int or top_k < 0:
        raise ValueError(f"top_k must be >= 0, but got {top_k}")
    
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError(f"top_p must be finite and in (0, 1], but got {top_p}")


def select_next_token(
    last_logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None
) -> torch.Tensor:
    """
    Args:
        logits: [batch_size, vocab_size]

    Return:
        [batch_size, 1]
    """
    if not do_sample:
        return last_logits.argmax(dim=-1, keepdim=True)

    scores = last_logits.float()
    scores = scores - scores.amax(dim=-1, keepdim=True)
    scores = scores / temperature

    if top_k > 0:
        k = min(top_k, scores.shape[-1])
        top_values, top_indices = torch.topk(scores, k, dim=-1)

        filtered= torch.full_like(scores, -torch.inf)
        filtered.scatter_(-1, top_indices, top_values)
        scores = filtered

    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        remove = torch.zeros_like(sorted_scores, dtype=torch.bool)
        remove[..., 1:] = cumulative_probs[..., : -1] >= top_p
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)

        scores = torch.full_like(scores, -torch.inf)
        scores.scatter_(-1, sorted_indices, sorted_scores)

    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)