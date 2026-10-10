import torch
from typing import List, Tuple

from core.model import TransformerLM

from .sampling import validate_sampling, select_next_token


@torch.no_grad
def generate_tokens(
    model: TransformerLM,
    prompt_ids: List[int],
    max_new_tokens: int = 100,
    eos_token_id: int | None = None,
    *,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 42
) -> Tuple[List[int], str]:
    if not prompt_ids:
        raise ValueError("Woah! Prompt must contain at least one token")
    
    if max_new_tokens < 0:
        raise ValueError(f"Woah! max_new_tokens must be >= 0, but got {max_new_tokens}")
    
    if any(
        token_id < 0 or token_id >= model.vocab_size
        for token_id in prompt_ids
    ):
        raise ValueError("Woah! Prompt contains invalid token ID, but I won't tell you which or where </")
    
    if eos_token_id is not None:
        if not 0 <= eos_token_id < model.vocab_size:
            raise ValueError(f"Woah! Invalid EOS token ID: {eos_token_id}")
        
    validate_sampling(temperature, top_k, top_p)

    model.eval()
    device = next(model.parameters()).device

    generator = None
    if do_sample:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    tokens = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    generated_ids = []
    finish_reason = "length"

    for _ in range(max_new_tokens):
        inputs = tokens[:, -model.context_length:]

        logits = model(inputs)

        next_token = select_next_token(
            logits[:, -1, :],
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator
        )
        tokens = torch.cat((tokens, next_token), dim=1)

        token_id = next_token.item()
        generated_ids.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            finish_reason = "eos"
            break

    return generated_ids, finish_reason
        
    