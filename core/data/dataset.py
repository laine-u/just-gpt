from pathlib import Path

import numpy as np
import torch


def load_tokens(path: Path, dtype: str, expected_tokens: int) -> np.memmap:
    path = Path(path)
    dtype = np.dtype(dtype)

    if expected_tokens <= 0:
        raise ValueError(f"Token file must not be empty")

    expected_bytes = expected_tokens * dtype.itemsize

    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Token file size mismatch: {path}")

    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        shape=(expected_tokens,)
    )


def get_batch(tokens, batch_size, context_length, *, rng):
    if batch_size <= 0 or context_length <= 0:
        raise ValueError(f"Batch size and context length muse be posivive")

    if len(tokens) < context_length + 1:
        raise ValueError(f"Not enough tokens for one window.")

    starts = torch.randint(
        high=len(tokens) - context_length,
        size=(batch_size,),
        generator=rng,
        device="cpu"
    ).tolist()

    chunks = np.stack([
        tokens[start: start + context_length + 1]
        for start in starts
    ]).astype(np.int64)

    chunks = torch.from_numpy(chunks)
    return chunks[:, :-1], chunks[:, 1:]