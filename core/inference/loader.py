from pathlib import Path

import tiktoken
import torch

from core.model import TransformerLM


def load_model(
    checkpoint_path: str | Path,
    device: str = "auto"
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    target_device = torch.device(device)

    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but not available")
    
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    cfg = checkpoint["config"]
    state = checkpoint["model"]

    tokenizer = tiktoken.get_encoding("gpt2")
    model = TransformerLM(
        vocab_size=state["tok_emb.W"].shape[0],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        theta=cfg["theta"]
    )
    model.load_state_dict(state, strict=True)
    model.to(target_device)
    model.eval()

    checkpoint_info = {
        "path": str(Path(checkpoint_path).resolve()),
        "step": checkpoint["step"],
        "config": cfg
    }
    # load_state_dict do copy operation, need manual free checkpoint
    del state, checkpoint
    return model, tokenizer, checkpoint_info
