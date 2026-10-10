import os
import json
from pathlib import Path
from typing import List, Dict


def load_prompts(path: str | Path) -> List[Dict]:
    prompts: List[Dict] = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompt file must contain a non-empty JSON list")
    
    seen = set()

    for index, item in enumerate(prompts):
        prompt_id = item.get("id")
        text = item.get("prompt")
        category = item.get("category", "uncategorized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or invalid prompt: {prompt_id}")

        if not isinstance(category, str):
            raise ValueError(f"Invalid category: {prompt_id}")

        seen.add(prompt_id)

    return prompts


def save_output(data: Dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )
    os.replace(tmp_path, path)