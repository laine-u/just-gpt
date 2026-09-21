from pathlib import Path
from itertools import islice
from contextlib import closing
from collections.abc import Iterable, Iterator
from tqdm import tqdm

import os
import json
import hashlib
import tiktoken
import numpy as np
import pyarrow.parquet as pq


# ==================================================
# splits
# ==================================================

def iter_split_documents(
    paths: Iterable[Path],
    split: str,
    *,
    batch_size: int = 1024,
    use_hash_split: bool = False,
    valid_ratio: float = 0.005,
    split_seed: int = 42,
    max_source_documents: int | None = None,
    max_documents: int | None = None
) -> Iterator[str]:
    with closing(iter_documents(paths, batch_size=batch_size)) as documents:
        selected = islice(documents, max_source_documents)

        if use_hash_split:
            selected = filter_by_split(
                selected,
                split,
                valid_ratio=valid_ratio,
                split_seed=split_seed
            )

        yield from islice(selected, max_documents)

def filter_by_split(
    documents: Iterable[str],
    split: str,
    *,
    valid_ratio: float = 0.005,
    split_seed: int = 42
) -> Iterator[str]:
    if split not in {"train", "valid"}:
        raise ValueError(f"Unknown split: {split}")
    
    if not 0 < valid_ratio < 1:
        raise ValueError(f"valid_ratio must be between 0 and 1")
    
    for text in documents:
        actual_split = get_document_split(
            text,
            valid_ratio=valid_ratio,
            split_seed=split_seed
        )

        if actual_split == split:
            yield text

def get_document_split(
    text: str,
    *,
    valid_ratio: float = 0.005,
    split_seed: int = 42,
) -> str:
    if not 0 < valid_ratio < 1:
        raise ValueError(f"valid_ratio must be between 0 and 1")
    
    payload = f"{split_seed}\0".encode("utf-8") + text.encode("utf-8")
    digest = hashlib.sha256(payload).digest()

    value = int.from_bytes(digest[:8], byteorder="big")
    threshold = int(valid_ratio * (1 << 64))
    return "valid" if value < threshold else "train"

# ==================================================
# process
# ==================================================

def iter_documents(paths: Iterable[Path], batch_size: int = 1024) -> Iterator[str]:
    """
    Iterate all parquet files at given paths.
    """
    for path in paths:
        parquet = pq.ParquetFile(path)

        try:
            for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
                texts = batch.column("text").to_pylist()

                for text in texts:
                    if text is None:
                        continue

                    if not isinstance(text, str):
                        raise TypeError(f"Expected a string in {path}, got {type(text)}")
                    
                    if not text.strip():
                        continue

                    yield text
        finally:
            parquet.close()

def write_tokens(
    documents: Iterable[str],
    output_path: Path,
    enc: tiktoken.Encoding,
    flush_tokens: int = 1_000_000,
):
    dtype = np.dtype("<u2")

    if enc.max_token_value > np.iinfo(dtype).max:
        raise ValueError("Tokenizer token IDs exceed uint16 capacity")
    
    if flush_tokens <= 0:
        raise ValueError(f"flush_tokens must be positive")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    buffer = []
    num_documents = 0
    num_tokens = 0

    with tmp_path.open("wb") as f:
        for text in documents:
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)

            buffer.extend(ids)
            num_documents += 1
            num_tokens += len(ids)

            if len(buffer) >= flush_tokens:
                np.asarray(buffer, dtype=dtype).tofile(f)
                buffer.clear()

        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(f)

    if num_documents == 0:
        raise ValueError(f"No documents were processed. Strange :/")

    verify_file(
        tmp_path,
        expected_tokens=num_tokens,
        expected_documents=num_documents,
        vocab_size=enc.n_vocab,
        eot_token=enc.eot_token
    )

    os.replace(tmp_path, output_path)

    return {
        "num_documents": num_documents,
        "num_tokens": num_tokens,
        "num_bytes": num_tokens * dtype.itemsize
    }

def verify_file(
    path: Path,
    *,
    expected_tokens: int,
    expected_documents: int,
    vocab_size: int,
    eot_token: int,
    chunk_tokens: int = 1_000_000,
):
    path = Path(path)
    dtype = np.dtype("<u2")

    if chunk_tokens <=0 or expected_tokens <= 0:
        raise ValueError(f"chunk_tokesn and expected_tokens must be positive")
    
    expected_bytes = expected_tokens * dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"File size does not match expected token count")
    
    tokens = np.memmap(path, dtype=dtype, mode="r")

    checked_tokens = 0
    eot_count = 0

    for start in tqdm(
        range(0, len(tokens), chunk_tokens),
        desc=f"Checking {path.name}",
        unit="chunk",
        mininterval=0.5,
    ):
        chunk = tokens[start: start + chunk_tokens]

        if int(chunk.min() < 0 or int(chunk.max()) >= vocab_size):
            raise ValueError(f"Invalid token ID in range [{start}, {start + len(chunk)}]")
        
        eot_count += int(np.count_nonzero(chunk == eot_token))
        checked_tokens += len(chunk)

    if checked_tokens != expected_tokens:
        raise ValueError(f"Checked token count mismatch")
    
    if eot_count != expected_documents:
        raise ValueError(f"Expedcted {expected_documents} EOTs, found {eot_count}")
    
    if int(tokens[-1] != eot_token):
        raise ValueError(f"Last token must be EOT")

def verify_tokenization(
    token_path,
    documents: Iterable[str],
    enc: tiktoken.Encoding,
):
    actual = np.memmap(token_path, dtype="<u2", mode="r")
    offset = 0

    for text in documents:
        expected_ids = enc.encode_ordinary(text)
        expected_ids.append(enc.eot_token)

        expected = np.asarray(expected_ids, dtype="<u2")
        end = offset + len(expected)

        if not np.array_equal(actual[offset:end], expected):
            raise ValueError(f"Tokenization mismatch in range [{offset}: {end}]")
        
        offset = end

    if offset != len(actual):
        raise ValueError(f"Token file data mismatched")

def save_metadata(
    enc,
    split_paths,
    stats,
    output_dir: Path,
    dataset: str,
    max_documents: int | None,
    max_source_documents: int | None,
    split_config: dict
):
    meta = {
        "format_version": 1,
        "dataset": dataset,
        "tokenizer": enc.name,
        "vocab_size": enc.n_vocab,
        "eot_token": enc.eot_token,
        "dtype": "<u2",
        "max_source_documents": max_source_documents,
        "max_documents_per_split": max_documents,
        "source_files": {
            split: [str(path) for path in paths]
            for split, paths in split_paths.items()
        },
        "splits": {
            split: {
                "file": f"{split}.bin",
                **info,
            }
            for split, info in stats.items()
        },
        "split_config": split_config,
    }

    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
