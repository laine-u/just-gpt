import argparse
import tiktoken

from pathlib import Path
from contextlib import closing
from tqdm import tqdm

from core.data.preprocessing import (
    iter_split_documents,
    write_tokens,
    save_metadata,
    verify_tokenization
)

# ==================================================
# Let there be parser
# ==================================================

def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Expected an integer") from None

    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Parquet text into token files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--raw-dir", type=Path, required=True,
        help="Directory containing the source Parquet files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for token files and metadata"
    )
    parser.add_argument(
        "--dataset", default="tinystories",
        help="Dataset name"
    )
    parser.add_argument(
        "--train-pattern", default="train-*.parquet",
        help="Training file pattern relative to raw-dir"
    )
    parser.add_argument(
        "--valid-pattern", default="validation-*.parquet",
        help="Validating file pattern relative to raw-dir"
    )
    parser.add_argument(
        "--tokenizer", default="gpt2",
        help="tiktoken encoding name"
    )
    parser.add_argument(
        "--max-source-documents", type=positive_int, default=None,
        help="Maximum source documents before split filtering"
    )
    parser.add_argument(
        "--max-documents", type=positive_int, default=None,
        help="Maximum valid documents per split; omit to process all"
    )
    parser.add_argument(
        "--batch-size", type=positive_int, default=1024,
        help="Parquet rows per read batch"
    )
    parser.add_argument(
        "--flush-tokens", type=positive_int, default=1_000_000,
        help="Token buffer threshold for writting"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Re-tokenize source documents and compare with output"
    )
    parser.add_argument(
        "--valid-ratio", type=float, default=0.005,
        help="validation document ratio for hash splitting"
    )
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Seed for deterministic document splitting"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --------------------------------------------------
    # check arguments
    # --------------------------------------------------

    if not args.raw_dir.is_dir():
        parser.error(f"Raw directory does not exist: {args.raw_dir}. Did you forget to download raw files first?")

    use_hash_split = args.dataset == "owt"
    if use_hash_split:
        if not 0 < args.valid_ratio < 1:
            parser.error("--valid-ratio must be between 0 and 1")

        source_paths = sorted(args.raw_dir.glob(args.train_pattern))
        split_paths = {
            "train": source_paths,
            "valid": source_paths,
        }
        split_config = {
            "method": "sha256_text",
            "seed": args.split_seed,
            "valid_ratio": args.valid_ratio,
        }
    else:
        split_paths = {
            "train": sorted(args.raw_dir.glob(args.train_pattern)),
            "valid": sorted(args.raw_dir.glob(args.valid_pattern))
        }
        split_config = {"method": "provided"}

    for split, paths in split_paths.items():
        if not paths:
            parser.error(f"Missing Parquet files for split: {split}")

        print(f"{split}: {len(paths)} Parquet files")

    document_options = {
        "batch_size": args.batch_size,
        "use_hash_split": use_hash_split,
        "valid_ratio": args.valid_ratio,
        "split_seed": args.split_seed,
        "max_source_documents": args.max_source_documents,
        "max_documents": args.max_documents,
    }

    output_dir = args.output_dir

    # --------------------------------------------------
    # tokenize
    # --------------------------------------------------

    enc = tiktoken.get_encoding(args.tokenizer)

    stats = {}
    for split, paths in split_paths.items():
        with closing(
            iter_split_documents(paths, split, **document_options)
        ) as documents:
            with tqdm(
                documents,
                total=args.max_documents,
                desc=f"Tokenizing {split}",
                unit="doc",
                mininterval=0.5,
            ) as progress:
                stats[split] = write_tokens(
                    documents=progress,
                    output_path=output_dir / f"{split}.bin",
                    enc=enc,
                    flush_tokens=args.flush_tokens,
                )
        print(split, stats[split])

    # --------------------------------------------------
    # verify (by re-tokenize)
    # --------------------------------------------------

    if args.verify:
        for split, paths in split_paths.items():
            with closing(
                iter_split_documents(paths, split, **document_options)
            ) as documents:
                with tqdm(
                    documents,
                    total=stats[split]["num_documents"],
                    desc=f"Verifying tokenization {split}",
                    unit="doc",
                    mininterval=0.5,
                ) as progress:
                    verify_tokenization(
                        token_path=output_dir / f"{split}.bin",
                        documents=progress,
                        enc=enc
                    )

    # --------------------------------------------------
    # save metadata
    # --------------------------------------------------

    save_metadata(
        enc,
        split_paths,
        stats,
        output_dir,
        args.dataset,
        args.max_documents,
        args.max_source_documents,
        split_config,
    )


if __name__ == "__main__":
    main()