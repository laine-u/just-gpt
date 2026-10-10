
import csv
import argparse

from pathlib import Path

from core.analysis.summarize import collect_summary, summarize_groups
from core.analysis.plot import plot_all


def save_csv(rows, path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze experiment result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = args.input
    rows = collect_summary(input_dir)
    grouped_rows = summarize_groups(rows)

    save_csv(rows, output_dir / "scaling_summary.csv")
    save_csv(grouped_rows, output_dir / "scaling_grouped.csv")

    plot_all(rows, output_dir / "figures")


if __name__ == "__main__":
    main()
