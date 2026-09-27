
import csv
from pathlib import Path
from core.analysis.summarize import collect_summary, summarize_groups
from core.analysis.plot import plot_all


def save_csv(rows, path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {path}")


def main():
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_summary(Path("results"))
    grouped_rows = summarize_groups(rows)

    save_csv(rows, output_dir / "scaling_summary.csv")
    save_csv(grouped_rows, output_dir / "scaling_grouped.csv")

    plot_all(rows, output_dir / "figures")


if __name__ == "__main__":
    main()
