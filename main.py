import argparse
from pathlib import Path

from experiments import build_cfg, run_exp
from core.misc.utils import positive_int, positive_float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training a mini GPT over preprocessed token files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("Data and run")
    data.add_argument("--name", default="TinyStories")
    data.add_argument("--dataset", default="tinystories")
    data.add_argument("--data-dir", default="data/processed/tinystories-gpt2")
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--eval-seed", type=int, default=42)
    data.add_argument("--resume-from", type=Path, default=None, help="Run directory containing checkpoints")

    model = parser.add_argument_group("Model")
    model.add_argument("--num-layers", type=positive_int, default=2)
    model.add_argument("--d-model", type=positive_int, default=256)
    model.add_argument("--num-heads", type=positive_int, default=8)
    model.add_argument("--d-ff", type=positive_int, default=768)
    model.add_argument("--theta", type=positive_float, default=10000.0)

    training = parser.add_argument_group("Training")
    training.add_argument("--token-budget", type=positive_int, default=1_000_000)
    training.add_argument("--batch-size", type=positive_int, default=4)
    training.add_argument("--context-length", type=positive_int, default=128)
    training.add_argument("--lr", type=positive_float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), metavar=("BETA1", "BETA2"))
    training.add_argument("--eps", type=positive_float, default=1e-8)

    schedule = parser.add_argument_group("Learning rate schedule")
    schedule.add_argument("--schedule", choices=["constant", "cosine", "wsd"], default="wsd")
    schedule.add_argument("--warmup-ratio", type=float, default=0.02)
    schedule.add_argument("--decay-ratio", type=float, default=0.15)
    schedule.add_argument("--min-lr-ratio", type=float, default=0.1)

    logging = parser.add_argument_group("Evaluation and logging")
    logging.add_argument("--num-eval-steps", type=positive_int, default=200)
    logging.add_argument("--log-every", type=int, default=100)
    logging.add_argument("--eval-every", type=int, default=500)
    logging.add_argument("--checkpoint-every", type=int, default=1000)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.d_model % args.num_heads != 0:
        parser.error("--d-model must be divisible by --num-heads")

    if (args.d_model // args.num_heads) % 2 != 0:
        parser.error("RoPE requires an even head dimension")

    options = vars(args).copy()
    resume_from = options.pop("resume_from")
    options["betas"] = tuple(options["betas"])

    try:
        cfg = build_cfg(**options)
    except ValueError as exc:
        parser.error(str(exc))

    run_exp(cfg, resume_from=resume_from)




if __name__ == "__main__":
    main()