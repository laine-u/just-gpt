from experiments import *


def main():
    cfg = build_cfg(
        name="TinyStories",
        dataset="tinystories",
        data_dir="data/processed/tinystories-gpt2",
        d_model=256,
        num_heads=8,
        d_ff=768,
    )
    run_exp(cfg)


if __name__ == "__main__":
    main()