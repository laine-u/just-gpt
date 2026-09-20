from experiments import *


def main():
    cfg = build_cfg(
        d_model=64,
        num_heads=4,
        d_ff=128,
    )
    run_exp(cfg)


if __name__ == "__main__":
    main()