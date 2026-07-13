"""Stage 2 entrypoint: controlled hard-example weighting."""

try:
    from train_stage import wrapper_main
except ImportError:
    from upgrade_models.train_stage import wrapper_main


if __name__ == "__main__":
    wrapper_main("stage2")
