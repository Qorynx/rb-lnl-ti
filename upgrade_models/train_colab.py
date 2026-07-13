"""Stage 1 entrypoint. Run from the repository root or directly by path."""

try:
    from train_stage import wrapper_main
except ImportError:
    from upgrade_models.train_stage import wrapper_main


if __name__ == "__main__":
    wrapper_main("stage1")
