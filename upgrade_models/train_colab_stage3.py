"""Stage 3 entrypoint: gated residual correction with base preservation."""

try:
    from train_stage import wrapper_main
except ImportError:
    from upgrade_models.train_stage import wrapper_main


if __name__ == "__main__":
    wrapper_main("stage3")
