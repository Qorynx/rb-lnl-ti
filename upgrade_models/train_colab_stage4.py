"""Stage 4 entrypoint: clean fine-tuning and final official-test evaluation."""

try:
    from train_stage import wrapper_main
except ImportError:
    from upgrade_models.train_stage import wrapper_main


if __name__ == "__main__":
    wrapper_main("stage4")
