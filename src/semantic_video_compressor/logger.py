import logging, sys, pathlib, datetime, os

def define_logger(run_dir: pathlib.Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)
