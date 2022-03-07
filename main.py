import argparse
import logging
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from prompt_toolkit.shortcuts import confirm

from preprocessing import DATASET_DIR, OUTPUT_DIR
from preprocessing.data_pipeline import process_default_pool


logger = logging.getLogger(__name__)

SEED = None
SEED = SEED if SEED else int(random.random() * 10000)
random.seed(SEED)
np.random.seed(SEED)


def reset_directory(path: Path):
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Path '{path.as_posix()}' is not a directory")
        answer = confirm(f"Delete output directory {path}")
        if answer:
            shutil.rmtree(path.as_posix())
        else:
            sys.exit(1)
    else:
        path.mkdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_false",
        help=f"Overwrite duplicate file names in directory {OUTPUT_DIR}",
    )
    parser.add_argument(
        "-r", "--reset", action="store_true", help=f"Recreate directory {OUTPUT_DIR}"
    )
    parser.add_argument(
        "-s", "--chunk-size", default=128, type=int, help="Image split size"
    )
    parser.add_argument(
        "-log",
        "--log",
        default="info",
        help=("Provide logging level. " "Example --log debug', default='info'"),
    )

    options = parser.parse_args()
    level = logging.getLevelName(options.log.upper())
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {options.log}")

    logging.basicConfig(
        format=(
            "%(asctime)s,"
            "%(msecs)d "
            "%(levelname)-8s "
            "[%(filename)s:%(lineno)d] "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=level,
    )

    logger.critical("SEED: %d", SEED)

    input_jpgs = list(DATASET_DIR.glob("*.jpg"))

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(exist_ok=True)
    elif options.reset:
        reset_directory(OUTPUT_DIR)

    start = time.perf_counter()

    process_default_pool(input_jpgs, OUTPUT_DIR, options.chunk_size, options.overwrite)

    logger.debug(
        "Processed %d images in %d seconds",
        len(input_jpgs),
        round(time.perf_counter() - start, 2),
    )
