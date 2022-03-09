import logging
import re
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm

from preprocessing import EDGE_DIR, OUTPUT_DIR
from preprocessing.custom_pool import CustomPool
from preprocessing.formatter import split_image
from preprocessing.watermarker import Watermarker


logger = logging.getLogger(__name__)


def process_image(args: Tuple[Path, Path, bool]):
    image_path = args[0]
    output_dir = args[1]
    overwrite = args[2]

    if output_dir.joinpath(image_path.name).exists and not overwrite:
        return

    try:
        watermarker = Watermarker(image_path)
    except ValueError:
        return

    watermarker.add_default()
    watermarker.save(output_dir)


def process_default_pool(
    images: List[Path], output_dir: Path, overwrite=True, processes=16
):
    output_dir.mkdir(exist_ok=True)

    with Pool(processes) as pool:
        with tqdm(total=len(images)) as pbar:
            for _ in pool.imap_unordered(
                process_image,
                ((image, output_dir, overwrite) for image in images),
            ):
                pbar.update()
        pool.close()
        pool.join()


def process_custom_pool(
    images: List[Path], output_dir: Path, overwrite=True, processes=16
):
    output_dir.mkdir(exist_ok=True)

    pool = CustomPool(processes)

    for i in tqdm(range(len(images))):
        pool.map(
            process_image,
            args=(
                (
                    images[i],
                    output_dir,
                    overwrite,
                ),
            ),
        )

    while pool.is_running():
        time.sleep(0.1)
        continue


if __name__ == "__main__":
    edge_paths = list(EDGE_DIR.glob("*.jpg"))
    process_image((edge_paths[0], OUTPUT_DIR, True))
    process_default_pool(edge_paths, output_dir=OUTPUT_DIR)
