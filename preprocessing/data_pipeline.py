import logging
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from preprocessing import DATASET_DIR, EDGE_DIR
from preprocessing.custom_pool import CustomPool
from preprocessing.formatter import split_image
from preprocessing.watermarker import Watermarker


logger = logging.getLogger(__name__)


def process_image(args: Tuple[Path, Path, bool]):
    image_path = args[0]
    output_dir = args[1]
    overwrite = args[2]

    try:
        watermarker = Watermarker(image_path)
    except ValueError:
        return

    watermarker.add_default()

    inputs = split_image(watermarker.image)
    outputs = split_image(image_path.as_posix())

    # get new file names for each chunk
    file_names = [
        f"{image_path.stem}_{str(i)}{image_path.suffix}" for i in range(len(inputs))
    ]

    # save files
    for file_name, input_image, output_image in zip(file_names, inputs, outputs):
        input_path = output_dir.joinpath("input", file_name)
        if not input_path.exists() or overwrite:
            Image.fromarray(input_image).save(input_path)

        output_path = output_dir.joinpath("output", file_name)
        if not output_path.exists() or overwrite:
            Image.fromarray(output_image).save(output_path)


def process_default_pool(
    images: List[Path], output_dir: Path, overwrite=True, processes=16
):
    output_dir.joinpath("input").mkdir(exist_ok=True)
    output_dir.joinpath("output").mkdir(exist_ok=True)

    with Pool(processes) as pool:
        with tqdm(total=len(images)) as pbar:
            for _ in pool.imap_unordered(
                process_image, ((image, output_dir, overwrite) for image in images)
            ):
                pbar.update()
        pool.close()
        pool.join()


def process_custom_pool(
    images: List[Path], output_dir: Path, overwrite=True, processes=16
):
    output_dir.joinpath("input").mkdir(exist_ok=True)
    output_dir.joinpath("output").mkdir(exist_ok=True)

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
    process_default_pool(list(EDGE_DIR.glob("*.jpg")), output_dir=DATASET_DIR)
