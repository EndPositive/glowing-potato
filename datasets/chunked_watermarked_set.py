import gc
import math
import random
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.io import read_image

from datasets.transform import TRANSFORM
from preprocessing import DATASET_DIR, OUTPUT_DIR


class DataSetType(Enum):
    Training = 0
    Validation = 1
    Test = 2


class ChunkedWatermarkedSet(VisionDataset):
    data_set_names: List[Path] = []

    watermarked_dir = OUTPUT_DIR
    original_dir = DATASET_DIR

    def __init__(
        self,
        data_set_type: DataSetType,
        transforms: Callable = TRANSFORM,
        split_size=(0.7, 0.1, 0.2),
        watermarked_dir=OUTPUT_DIR,
        original_dir=DATASET_DIR,
    ) -> None:
        super().__init__(DATASET_DIR, transforms=transforms)

        self.watermarked_dir = watermarked_dir
        self.original_dir = original_dir

        if (split_sum := sum(split_size)) != 1:
            raise ValueError(f"Split size should sum to 1 {split_size} -> {split_sum}")

        # Get all watermarked images that still have existing originals
        data_set_paths = [
            jpg_in_watermarked_dataset.name
            for jpg_in_watermarked_dataset in OUTPUT_DIR.glob("*.jpg")
            if DATASET_DIR.joinpath(jpg_in_watermarked_dataset.name).exists
        ]

        training_set, validation_set, testing_set = np.split(
            data_set_paths,
            [
                int(len(data_set_paths) * split_size[0]),
                int(len(data_set_paths) * (split_size[0] + split_size[1])),
            ],
        )

        if data_set_type == DataSetType.Training:
            self.data_set_names = training_set
        elif data_set_type == DataSetType.Validation:
            self.data_set_names = validation_set
        elif data_set_type == DataSetType.Test:
            self.data_set_names = testing_set
        else:
            raise ValueError(f"Unknown data set type {data_set_type}")

        gc.collect()

    def __get_watermarked_path(self, name):
        return self.watermarked_dir.joinpath(name)

    def __get_original_path(self, name):
        return self.original_dir.joinpath(name)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Read watermarked and original images as Tensors
        watermarked = read_image(
            self.__get_watermarked_path(self.data_set_names[index]).as_posix()
        )
        original = read_image(
            self.__get_original_path(self.data_set_names[index]).as_posix()
        )

        # Set new seed so that watermarked and original transforms both have the same randomness
        seed = math.floor(random.random() * 100000)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        watermarked = self.transforms(watermarked)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        original = self.transforms(original)

        return watermarked, original

    def __len__(self) -> int:
        return len(self.data_set_names)


if __name__ == "__main__":
    dataset = ChunkedWatermarkedSet(data_set_type=DataSetType.Training)
    watermarked_test, original_test = dataset[0]
    to_pil_image(watermarked_test).show()
    to_pil_image(original_test).show()
