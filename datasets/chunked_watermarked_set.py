import gc, time, threading
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Union
from copy import deepcopy

import numpy as np
import torch
import torchvision.transforms
from torchvision.datasets import VisionDataset
from torchvision.io import read_image

import datasets.transform
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
        device: torch.device,
        transforms: Callable,
        data_set_type: DataSetType = DataSetType.Training,
        split_size=(0.7, 0.1, 0.2),
        watermarked_dir=OUTPUT_DIR,
        original_dir=DATASET_DIR,
        include_fn=False,  # used for precomputing dataset features
    ) -> None:
        super().__init__(DATASET_DIR, transforms=transforms)

        self.watermarked_dir = watermarked_dir
        self.original_dir = original_dir

        self.device = device

        if len(split_size) == 2:
            split_size = (
                split_size[0],
                split_size[1],
                1 - split_size[0] - split_size[1]
            )

        if (split_sum := sum(split_size)) != 1:
            raise ValueError(f"Split size should sum to 1 {split_size} -> {split_sum}")

        # Get all watermarked images that still have existing originals
        data_set_paths = [
            jpg_in_watermarked_dataset.name
            for jpg_in_watermarked_dataset in original_dir.glob("*.jpg")
            if watermarked_dir.joinpath(jpg_in_watermarked_dataset.name).exists()
        ]

        self.training_set, self.validation_set, self.testing_set = np.split(
            data_set_paths,
            [
                int(len(data_set_paths) * split_size[0]),
                int(len(data_set_paths) * (split_size[0] + split_size[1])),
            ],
        )

        if data_set_type == DataSetType.Validation:
            self.data_set_names = self.validation_set
        elif data_set_type == DataSetType.Test:
            self.data_set_names = self.testing_set
        else:
            self.data_set_names = self.training_set

        self.include_fn = include_fn
        self.preloaded_data = {}
        self.preload_buffer = 20
        gc.collect()

    def train(self):
        other = deepcopy(self)
        other.data_set_names = other.training_set
        return other

    def validate(self):
        other = deepcopy(self)
        other.data_set_names = other.validation_set
        return other

    def test(self):
        other = deepcopy(self)
        other.data_set_names = other.testing_set
        return other

    def __get_watermarked_path(self, name):
        return self.watermarked_dir.joinpath(name)

    def __get_original_path(self, name):
        return self.original_dir.joinpath(name)

    def __load_images(self, index):
        watermarked = read_image(
            self.__get_watermarked_path(self.data_set_names[index]).as_posix()
        ).to(device=self.device)
        original = read_image(
            self.__get_original_path(self.data_set_names[index]).as_posix()
        ).to(device=self.device)
        return self.transforms(watermarked, original)

    def __preload(self, start_index, add=True):
        if start_index in self.preloaded_data: del self.preloaded_data[start_index]
        index = start_index+self.preload_buffer-1 if add else start_index
        self.preloaded_data[index] = None
        self.preloaded_data[index] = self.__load_images(index)


    def __getitem__(self, index: int) -> Union[tuple[Any, Any, Path], tuple[Any, Any]]:
        # Read watermarked and original images as Tensors
        if index in self.preloaded_data: 
            time_waited = 0
            while (data := self.preloaded_data[index]) is None: 
                print("\rwaiting", end="")
                time_waited += 0.001
                if time_waited > 3: 
                    print(f'\nExceeded waiting times')
                    del self.preloaded_data[index]
                    data = self.__load_images(index)
                    print(data)
                    break
                time.sleep(0.001)
            watermarked, original = data
            threading.Thread(target=self.__preload, args=(index,)).start()
        else:
            # print(f'\nFresh access needed for {index}')
            watermarked, original = self.__load_images(index)
            if index < 2:
                for i in range(self.preload_buffer):
                    if i == 0: continue
                    self.__preload(index+i, add=False)

        if self.include_fn:
            return watermarked, original, self.data_set_names[index]

        return watermarked, original

    def __len__(self) -> int:
        return len(self.data_set_names)


if __name__ == "__main__":
    watermarked_test, original_test = ChunkedWatermarkedSet(
        data_set_type=DataSetType.Training,
        device='cpu',
        transforms=datasets.transform.TRANSFORM_UNET
    )[0]
    torchvision.transforms.ToPILImage()(watermarked_test).show()
    torchvision.transforms.ToPILImage()(original_test).show()
