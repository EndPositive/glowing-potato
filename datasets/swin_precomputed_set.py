from typing import Tuple

from torch import Tensor
import torch
from torchvision.io import read_image
import numpy as np
from torchvision.datasets import VisionDataset

from datasets.chunked_watermarked_set import DataSetType
from preprocessing import OUTPUT_DIR_PRECOMPUTED, INPUT_DIR_PRECOMPUTED


class SwinPrecomputedSet(VisionDataset):
    def __init__(
            self,
            data_set_type: DataSetType,
            split_size=(0.7, 0.1, 0.2),
            x_dir=INPUT_DIR_PRECOMPUTED,
            y_dir=OUTPUT_DIR_PRECOMPUTED,
            device='cpu'
    ):
        super().__init__(x_dir)
        self.device = device

        def make_name(f, extension):
            return x_dir.joinpath(
                f.name.split('.')[0] + f'_{extension}.npy'
            )

        data_paths = []
        for y in y_dir.glob('*.npy'):
            f_features = make_name(y, 'features')
            f_residual = make_name(y, 'residual')

            if f_features.exists() and f_residual.exists():
                data_paths.append((y, f_features, f_residual))

        training_set, validation_set, testing_set = np.split(
            data_paths,
            [
                int(len(data_paths) * split_size[0]),
                int(len(data_paths) * (split_size[0] + split_size[1])),
            ],
        )

        if data_set_type == DataSetType.Training:
            self.data_paths = training_set
        elif data_set_type == DataSetType.Validation:
            self.data_paths = validation_set
        elif data_set_type == DataSetType.Test:
            self.data_paths = testing_set
        else:
            raise ValueError(f"Unknown data set type {data_set_type}")

    def __getitem__(self, index: int) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        y_hat = np.load(self.data_paths[index][0])
        y_hat = torch.from_numpy(y_hat)

        features = np.load(self.data_paths[index][1])
        features = torch.from_numpy(features)

        residuals = np.load(self.data_paths[index][2])
        residuals = torch.from_numpy(residuals)

        return (
            (features, residuals),
            y_hat
        )

    def __len__(self):
        return len(self.data_paths)
