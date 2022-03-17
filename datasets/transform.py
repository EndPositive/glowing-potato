import torch
import torchvision.transforms.functional as TF
import numpy as np
from torch import Tensor
from typing import Tuple


class CropMirrorTransform:
    def __init__(self, outer_size, inner_size=None):
        self.outer = outer_size if type(outer_size) == tuple else (outer_size, outer_size)
        if inner_size is None:
            self.inner = self.outer
        else:
            self.inner = inner_size if type(inner_size) == tuple else (inner_size, inner_size)

    def __call__(self, x: Tensor, y: Tensor, norm=True) -> Tuple[Tensor, Tensor]:
        # check sizes match
        assert(x.size() == y.size())

        _, h, w = x.size()

        # normalize first
        if norm:
            x = torch.divide(x, 255)
            y = torch.divide(y, 255)

        # pad to align image to inner_size
        pad1_h = 0 if h % self.inner[0] == 0 else self.inner[0] - h % self.inner[0]
        pad1_w = 0 if w % self.inner[1] == 0 else self.inner[1] - w % self.inner[1]

        # pad to account for difference between inner and outer
        diff_h = self.outer[0] - self.inner[0]
        diff_w = self.outer[1] - self.inner[1]

        padded_y = torch.nn.ReflectionPad2d(
            (
                pad1_w // 2,  # left
                (pad1_w + 1) // 2,  # right
                pad1_h // 2,  # top
                (pad1_h + 1) // 2,  # bottom
            )
        )(y)

        padded_x = torch.nn.ReflectionPad2d(
            (
                (pad1_w + diff_w) // 2,      # left
                (pad1_w + diff_w + 1) // 2,  # right
                (pad1_h + diff_h) // 2,      # top
                (pad1_h + diff_h + 1) // 2,  # bottom
            )
        )(x)

        # get random coordinates where to start crop
        _, hpy, wpy = padded_y.size()
        top = np.random.randint(0, hpy - self.inner[0] + 1)
        left = np.random.randint(0, wpy - self.inner[1] + 1)

        # crop outer for x, inner for y
        return TF.crop(
            padded_x, top, left, self.outer[0], self.outer[1]
        ), TF.crop(
            padded_y, top, left, self.inner[0], self.inner[1]
        )


TRANSFORM_SWIN = CropMirrorTransform(128)

TRANSFORM_UNET = CropMirrorTransform(572, 388)
