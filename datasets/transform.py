import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import numpy as np
from torch import Tensor
from typing import Tuple
from PIL import Image
from itertools import product


class CropMirrorTransform:
    def __init__(self, outer_size, inner_size=None):
        self.outer = outer_size if type(outer_size) == tuple else (outer_size, outer_size)
        if inner_size is None:
            self.inner = self.outer
        else:
            self.inner = inner_size if type(inner_size) == tuple else (inner_size, inner_size)

    def get_padding(self, x):
        _, h, w = x.size()

        # pad to align image to inner_size
        pad1_h = 0 if h % self.inner[0] == 0 else self.inner[0] - h % self.inner[0]
        pad1_w = 0 if w % self.inner[1] == 0 else self.inner[1] - w % self.inner[1]

        # pad to account for difference between inner and outer
        diff_h = self.outer[0] - self.inner[0]
        diff_w = self.outer[1] - self.inner[1]

        inner_padding = (
            pad1_w // 2,  # left
            (pad1_w + 1) // 2,  # right
            pad1_h // 2,  # top
            (pad1_h + 1) // 2,  # bottom
        )

        outer_padding = (
            (pad1_w + diff_w) // 2,  # left
            (pad1_w + diff_w + 1) // 2,  # right
            (pad1_h + diff_h) // 2,  # top
            (pad1_h + diff_h + 1) // 2,  # bottom
        )

        return inner_padding, outer_padding

    def get_prediction_batch(self, x: Image):
        # convert to tensor
        x = TF.to_tensor(x)

        # pad
        inner, outer = self.get_padding(x)
        x_inner = torch.nn.ReflectionPad2d(inner)(x)
        x = torch.nn.ReflectionPad2d(outer)(x)

        # get points where to crop
        _, h, w = x_inner.size()
        h_coords = np.arange(0, h, self.inner[0])
        w_coords = np.arange(0, w, self.inner[1])
        crops = product(h_coords, w_coords)

        # unsqueeze for batch dim so we can later concatenate
        x = torch.unsqueeze(x, 0)

        # convolve over image and crop input, output pair
        inputs = [
            TF.crop(x, top, left, self.outer[0], self.outer[1])
            for top, left in crops
        ]
        return torch.cat(inputs)

    def image_from_prediction(self, prediction: Tensor, original: Image) -> Image:
        # convert original image to tensor
        original = TF.to_tensor(original)

        # get inner padding and pad original
        inner, _ = self.get_padding(original)
        padded = torch.nn.ReflectionPad2d(inner)(original)

        # calculate number of patches in each dimension
        _, h, w = padded.size()
        h_patches = h // self.inner[0]
        w_patches = w // self.inner[1]

        # sanity check: number of patches matches expectation
        assert(h_patches * w_patches == prediction.size()[0])

        # merge patches
        prediction = prediction.reshape(h_patches, w_patches, 3, self.inner[0], self.inner[1])
        # print([torch.cat([x for x in line], -1).size() for line in prediction])
        prediction = torch.cat([
            torch.cat([x for x in line], -1) for line in prediction
        ], -2)

        # crop to remove padding
        prediction = TF.crop(prediction, inner[2], inner[0], original.size()[1], original.size()[2])

        return T.ToPILImage()(prediction)

    def __call__(self, x: Tensor, y: Tensor, norm=True) -> Tuple[Tensor, Tensor]:
        # check sizes match
        assert(x.size() == y.size())

        # normalize first
        if norm:
            x = torch.divide(x, 255)
            y = torch.divide(y, 255)

        inner, outer = self.get_padding(x)

        padded_x = torch.nn.ReflectionPad2d(outer)(x)
        padded_y = torch.nn.ReflectionPad2d(inner)(y)

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
