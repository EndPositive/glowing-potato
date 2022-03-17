from typing import List

import numpy as np
from PIL import Image

from preprocessing import RESOURCES_DIR


class Asset:
    def __init__(
        self,
        path,
        scale_std=0.3,
        opacity_range=(0.3, 0.7),
        relative_image_scale=0.2
    ):
        self.original = Image.open(path).convert('RGBA')
        self.scale_std = scale_std
        self.opacity_range = opacity_range
        self.relative_image_scale = relative_image_scale

    def __call__(
        self,
        other: Image,
        angle=None,
        position=None
    ) -> Image:
        modified = self.original.copy()
        other = other.convert('RGBA')

        # opacity
        new_alpha = np.random.rand() * (self.opacity_range[1] - self.opacity_range[0]) + self.opacity_range[0]
        pixels = modified.load()
        for x in range(modified.width):
            for y in range(modified.height):
                r, g, b, a = pixels[x, y]
                pixels[x, y] = r, g, b, int(a * new_alpha)

        # relative_image_scale specifies how big we want the asset to be (on average)
        # compared to the image we paste it on; norm_scale controls the size to approach this value
        norm_scale = min(*self.original.size) / max(*other.size) * self.relative_image_scale

        # we multiply norm_scale by another piece of randomness (sample from a standard distribution)
        scale_factor = np.abs(np.random.normal(1.1, self.scale_std)) * norm_scale
        nw = int(modified.width * scale_factor)
        nh = int(modified.height * scale_factor)
        modified.thumbnail((nw, nh))

        # rotation
        if angle is None:
            angle = np.random.randint(360)

        modified = modified.rotate(angle, expand=True)

        # paste onto other image
        if position is None:
            position = (
                np.random.randint(0, int(0.9 * other.height)),
                np.random.randint(0, int(0.9 * other.width)),
            )

        other.paste(modified, position, modified)
        return other


class AssetWatermarker:
    def __init__(
        self,
        font='calibri',
    ):
        self.font = font
        self.assets = self.load_assets()

    def load_assets(self) -> List[Asset]:
        return []

    def __call__(
        self,
        image,
        n_picks=7
    ) -> Image:
        if type(image) == str:
            image = Image.open(image)

        for _ in range(n_picks):
            image = np.random.choice(self.assets)(image)

        return image


if __name__ == '__main__':
    # a = Asset(RESOURCES_DIR.joinpath('shutterstock', 'sh-white.png').as_posix())
    a = Asset(RESOURCES_DIR.joinpath('shutterstock', 'logo.png').as_posix())
    img = Image.open(RESOURCES_DIR.joinpath('edge', '0c3ee986fa326b1a.jpg').as_posix())
    a(img).show()
