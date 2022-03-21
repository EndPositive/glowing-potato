from typing import List
from PIL import Image
from preprocessing import RESOURCES_DIR
from preprocessing.asset_watermarker import AssetWatermarker, Asset


class ShutterstockWatermarker(AssetWatermarker):
    def __init__(self):
        super().__init__()

    def load_assets(self) -> List[Asset]:
        def get_f(fn):
            return RESOURCES_DIR.joinpath('shutterstock', fn).as_posix()

        return [
            Asset(get_f('logo.png')),
            Asset(get_f('logo-fill.png')),
            Asset(get_f('logo-black.png')),
            Asset(get_f('sh-outline.png'), relative_image_scale=0.3),
            Asset(get_f('logo-corners.png'))
        ]


if __name__ == '__main__':
    s = ShutterstockWatermarker()
    img = Image.open(RESOURCES_DIR.joinpath('edge', '0c3ee986fa326b1a.jpg').as_posix())
    s(img, grid_layout=(5, 4), n_picks=1).show()
