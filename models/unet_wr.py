from wr_base import WRBase
from models.unet import UNet
from torch import nn, optim
from datasets.transform import TRANSFORM_UNET


class UNetWR(WRBase):
    def __init__(self, image_size=(572, 572)):
        super().__init__(image_size)

        self._model = UNet(n_channels=3, n_classes=3)
        self._lossfn = nn.L1Loss()
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=0.001, weight_decay=0.0001
        )
        self._transforms = TRANSFORM_UNET


if __name__ == '__main__':
    m = UNetWR()
    m.train(
        batch_size=1
    )
