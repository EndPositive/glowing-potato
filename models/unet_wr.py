from swin_wr_base import SwinWRBase
from unet.unet.unet_model import UNet
from torch import nn, optim


class UnetWR(SwinWRBase):
    def __init__(self, image_size=(572, 572)):
        super().__init__(image_size)

        self._model = UNet(n_channels=3, n_classes=3)
        self._lossfn = nn.L1Loss()
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=0.001, weight_decay=0.0001
        )
