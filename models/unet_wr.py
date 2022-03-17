from wr_base import WRBase
from models.unet import UNet
from torch import nn, optim
from datasets.transform import CropMirrorTransform


class UNetWR(WRBase):
    def __init__(self, input_size=(572, 572), output_size=(388, 388)):
        super().__init__(input_size)

        self.output_size = output_size
        self._model = UNet(n_channels=3, n_classes=3)
        self._lossfn = nn.L1Loss()
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=0.001, weight_decay=0.0001
        )
        self._transforms = CropMirrorTransform(input_size, output_size)


if __name__ == '__main__':
    m = UNetWR(
        input_size=384,
        output_size=196
    )
    m.train(
        batch_size=16
    )
