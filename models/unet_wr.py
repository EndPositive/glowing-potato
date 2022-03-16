from swin_wr_base import SwinWRBase
from unet.unet.unet_model import UNet
from torch import nn, optim
from datasets.chunked_watermarked_set import ChunkedWatermarkedSet, DataSetType
from datasets.transform import TRANSFORM_UNET


class UNetWR(SwinWRBase):
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
        batch_size=1,
        data_set=ChunkedWatermarkedSet(
            DataSetType.Training, m.device, TRANSFORM_UNET
        )
    )
