from wr_base import WRBase
from models.unet import UNet
from torch import nn, optim
from datasets.transform import CropMirrorTransform
from datasets.chunked_watermarked_set import ChunkedWatermarkedSet, DataSetType
from torch.utils.data import DataLoader
from PIL import Image
import torch
from tqdm import tqdm


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

    def predict(self, img: Image, max_batch_size=8) -> Image:
        batch = self._transforms.get_prediction_batch(img)
        batches = torch.split(batch, max_batch_size)
        pred = torch.cat([self(x) for x in tqdm(batches)], 0)
        return self._transforms.image_from_prediction(pred, img)

    def show_sample(self):
        loader = DataLoader(
            ChunkedWatermarkedSet(
                data_set_type=DataSetType.Test, device=self.device, transforms=self._transforms
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        for x, y in iter(loader):
            Image.fromarray(self._decode_output(x)[0]).show()
            Image.fromarray(self._decode_output(self(x))[0]).show()
            Image.fromarray(self._decode_output(y)[0]).show()
            break


if __name__ == '__main__':
    m = UNetWR(
        input_size=288,
        output_size=100,
    )
    m.load('../ckpt_2.pth')
    x = Image.open('../resources/edge/0a1aee5d7701ce5c.jpg')
    # x.thumbnail((400, 400))
    y = m.predict(
        x, max_batch_size=1
    )

    x.show()
    y.show()
