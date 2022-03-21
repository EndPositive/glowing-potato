from wr_base import WRBase
from models.unet import UNet
from torch import nn, optim
from datasets.transform import CropMirrorTransform
from datasets.chunked_watermarked_set import ChunkedWatermarkedSet, DataSetType
from torch.utils.data import DataLoader
from PIL import Image
import torch
from tqdm import tqdm
from preprocessing import DATASET_DIR


class UNetWR(WRBase):
    def __init__(self, input_size=(572, 572), output_size=(388, 388)):
        super().__init__(input_size)

        self.output_size = output_size
        self._model = UNet(n_channels=3, n_classes=3)
        self._model.to(self.device)
        print(f"Running model on {self.device}")
        self._lossfn = nn.L1Loss()
        self._optimizer = optim.Adam(self._model.parameters())
        self._transforms = CropMirrorTransform(input_size, output_size)

    def predict(self, img: Image, max_batch_size=8) -> Image:
        self.eval()
        batch = self._transforms.get_prediction_batch(img)
        batches = torch.split(batch, max_batch_size)
        with torch.no_grad():
            pred = torch.cat([self(x.to(self.device)) for x in tqdm(batches)], 0)
        return self._transforms.image_from_prediction(pred, img)

    def show_sample(self):
        self.eval()
        loader = DataLoader(
            ChunkedWatermarkedSet(
                data_set_type=DataSetType.Test,
                device=self.device,
                transforms=self._transforms,
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


if __name__ == "__main__":
    i, o = 288, 100

    m = UNetWR(
        input_size=i,
        output_size=o,
    )

    m.load("../ckpt_2.pth")

    m.train_model(
        n_epochs=1,
        batch_size=4,
        save_every=1,
        save_path="reprod_input_weights",
        data_set=ChunkedWatermarkedSet(
            data_set_type=DataSetType.Training,
            split_size=(0.05, 0.7, 1 - 0.05 - 0.7),
            device=m.device,
            transforms=CropMirrorTransform(i, o),
            watermarked_dir=DATASET_DIR,
            # watermarked_dir=OUTPUT_DIR,
            original_dir=DATASET_DIR,
        ),
    )

    x = Image.open("../resources/edge/0c3ee986fa326b1a.jpg")
    y = m.predict(x, max_batch_size=8)
    x.show()
    y.show()
