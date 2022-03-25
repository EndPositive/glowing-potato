import sys, os, threading, random
from pathlib import Path

sys.path.append(Path(__file__).parents[1].absolute().as_posix())

from wr_base import WRBase
from models.unet import UNet
from torch import nn, optim
from datasets.transform import CropMirrorTransform
from datasets.chunked_watermarked_set import ChunkedWatermarkedSet, DataSetType
from torch.utils.data import DataLoader
from PIL import Image
import torch
from tqdm import tqdm
from IPython.display import display
from preprocessing import DATASET_DIR, OUTPUT_DIR, DATASET_TEST_DIR, OUTPUT_DIR_2


class UNetWR(WRBase):
    def __init__(self, input_size=(572, 572), output_size=(388, 388)):
        super().__init__(input_size)

        self.output_size = output_size
        self._model = UNet(n_channels=3, n_classes=3)
        self._model.to(self.device)
        print(f"Running model on {self.device}")
        self._lossfn = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters())
        self._transforms = CropMirrorTransform(input_size, output_size)

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


def newest(p): # gets latest modified file
    paths = [os.path.join(p, b) for b in os.listdir(p)]
    return max(paths, key=os.path.getmtime)

if __name__ == "__main__":
    i, o = 288, 100

    m = UNetWR(
        input_size=i,
        output_size=o,
    )

    train = sys.argv[-1] == "train"

    # model_to_load = "ckpt/unet_start.pth"
    model_to_load = newest("ckpt/unet_chaos")

    m.load(model_to_load)

    if train:
        m.train_model(
            n_epochs=-1,
            batch_size=12,
            save_every=1,
            save_path="ckpt/unet_chaos",
            data_shuffle=False,
            data_set=ChunkedWatermarkedSet(
                data_set_type=DataSetType.Training,
                device=m.device,
                transforms=CropMirrorTransform(i, o),
                # watermarked_dir=DATASET_DIR,
                watermarked_dir=OUTPUT_DIR,
                original_dir=DATASET_DIR,
            ),
        )
    else:
        print("Using model: " + model_to_load.split("\\")[-1])
        img_name = "resources/out/" + random.choice(os.listdir("resources/out/"))
        # img_name = "resources/out/0a0c9a2d7341096f.jpg"
        # img_name = "resources/real/123rf.webp"
        # img_name = "resources/real/old2.jpg"
        # img_name = "resources/real/123rf_3.webp"
        x = Image.open(img_name)
        
        y = m.predict(x, max_batch_size=8)
        threading.Thread(target=x.show).start()
        threading.Thread(target=y.show).start()
