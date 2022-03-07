import sys
from pathlib import Path
sys.path.append(Path(__file__).parents[1].absolute().as_posix())

import numpy as np
import torch
from PIL import Image

from swinir.models.network_swinir import SwinIR
from wrmodel import WRmodel

class SwinWR(WRmodel):
    def __init__(self, image_size=(128, 128), train_last_layer_only=False):
        super().__init__(image_size)
        self.model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=self.image_size[0],
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )

        self.model.load_state_dict(
            torch.utils.model_zoo.load_url(
                "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
            )["params"]
        )

        if train_last_layer_only:
            for _, param in self.model.named_parameters():
                param.requires_grad = False

            self.model.conv_last.weight.requires_grad = True
            self.model.conv_last.bias.requires_grad = True

    def _decode_output(self, x):
        x = x.cpu().detach().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        return (x * 255).astype(np.uint8)

    def _encode_input(self, x):
        if len(x.shape) == 3:
            x = np.expand_dims(x, 0)
        x = np.transpose(x, (0, 3, 1, 2)) / 255  # convert to channels first
        return torch.from_numpy(x).float().to("cpu")

    def __call__(self, img):
        return self.predict(img)

    def forward_pass(self, x):
        return self.model(x)


if __name__ == "__main__":
    with torch.no_grad():
        model = SwinWR()
        Image.fromarray(model("../resources/dataset/input/0c3ee986fa326b1a_7.jpg")).show()
