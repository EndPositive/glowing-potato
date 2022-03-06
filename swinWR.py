import numpy as np
import torch
from PIL import Image

from preprocessing import OUTPUT_DIR
import preprocessing.formatter as processing
from swinir.models.network_swinir import SwinIR


class SwinWR:
    def __init__(self, train_last_layer_only=False):
        self.model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=128,
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

    @staticmethod
    def _output_to_array(x):
        x = x.cpu().detach().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        return (x * 255).astype(np.uint8)

    @staticmethod
    def _array_to_input(x):
        if len(x.shape) == 3:
            x = np.expand_dims(x, 0)
        x = np.transpose(x, (0, 3, 1, 2)) / 255 # convert to channels first
        return torch.from_numpy(x).float().to('cpu') # add batch dimension

    def __call__(self, img):
        if type(img) == str:
            img = Image.open(img)    

        padded, padding = processing.pad(img, (128, 128), return_padding=True)
        split = processing.split_image(padded, (128, 128))
        output = self._output_to_array(self.model(self._array_to_input(split)))
        unsplit = processing.unsplit_image(output, padded.shape[0])
        unpadded = processing.unpad(unsplit, padding)
        return unpadded


if __name__ == '__main__':
    with torch.no_grad():
        model = SwinWR()
        Image.fromarray(
            model('resources/dataset/input/0c3ee986fa326b1a_7.jpg')
        ).show()
