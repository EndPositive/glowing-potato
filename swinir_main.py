import numpy as np
import torch
from PIL import Image

from preprocessing import OUTPUT_DIR
from swinir.models.network_swinir import SwinIR


def load_swinir(train_only_last=False):
    model = SwinIR(
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
    model.load_state_dict(
        torch.utils.model_zoo.load_url(
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
        )["params"]
    )

    if train_only_last:
        for _, param in model.named_parameters():
            param.requires_grad = False

        model.conv_last.weight.requires_grad = True
        model.conv_last.bias.requires_grad = True

    return model


def array_to_input(x):
    if len(x.shape) == 3:
        x = np.expand_dims(x, 0)
    x = np.transpose(x, (0, 3, 1, 2)) / 255 # convert to channels first
    return torch.from_numpy(x).float().to('cpu') # add batch dimension


def output_to_array(x):
    x = x.cpu().detach().numpy()
    x = np.transpose(x, (0, 2, 3, 1))
    return (x * 255).astype(np.uint8)


def predict_full_image(nn, img):
    import preprocessing.formatter as processing

    if type(img) == str:
        img = Image.open(img)    

    padded, padding = processing.pad(img, (128, 128), return_padding=True)
    split = processing.split_image(padded, (128, 128))
    output = output_to_array(nn(array_to_input(split)))
    unsplit = processing.unsplit_image(output, padded.shape[0])
    unpadded = processing.unpad(unsplit, padding)
    return unpadded


if __name__ == '__main__':
    with torch.no_grad():
        Image.fromarray(
            # predict_full_image(load_swinir(), 'resources/edge/0c3ee986fa326b1a.jpg')
            predict_full_image(load_swinir(), 'resources/dataset/input/0c3ee986fa326b1a_7.jpg')
        ).show()
