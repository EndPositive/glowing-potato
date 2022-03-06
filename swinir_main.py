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


def image_to_input(x):
    x = np.transpose(x, (2, 0, 1))  # convert to channels first
    return torch.from_numpy(x).float().unsqueeze(0).to("cpu")  # add batch dimension


if __name__ == "__main__":
    nn = load_swinir()

    sample_file_path = OUTPUT_DIR.joinpath("input", "0bef511619cf3bb4_6.jpg")
    x = np.asarray(Image.open(sample_file_path))[:128, :128, :] / 255

    with torch.no_grad():
        tensor_out = nn(image_to_input(x))
        np_out = tensor_out.cpu().detach().numpy()
        np_out = (np_out * 255).astype(np.uint8)
        out = Image.fromarray(np.transpose(np_out[0], (1, 2, 0)))
        out.show()
