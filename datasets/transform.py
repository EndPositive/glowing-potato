import torch
import torchvision.transforms as T


def make_transform(img_size=(128, 128)):
    return T.Compose(
        [
            T.RandomCrop(img_size, pad_if_needed=False),
            T.Lambda(lambda tensor: torch.divide(tensor, 255)),
        ]
    )


TRANSFORM = make_transform()

TRANSFORM_UNET = make_transform(572)

