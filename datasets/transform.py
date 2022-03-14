import torch
import torchvision.transforms as T

TRANSFORM = T.Compose(
    [
        T.RandomCrop(128, pad_if_needed=False),
        T.Lambda(lambda tensor: torch.divide(tensor, 255)),
    ]
)
