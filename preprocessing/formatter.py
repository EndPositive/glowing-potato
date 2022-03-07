from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from numpy import ndarray


def pad(img: Union[ndarray, Image.Image], align_to, return_padding=False):
    """
    Pads the image to align first two axes to the given dimensions

    Arguments
        img: np.array of shape (h, w, 3) or Image object
        align_to: tuple of size 2
    Returns
        padded: the padded image, in array form
    """

    if isinstance(img, Image.Image):
        img = np.asarray(img)

    h, w, _ = np.shape(img)

    if h % align_to[0] == 0:
        vertical_padding = 0
    else:
        vertical_padding = align_to[0] - h % align_to[0]
    padding_top = vertical_padding // 2
    padding_bottom = (vertical_padding + 1) // 2

    if w % align_to[1] == 0:
        horizontal_padding = 0
    else:
        horizontal_padding = align_to[1] - w % align_to[1]
    padding_left = horizontal_padding // 2
    padding_right = (horizontal_padding + 1) // 2

    padded = np.pad(
        img,
        ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)),
        "constant",
    )

    if return_padding:
        return padded, (padding_top, padding_bottom, padding_left, padding_right)
    return padded


def unpad(img: ndarray, padding_size):
    return img[
        padding_size[0] : -padding_size[1] if padding_size[1] != 0 else None,
        padding_size[2] : -padding_size[3] if padding_size[3] != 0 else None,
        :,
    ]


def split_image(img: Union[ndarray, Image.Image, str, Path], chunk_size=(256, 256)):
    """
    Splits image into eqauly sized smaller chunks

    Arguments
        img: either PIL.Image or file path (as str)
        chunk_size: tuple of size 2 specifying the size of each chunk
    Returns
        chunks: array of PIL image
    """
    if isinstance(img, str) or isinstance(img, Path):
        img = Image.open(img)

    if isinstance(img, Image.Image):
        img = pad(img, chunk_size)

    # convert to np array
    img = np.array(img)

    # sanity check
    if img.shape[0] % chunk_size[0] != 0 or img.shape[1] % chunk_size[1] != 0:
        raise ValueError(
            "Image cannot be split in equal sized chunks. You should pad before."
        )

    # split image into chunks of chunk_size
    img = np.array(
        [
            img[
                row * chunk_size[0] : (row + 1) * chunk_size[0],
                col * chunk_size[1] : (col + 1) * chunk_size[1],
            ]
            for row in range(img.shape[0] // chunk_size[0])
            for col in range(img.shape[1] // chunk_size[1])
        ]
    )

    # convert each chunk to PIL Image and return
    return img


def unsplit_image(chunks, image_size):
    """
    Reconstructs original image from the chunks produced by split_image

    Arguments
        chunks: list of Image or np.array representing images produced by split_image
        image_size: either integer (pre-split height) or (height, width) tuple
    Returns
        image put back together
    """
    if isinstance(chunks[0], Image.Image):
        chunks = list(map(np.asarray, chunks))

    image_height = image_size if isinstance(image_size, int) else image_size[0]
    chunk_height = chunks[0].shape[0]

    n_rows = image_height // chunk_height
    chunks = np.split(chunks, n_rows)
    chunks = [np.hstack(x) for x in chunks]
    chunks = np.vstack(chunks)
    return chunks
