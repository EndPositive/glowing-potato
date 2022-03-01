import numpy as np
from PIL import Image


def pad(img, align_to):
    """
    Pads the image to align first two axes to the given dimensions

    Arguments
        img: np.array of shape (h, w, 3)
        align_to: tuple of size 2
    Returns
        padded: the padded image
    """

    h, w, _ = np.shape(img)
    if h == align_to[0] and w == align_to[1]:
        return img

    padding_top = (align_to[0] - h % align_to[0]) // 2
    padding_bottom = (align_to[0] - h % align_to[0] + 1) // 2
    padding_left = (align_to[1] - w % align_to[1]) // 2
    padding_right = (align_to[1] - w % align_to[1] + 1) // 2
    return np.pad(img, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), 'constant')


def split_image(img, chunk_size=(256, 256)):
    """
    Splits image into eqauly sized smaller chunks

    Arguments
        img: either PIL.Image or file path (as str)
        chunk_size: tuple of size 2 specifying the size of each chunk
    Returns
        chunks: array of PIL image
    """
    if type(img) == str:
        img = Image.open(img)

    # convert to np array for easy manipulation
    arr_img = np.array(img)

    # pad in order to make image splittable into chunks of same size
    arr_img = pad(arr_img, chunk_size)

    # split image into chunks of chunk_size
    arr_img = [
        arr_img[
            row * chunk_size[0] : (row + 1) * chunk_size[0],
            col * chunk_size[1] : (col + 1) * chunk_size[1]
        ]
        for row in range(arr_img.shape[0] // chunk_size[0])
        for col in range(arr_img.shape[1] // chunk_size[1])
    ]

    # convert each chunk to PIL Image and return
    return list(map(Image.fromarray, arr_img))

