import numpy as np
from PIL import Image


def pad(img, align_to, return_padding=False):
    """
    Pads the image to align first two axes to the given dimensions

    Arguments
        img: np.array of shape (h, w, 3) or Image object
        align_to: tuple of size 2
    Returns
        padded: the padded image, in array form
    """

    if type(img) == Image.Image:
        img = np.asarray(img)

    h, w, _ = np.shape(img)
    if h == align_to[0] and w == align_to[1]:
        return img

    padding_top = (align_to[0] - h % align_to[0]) // 2
    padding_bottom = (align_to[0] - h % align_to[0] + 1) // 2
    padding_left = (align_to[1] - w % align_to[1]) // 2
    padding_right = (align_to[1] - w % align_to[1] + 1) // 2
    padded =  np.pad(img, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), 'constant')

    if return_padding:
        return padded, (padding_top, padding_bottom, padding_left, padding_right)
    return padded


def unpad(img, padding_size):
    return img[
        padding_size[0]: -padding_size[1],
        padding_size[2]: -padding_size[3],
        :
    ]


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
        img = pad(img, chunk_size)

    if type(img) == Image.Image:
        # convert to np array
        img = np.array(img)

    # sanity check
    if img.shape[0] % chunk_size[0] != 0 or \
        img.shape[1] % chunk_size[1] != 0:
        raise ValueError('Image cannot be split in equal sized chunks. You should pad before.')

    # split image into chunks of chunk_size
    img = [
        img[
            row * chunk_size[0] : (row + 1) * chunk_size[0],
            col * chunk_size[1] : (col + 1) * chunk_size[1]
        ]
        for row in range(img.shape[0] // chunk_size[0])
        for col in range(img.shape[1] // chunk_size[1])
    ]

    # convert each chunk to PIL Image and return
    return list(map(Image.fromarray, img))


def unsplit_image(chunks, image_size):
    """
    Reconstructs original image from the chunks produced by split_image

    Arguments
        chunks: list of Image objects or np.array representing images produced by split_image
        image_size: either integer (pre-split height) or (height, width) tuple
    Returns
        image put back together
    """
    if type(chunks[0]) == Image.Image:
        chunks = list(map(np.asarray, chunks))

    image_height = image_size if type(image_size) == int else image_size[0]
    chunk_height = chunks[0].shape[0]

    n_rows = image_height // chunk_height
    n_cols = len(chunks) // n_rows

    rows = []
    for row in range(n_rows):
        # piece the image together column by column
        rows.append(np.hstack(chunks[row * n_cols: (row + 1) * n_cols]))

    return np.vstack(rows)

