from asyncio import BaseTransport
from contextlib import AbstractAsyncContextManager
import sys

sys.path.append("..")

from pathlib import Path
from PIL import Image
from abc import abstractclassmethod
from collections.abc import Iterable
from preprocessing import formatter as processing
import numpy as np

from data_set import ChunkedWatermarkedSet


class WRmodel:
    def __init__(self, image_size=(128, 128)):
        # constants
        self.image_size = image_size

    @abstractclassmethod
    def train_epoch(self):
        pass

    @abstractclassmethod
    def forward_pass(self):
        pass

    @abstractclassmethod
    def _encode_input(self, x):
        pass

    @abstractclassmethod
    def _decode_output(self, y):
        pass

    def predict(self, img):
        if isinstance(img, str) or isinstance(img, Path):
            img = Image.open(img)
            img = np.asarray(img)

        # shape should be image or batch of images
        assert len(np.shape(img)) == 3 or len(np.shape(img)) == 4

        # predict multiple images
        if len(np.shape(img)) == 4:
            return [self.predict(x) for x in img]

        # pad to align to img size
        padded, padding = processing.pad(img, self.image_size, return_padding=True)

        # split image in multiple blocks, if too big
        split = processing.split_image(padded, self.image_size)

        # normalize and get input in tensor form
        encoded_input = self._encode_input(split)

        # run the network and decode the output (get numpy from tensor and un-normalize)
        output = self._decode_output(self.forward_pass(encoded_input))

        # piece image back together
        unsplit = processing.unsplit_image(output, padded.shape[0])

        # unpad and return
        return processing.unpad(unsplit, padding)

    @abstractclassmethod
    def save(self, path):
        pass

    @abstractclassmethod
    def load(self, path):
        pass

    @abstractclassmethod
    def train_epoch(self, trainset: ChunkedWatermarkedSet):
        pass

    @abstractclassmethod
    def train(self, n_epochs=0):
        pass
