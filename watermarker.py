#!/usr/bin/env python3

import os.path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prompt_toolkit.shortcuts import confirm
from logger import logger

INPUT_DIR = "resources/edge"
OUTPUT_DIR = "out"

# Randomness
SEED = None
SEED = SEED if SEED else int(random.random() * 10000)
random.seed(SEED)
np.random.seed(SEED)

# Background
BG_RGB = (0, 0, 0)
BG_RGB = BG_RGB if BG_RGB else tuple(np.random.choice(range(256), size=3))
BG_OPACITY = 90
BG_OPACITY = BG_OPACITY if BG_OPACITY else random.choice(range(70, 90))
BG_RGBA = BG_RGB + (BG_OPACITY,)

# Text
FONT_RGB = (255, 255, 255)
FONT_RGB = FONT_RGB if FONT_RGB else tuple(np.random.choice(range(256), size=3))
FONT_SIZE = 50
WATERMARK_TEXT = "Getty Images"


class Watermarker:
    def __init__(self, filename):
        if not filename.endswith(".jpg"):
            raise ValueError("File is not a jpg")
        self.image = Image.open(f"{INPUT_DIR}/{filename}")
        if self.image.mode not in ["RGB", "RGBA"]:
            raise ValueError(f"{filename} has incorrect mode {self.image.mode}")
        self.draw = ImageDraw.Draw(self.image, "RGBA")

    def add_text(self):
        w, h = font.getsize(WATERMARK_TEXT)
        padding = 20
        x = int(random.random() * (self.image.width - (w + padding))) + (w + padding) / 2
        y = int(random.random() * (self.image.height - (h + padding))) + (h + padding) / 2
        x1 = x - (w / 2)
        x2 = x + (w / 2)
        y1 = y - (h / 2)
        y2 = y + (h / 2)

        self.draw.rectangle((x1 - padding, y1 - padding, x2 + padding, y2 + padding), BG_RGBA, BG_RGBA, 0)

        self.draw.text((x1, y1), WATERMARK_TEXT, fill=FONT_RGB, font=font)

    def save(self):
        self.image.save(f"out/{filename}")
        logger.debug(f"Added watermark to {filename}")


if __name__ == "__main__":
    logger.info(f"SEED: {SEED}")

    jpgs = os.listdir(INPUT_DIR)

    if os.path.exists(OUTPUT_DIR):
        answer = confirm(f"Delete output directory {OUTPUT_DIR}")
        if answer:
            shutil.rmtree(OUTPUT_DIR)
        else:
            exit(1)

    os.makedirs(OUTPUT_DIR)

    font = ImageFont.truetype("resources/calibri.ttf", FONT_SIZE)

    skipped = []
    for filename in jpgs:
        SEED += 1
        random.seed(SEED)
        try:
            watermarker = Watermarker(filename)
        except ValueError as e:
            logger.error(e)
            skipped += filename
            continue
        watermarker.add_text()
        watermarker.save()

    logger.info(f"Processed {len(jpgs)} images. Skipped {len(skipped)}.")
