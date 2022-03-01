#!/usr/bin/env python3
import math
import os.path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prompt_toolkit.shortcuts import confirm
from tqdm import tqdm

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
    def __init__(self, file_path, font=ImageFont.truetype("resources/calibri.ttf", FONT_SIZE)):
        if not file_path.endswith(".jpg"):
            raise ValueError("File is not a jpg")

        self.filename = os.path.basename(file_path)
        self.font = font
        self.image = Image.open(file_path)
        if self.image.mode not in ["RGB", "RGBA"]:
            raise ValueError(f"{self.filename} has incorrect mode {self.image.mode}")

        self.draw = ImageDraw.Draw(self.image, "RGBA")

    def add_text(self):
        w, h = self.font.getsize(WATERMARK_TEXT)
        padding = 20
        x = int(random.random() * (self.image.width - (w + padding))) + (w + padding) / 2
        y = int(random.random() * (self.image.height - (h + padding))) + (h + padding) / 2
        x1 = x - (w / 2)
        x2 = x + (w / 2)
        y1 = y - (h / 2)
        y2 = y + (h / 2)

        self.draw.rectangle((x1 - padding, y1 - padding, x2 + padding, y2 + padding), BG_RGBA, BG_RGBA, 0)

        self.draw.text((x1, y1), WATERMARK_TEXT, fill=FONT_RGB, font=self.font)

    def add_simple_grid(self):
        spacing = self.image.width / 5
        offset_x = random.random() * spacing
        offset_lower_x = random.random() * spacing
        lines_in_image = math.ceil(self.image.width / spacing)
        lines_before_image = math.ceil(self.image.height / spacing)
        for line_no in range(-lines_before_image, lines_in_image):
            x0 = line_no * spacing + offset_x
            x1 = x0 + self.image.height + offset_lower_x
            self.draw.line(((x0, 0), (x1, self.image.height)), BG_RGBA, width=2)
            self.draw.line(((x0, self.image.height), (x1, 0)), BG_RGBA, width=2)

    def save(self, output_dir):
        self.image.save(f"{output_dir}/{self.filename}")
        logger.debug(f"Added watermark to {self.filename}")


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

    skipped = []
    for i in tqdm(range(len(jpgs))):
        SEED += 1
        random.seed(SEED)
        try:
            watermarker = Watermarker(file_path=f"{INPUT_DIR}/{jpgs[i]}")
        except ValueError as e:
            logger.error(e)
            skipped += jpgs[i]
            continue
        watermarker.add_text()
        watermarker.add_simple_grid()
        watermarker.save(output_dir=OUTPUT_DIR)

    logger.info(f"Processed {len(jpgs)} images. Skipped {len(skipped)}.")
