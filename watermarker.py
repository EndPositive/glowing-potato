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

def getRandomColor():
    return tuple(np.random.choice(range(256), size=3)) + (random.choice(range(70, 90)),)

def radians(deg):
    return deg * math.pi / 180

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

    def rotated_text(self, angle, xy, text, fill, upscale=False, center_coords=False, *args, **kwargs):
        # Upscale increases text quality when rotated at none 90 degree angle
        max_dim = max(*self.image.size)
        mask = Image.new('L', (max_dim*2, max_dim*2), 0)
        draw = ImageDraw.Draw(mask)
        if center_coords: 
            half_text_height = int(draw.textsize(text, self.font)[1] / 2)
            xy = (xy[0], xy[1] - (half_text_height + math.cos(radians(angle)) * half_text_height))

        draw.text((max_dim, max_dim), text, 255, *args, **kwargs)
        rotated_mask = mask.rotate(angle, resample=(3 if upscale else 0))   

        mask_xy = (max_dim - xy[0], max_dim - xy[1])
        final_mask = rotated_mask.crop(mask_xy + (mask_xy[0] + self.image.width, mask_xy[1] + self.image.height))

        self.image.paste(Image.new('RGBA', self.image.size, fill), final_mask)

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
        # self.rotated_text(0, (x1, y1), WATERMARK_TEXT, fill=FONT_RGB, font=self.font, upscale=False)

    def draw_angled_line_y(self, xy, y2, angle, fill=BG_RGBA):
        xy2 = (xy[0] + int((y2 - xy[1]) / math.tan(radians(angle))), y2) if angle else (xy[0], y2)
        self.draw.line((*xy, xy2), fill, width=4)

    def draw_angled_line_x(self, xy, x2, angle, fill=BG_RGBA):
        xy2 = (x2, xy[1] + int((x2 - xy[0]) * math.tan(radians(angle))))
        self.draw.line((*xy, xy2), fill, width=4)

    def draw_grid(self, angle, lines=5, offset_x=0, offset_y=0, line_color=BG_RGBA):
        added_angle = 45 if (angle := angle % 45) else 0 # Cursed python code
        start_X, start_Y = -self.image.width * 2 + offset_x, -self.image.height * 2 + offset_y
        for i in range(total_lines := lines * 3):
            spacing_W, spacing_H = self.image.width * 3 / total_lines, self.image.height * 3 / total_lines
            self.draw_angled_line_y((int(start_X + i * spacing_W), 0), self.image.height, angle + added_angle, fill=line_color)
            self.draw_angled_line_x((self.image.width, int(start_Y + i * spacing_H)), 0, -angle, fill=line_color)

    def draw_randomized_grid(self):
        angle = random.randrange(0, 90)
        lines = random.randrange(3,10)
        offset_x, offset_y = random.randrange(0,100), random.randrange(0,100)
        fill_color = getRandomColor()
        self.draw_grid(angle, lines, offset_x, offset_y, fill_color)

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
            # self.rotated_text(45, (int((x1 - x0) / 2) + x0, int(self.image.height / 2)), "TEST", fill=FONT_RGB, font=self.font, upscale=False, center_coords=True)

    def save(self, output_dir=OUTPUT_DIR):
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
        watermarker.draw_randomized_grid()
        watermarker.save()

    logger.info(f"Processed {len(jpgs)} images. Skipped {len(skipped)}.")
