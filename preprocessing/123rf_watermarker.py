import os,sys
from pathlib import Path

sys.path.append(Path(__file__).parents[1].absolute().as_posix())
from PIL import Image
import PIL, time
import numpy as np
from tqdm import tqdm
from preprocessing.custom_pool import CustomPool
from PIL import ImageEnhance
from preprocessing import DATASET_TEST_DIR, OUTPUT_DIR_2

class RF_Watermarker():
    def __init__(self):
        self.watermark = Image.open("resources/123rf_clean.png")
        self.watermark = self.brightenImage(self.watermark, 5)

    def getWatermark(self): return self.watermark.copy()

    def coinflip(self): return np.random.randint(2) == 1

    def flipWaterMark(self, img):
        if self.coinflip(): img = PIL.ImageOps.flip(img)
        if self.coinflip(): img = PIL.ImageOps.mirror(img)
        return img

    def brightenImage(self, img, brightness):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(brightness)

    def randomizeBrightness(self, img):
        brightness = np.random.randint(12) / 10
        if self.coinflip(): brightness = -brightness * 2
        return self.brightenImage(img, brightness)

    def getRandomAlpha(self, fg):
        alpha = 120 + np.random.randint(80)
        channel = fg.getchannel('A')
        return channel.point(lambda i: alpha if i>0 else 0)

    def addWatermark(self, img_file):
        bg = Image.open(img_file)
        if bg.mode not in ["RGB"]: return 
        fg = self.getWatermark()
        fg = self.flipWaterMark(fg)
        fg = self.randomizeBrightness(fg)
        alpha = self.getRandomAlpha(fg)
        fg.putalpha(alpha)
        fg = fg.resize(bg.size, PIL.Image.ANTIALIAS)
        bg.paste(fg, (0, 0), fg.convert('RGBA'))
        file_name = str(img_file).split("\\")[-1]
        bg.save(f'{OUTPUT_DIR_2}/{file_name}')

p = CustomPool(100)
RF = RF_Watermarker()

input_jpgs = [str(x) for x in DATASET_TEST_DIR.glob("*.jpg")]
existing_jpgs = [str(x).split("\\")[-1] for x in OUTPUT_DIR_2.glob("*.jpg")]
for img_file in tqdm(input_jpgs):
    if img_file.split("\\")[-1] in existing_jpgs: continue
    p.map(RF.addWatermark, [img_file])
    
# res = RF.addWatermark("resources/edge/0c3ee986fa326b1a.jpg")
# res.show()