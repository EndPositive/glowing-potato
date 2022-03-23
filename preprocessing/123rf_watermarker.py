from PIL import Image
import PIL, time
import numpy as np
from PIL import ImageEnhance

class RF_Watermarker():
    def __init__(self):
        self.watermark = Image.open("resources/123rf.png")
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
        brightness = np.random.randint(25) / 10
        if self.coinflip(): brightness = -brightness
        return self.brightenImage(img, brightness)

    def getRandomAlpha(self, fg):
        alpha = 120 + np.random.randint(80)
        channel = fg.getchannel('A')
        return channel.point(lambda i: alpha if i>0 else 0)

    def addWatermark(self, img_file):
        bg = Image.open(img_file)
        fg = self.getWatermark()
        fg = self.flipWaterMark(fg)
        fg = self.randomizeBrightness(fg)
        alpha = self.getRandomAlpha(fg)
        fg.putalpha(alpha)
        fg = fg.resize(bg.size, PIL.Image.ANTIALIAS)
        bg.paste(fg, (0, 0), fg.convert('RGBA'))
        return bg

RF = RF_Watermarker()
res = RF.addWatermark("resources/edge/0c3ee986fa326b1a.jpg")
res.show()