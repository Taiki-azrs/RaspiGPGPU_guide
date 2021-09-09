# coding:utf-8
# =================================== #
# RaspberryPi HDMI出力モジュール
# =================================== #
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# printImgのoption
PUT      = 0    # そのままのサイズで出力する
ADD_BACK = 1    # 画像に画面サイズと同じ大きさの背景を追加する
RESIZE   = 2    # 画像をリサイズする

# モニタサイズを取得
def getResolution():
  f = open('/sys/class/graphics/fb0/virtual_size', 'r')
  size = f.read()
  f.close()

  size = list(map(int, size.split(",")))
  return size

# 画面のクリア
def bufClear():
    os.system('dd if=/dev/zero of=/dev/fb0 > /dev/null 2>&1')

def addBackImage(img, width, height):
    back = Image.new('L', (width, height), 0)
    back.paste(img)

    return back

def addText(pil_img, x, y, str):
    draw = ImageDraw.Draw(pil_img)# im上のImageDrawインスタンスを作る
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 32)
    draw.text((x, y), str, font=font, fill=255)

def printImg(pil_img, width, height, option):
    if option == 1:
        pil_img = addBackImage(pil_img, width, height)
    # elif option == 2:
    #     pil_img = resizeImage(img, width, height)

    im = pil_img.convert('RGBA')
    disp_img = np.asarray(im)

    f = os.open('/dev/fb0', os.O_RDWR)  # fb0:HDMI, fb1:3.5inch LCD
    os.write(f, disp_img)
    os.close(f)

def main():
  bufClear()
  pil_img = Image.open('./LLL.png').convert('L')
  img = np.asarray(pil_img)
  
  pil_img = Image.fromarray(img.astype(np.uint8))
  pil_img = addBackImage(pil_img, *getResolution())
  addText(pil_img, *(500, 0), "hello!")
  
  printImg(pil_img, *getResolution(), PUT)

if __name__ == '__main__':
  main()
