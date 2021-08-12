import numpy as np
from PIL import Image, ImageFilter

im = Image.open('./LLL.png').convert('L')
img = np.asarray(im)
print(img.dtype)  # データ型

pil_img = Image.fromarray(img)
pil_img.save('./out.png')

print(img.size)