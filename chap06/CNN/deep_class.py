# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from tools.deep_convnet import DeepConvNet
from PIL import Image
from tools.functions import softmax
import time
arg=sys.argv
network = DeepConvNet()  
img = np.array(Image.open(arg[1]).convert('L'))
img=img.reshape(1,1,28,28)/255.0



# パラメータのload
network.load_params("param/deep_params.pkl")
print("softmax:")
start=time.time()
ans=network.classi(img)
print(softmax(ans))
elapsed_time=time.time()-start
print("====================")
print("Classification:")
print(np.argmax(ans))
print("====================")
print ("elapsed_time:{:.4g}".format(elapsed_time*1000) + "[msec]")
