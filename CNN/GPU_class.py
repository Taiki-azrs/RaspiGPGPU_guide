# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from PIL import Image
from tools.GPU_simple_convnet import SimpleConvNet
from tools.functions import softmax
import time

arg=sys.argv
max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
img = np.array(Image.open(arg[1]).convert('L'))
img=img.reshape(1,1,28,28)/255.0

# パラメータのload
network.load_params("params.pkl")

ans=network.classi(img)
print("softmax:")
start=time.time()
print(softmax(ans))
elapsed_time=time.time()-start
print("====================")
print("Classification:")
print(np.argmax(ans))
print("====================")
print ("elapsed_time:{:.4g}".format(elapsed_time*1000) + "[msec]")
