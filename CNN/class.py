# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from PIL import Image
from tools.simple_convnet import SimpleConvNet
from tools.functions import softmax


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
print(softmax(ans))
print("Classification")
print(np.argmax(ans))
