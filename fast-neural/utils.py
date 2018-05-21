# originally written by Abhishek Kadian
# https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py

import os
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.models import vgg16

from vgg16 import Vgg16

def tensor_load_rgbimage(filename, scale=2):
    # 画像 -> tensor
    img = Image.open(filename)
    img_size = img.size[0] * img.size[1]
    while img_size > 500000: # 画像サイズを圧縮
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
        img_size = img.size[0] * img.size[1]
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def tensor_save_rgbimage(tensor, filename, cuda=False):
    # tensor -> 画像
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

def tensor_save_bgrimage(tensor, filename, cuda=False):
    # tensor -> 画像
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def subtract_imagenet_mean_batch(batch, cuda):
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    mean = dtype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))
    return batch

def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch

def init_vgg16(model_folder):
    # VGG16の重みを初期化
    if not os.path.exists(model_folder+'/vgg16.weight'):
        vgg_load = vgg16(pretrained=True)
        vgg = Vgg16()
        for param, param_pre in zip(vgg.parameters(), vgg_load.parameters()):
            param.data = param_pre.data
        torch.save(vgg.state_dict(), model_folder+'/vgg16.weight')