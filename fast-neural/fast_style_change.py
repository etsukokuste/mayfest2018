# originally written by Abhishek Kadian
# https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/neural_style.py

"""
how to use:
1. download Microsoft COCO dataset (or similar dataset)
2. $python fast_style_train.py content_img model
   model = {0:Balla, 1:Dubuffet, 2:Gogh, 3:Munch}
"""

import sys
import time
import torch
from torch.autograd import Variable

from . import utils
from .transformer_net import TransformerNet


def stylize(content_image, model, output_image_path, cuda=False):
    start = time.time()
    content_image = utils.tensor_load_rgbimage(content_image)
    content_image = content_image.unsqueeze(0)
    if cuda:
        content_image = content_image.cuda()

    content_image = Variable(utils.preprocess_batch(content_image), requires_grad=False)
    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(model))
    if cuda:
        style_model.cuda()

    output = style_model(content_image)
    stop = time.time()
    print('stylized in '+str(stop - start) + '(s).')
    utils.tensor_save_bgrimage(output.data[0], output_image_path, cuda)


if __name__ == '__main__':
    images = sys.argv
    content_img = images[1]
    model = images[2]
    cuda = torch.cuda.is_available()

    if model == '0':
        pretrained_model = 'models/balla_circular_planes.model'
    elif model == '1':
        pretrained_model = 'models/dubuffet_fugitive_presences.model'
    elif model == '2':
        pretrained_model = 'models/gogh_irises.model'
    elif model == '3':
        pretrained_model = 'models/munch_the_scream.model'
    else:
        print('please enter a vaild style image.')
        sys.exit(1)

    out_path = str(time.ctime()).replace(' ', '_').replace(':', '_') + '_' + model + '.jpg'
    stylize(content_img, pretrained_model, out_path, cuda)