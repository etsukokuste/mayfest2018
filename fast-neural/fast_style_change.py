import sys
import time

import torch
from torch.autograd import Variable

import utils
from transformer_net import TransformerNet


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
        model_ = 'models/balla_circular_planes.model'
    elif model == '1':
        model_ = 'models/dubuffet_fugitive_presences.model'
    elif model == '2':
        model_ = 'models/gogh_irises.model'
    elif model == '3':
        model_ = 'models/munch_the_scream.model'
    else:
        print('please enter a vaild style image.')
        sys.exit(1)
    out_path = str(time.ctime()).replace(' ', '_').replace(':', '_') + '_' + model + '.jpg'
    stylize(content_img, model_, out_path, cuda)
