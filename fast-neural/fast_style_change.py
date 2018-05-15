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
    # out_path = images[3]

    if model == '0':
        model_ = 'results/epoch_2_Fri_May_11_19_25_56_2018_1.0_50.0.model'
    elif model == '1':
        model_ = 'results/epoch_2_Sat_May_12_22_52_27_2018_1.0_50.0.model'
    elif model == '2':
        model_ = 'results/epoch_2_Mon_May_14_22_38_32_2018_1.0_50.0.model'
    else:
        print('please enter a vaild style image.')
        sys.exit(1)
    out_path = str(time.ctime()).replace(' ', '_').replace(':', '_') + '_' + model + '.jpg'
    stylize(content_img, model_, out_path)
