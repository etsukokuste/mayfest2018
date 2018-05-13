import os
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import utils
from transformer_net import TransformerNet
from vgg16 import Vgg16


def train(data_dir, style_image, cuda=False, epochs=2, batch_size=4, image_size=256, seed=42, lr=1e-3, content_weight=1.0, style_weight=50.0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}

    else:
        kwargs = {}

    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])

    train_dataset = datasets.ImageFolder(data_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, **kwargs)

    transformer = TransformerNet()
    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16()
    vgg_model_dir = 'models'  # VGG model dir
    utils.init_vgg16(vgg_model_dir)
    vgg.load_state_dict(torch.load(vgg_model_dir+'/vgg16.weight'))

    if cuda:
        transformer.cuda()
        vgg.cuda()

    for param in vgg.parameters():
        param.requires_grad = False

    style = utils.tensor_load_rgbimage(style_image)
    style = style.repeat(batch_size, 1, 1, 1)
    style = utils.preprocess_batch(style)
    if cuda:
        style = style.cuda()
    style_v = Variable(style, requires_grad=False)
    style_v = utils.subtract_imagenet_mean_batch(style_v, cuda)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    print('start training...')
    for e in range(epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))
            if cuda:
                x = x.cuda()
            y = transformer(x)
            xc = Variable(x.data.clone(), requires_grad=True)

            y = utils.subtract_imagenet_mean_batch(y, cuda)
            xc = utils.subtract_imagenet_mean_batch(xc, cuda)

            features_y = vgg(y)
            features_xc = vgg(xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)
            content_loss = content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            if cuda:
                agg_content_loss += content_loss.data.item()
                agg_style_loss += style_loss.data.item()
            else:
                agg_content_loss += content_loss.data[0]
                agg_style_loss += style_loss.data[0]

            if (batch_id + 1) % 20 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(

                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)


    # save model
    transformer.eval()
    transformer.cpu()
    save_model_filename = "/epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + str(content_weight) + "_" + str(style_weight) + ".model"
    save_model_dir = 'models' # save model dir
    save_model_path = save_model_dir + save_model_filename
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


if __name__ == '__main__':
    data_dir = 'data'
    style_image = 'images/1590.jpg'

    cuda = torch.cuda.is_available()
    print(cuda)

    train(data_dir, style_image, cuda=cuda)