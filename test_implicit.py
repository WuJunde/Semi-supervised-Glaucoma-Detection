#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt


import argparse
from skimage import io
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix

from torch.autograd import Variable
from models.implicitefficientnet import EfficientNet
from conf import settings
from utils import get_network, get_test_dataloader, get_implicit_dataloader

def visualize_map():
    for batch_index, (images, labels_seg, labels_class, name) in enumerate(Glaucoma_implicit_loader):
        images = Variable(images)
        labels_seg = Variable(labels_seg)

        labels_seg = labels_seg.cuda(device=GPUdevice)
        images = images.cuda(device=GPUdevice)
        labels_class = labels_class.cuda(device = GPUdevice)
        labels_class = labels_class.to(torch.float32)

        implicit_label = implicitnet(labels_seg, labels_class, images)

        num = implicit_label.size(0)
        implicit_label = implicit_label.squeeze(1).cpu().detach().numpy()
        for i in range(num):

            image_np = implicit_label[i, :, :]
            # image = Image.fromarray(image_np, 'L')
            io.imsave('./map/{:s}.jpg'.format(name[i]), np.uint8(image_np * 255))

        print('Updating implicit labels')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()

    GPUdevice = torch.device('cuda', args.gpu_device)
    implicitnet = EfficientNet.from_name('efficientnet-b4')
    implicitnet = implicitnet.cuda(device=GPUdevice)

    Glaucoma_implicit_loader = get_implicit_dataloader(
        settings.GLAUCOMA_TRAIN_MEAN,
        settings.GLAUCOMA_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    implicitnet.load_state_dict(torch.load(args.weights), args.gpu)
    #print(net)
    implicitnet.eval()
    auc = visualize_map()