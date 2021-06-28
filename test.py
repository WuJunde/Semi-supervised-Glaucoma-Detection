#test.py
#!/usr/bin/env python3

""" test neuron network performace

"""

import argparse
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt


import argparse

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix

from torch.autograd import Variable

from conf import settings
from utils import get_network, get_test_dataloader

def eval_training():

    test_loss_seg = 0.0  # cost function error
    test_loss_class = 0.0
    total_number = len(Glaucoma_test_loader.dataset)
    groundtruths = []
    predictions = []
    for batch_index, (images, labels_seg, labels_class, name) in enumerate(Glaucoma_test_loader):
        images = Variable(images)
        labels_class = Variable(labels_class)
        labels_seg = Variable(labels_seg)

        labels_class = labels_class.cuda(device=GPUdevice)
        labels_class = labels_class.to(torch.float32)
        labels_seg = labels_seg.cuda(device=GPUdevice)
        images = images.cuda(device=GPUdevice)

        [outputs_class,outputs_implicit] = net(images)
        outputs_class = outputs_class.squeeze(1)

        # loss_class = loss_function_class(outputs_class, labels_class)
        # test_loss_class += loss_class.item()

        groundtruths.extend(labels_class.data.cpu().numpy())
        predictions.extend(nn.Sigmoid()(outputs_class).data.cpu().numpy())

    auc = roc_auc_score(groundtruths, predictions)
    prediction1 = []
    prediction2 = []
    prediction3 = []
    prediction4 = []
    prediction5 = []
    prediction6 = []
    prediction7 = []
    prediction8 = []
    prediction9 = []
    for score in predictions:
        if score >= 0.1:
            prediction1.append(1)
        else:
            prediction1.append(0)

        if score >= 0.2:
            prediction2.append(1)
        else:
            prediction2.append(0)

        if score >= 0.3:
            prediction3.append(1)
        else:
            prediction3.append(0)

        if score >= 0.4:
            prediction4.append(1)
        else:
            prediction4.append(0)

        if score >= 0.5:
            prediction5.append(1)
        else:
            prediction5.append(0)

        if score >= 0.6:
            prediction6.append(1)
        else:
            prediction6.append(0)

        if score >= 0.7:
            prediction7.append(1)
        else:
            prediction7.append(0)

        if score >= 0.8:
            prediction8.append(1)
        else:
            prediction8.append(0)

        if score >= 0.9:
            prediction9.append(1)
        else:
            prediction9.append(0)


    cm = confusion_matrix(groundtruths, prediction1)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction1)
    # add informations to tensorboard
    print('\t Test auc: %.4f' % auc)
    print('\t Test1 acc: %.4f' % acc)
    print('\t sen : %.4f' % sen)
    print('\t spec : %.4f' % spec)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    cm = confusion_matrix(groundtruths, prediction2)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction2)
    # add informations to tensorboard
    print('\t Test auc: %.4f' % auc)
    print('\t Test2 acc: %.4f' % acc)
    print('\t sen : %.4f' % sen)
    print('\t spec : %.4f' % spec)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    cm = confusion_matrix(groundtruths, prediction3)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction3)
    # add informations to tensorboard
    print('\t Test auc: %.4f' % auc)
    print('\t Test3 acc: %.4f' % acc)
    print('\t sen : %.4f' % sen)
    print('\t spec : %.4f' % spec)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    cm = confusion_matrix(groundtruths, prediction4)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction4)
    # add informations to tensorboard
    print('\t Test auc: %.4f' % auc)
    print('\t Test4 acc: %.4f' % acc)
    print('\t sen : %.4f' % sen)
    print('\t spec : %.4f' % spec)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    cm = confusion_matrix(groundtruths, prediction5)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction5)
    # add informations to tensorboard
    print('\t Test auc: %.4f' % auc)
    print('\t Test5 acc: %.4f' % acc)
    print('\t sen : %.4f' % sen)
    print('\t spec : %.4f' % spec)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    cm = confusion_matrix(groundtruths, prediction6)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction6)
    # add informations to tensorboard
    print('\t Test auc: %.4f' % auc)
    print('\t Test6 acc: %.4f' % acc)
    print('\t sen : %.4f' % sen)
    print('\t spec : %.4f' % spec)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    cm = confusion_matrix(groundtruths, prediction7)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction7)
    # add informations to tensorboard
    print('\t Test auc: %.4f' % auc)
    print('\t Test7 acc: %.4f' % acc)
    print('\t sen : %.4f' % sen)
    print('\t spec : %.4f' % spec)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    cm = confusion_matrix(groundtruths, prediction8)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction8)
    # add informations to tensorboard
    print('\t Test auc: %.4f' % auc)
    print('\t Test8 acc: %.4f' % acc)
    print('\t sen : %.4f' % sen)
    print('\t spec : %.4f' % spec)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    cm = confusion_matrix(groundtruths, prediction9)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction9)
    # add informations to tensorboard
    print('\t Test auc: %.4f' % auc)
    print('\t Test9 acc: %.4f' % acc)
    print('\t sen : %.4f' % sen)
    print('\t spec : %.4f' % spec)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    return auc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-type', type=str, default='map', help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-test_data', type=list, default=['LAG_test'], help='do we need dynamic pool')
    args = parser.parse_args()

    GPUdevice = torch.device('cuda', args.gpu_device)
    net = get_network(args, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = False)

    Glaucoma_test_loader = get_test_dataloader(
        args.test_data,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    net.load_state_dict(torch.load(args.weights), args.gpu)
    #print(net)
    net.eval()
    auc = eval_training()