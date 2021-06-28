# train.py
#!/usr/bin/env	python3

""" train network using pytorch

"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
from models.implicitefficientnet import EfficientNet
#from models.discriminatorlayer import discriminator
from conf import settings
import time
from utils import get_network, get_training_dataloader, get_test_dataloader, get_implicit_dataloader,get_valuation_dataloader, get_ib_dataloader,get_pool_dataloader,get_pool_reverse_dataloader, WarmUpLR, cka_loss

def train_base_implicit(epoch):
    auc = 0
    total_number = len(Glaucoma_training_loader.dataset)
    y_true = np.zeros(shape=(total_number))
    y_pred = np.zeros(shape=(total_number))
    net_frozen = get_network(args, use_gpu=args.gpu, gpu_device=GPUdevice,
                             distribution=False)  # replica net for preventing distribution problem
    for batch_index, (images, labels_seg, labels_class, name) in enumerate(Glaucoma_training_loader):   # stage1 training data update
        total_number = len(Glaucoma_training_loader.dataset)
        groundtruths = []
        predictions = []
        loss_class = 0
        loss_map = 0
        for batch_index, (images, labels_seg, labels_class, name) in enumerate(Glaucoma_training_loader):
            images = Variable(images)
            labels_class = Variable(labels_class)
            labels_seg = Variable(labels_seg)

            labels_class = labels_class.cuda(device=GPUdevice)
            labels_class = labels_class.to(torch.float32)
            labels_seg = labels_seg.cuda(device=GPUdevice)
            images = images.cuda(device=GPUdevice)

            # images = torch.cat((images, labels_seg), 1)
            outputs_class = net(images)

            optimizer.zero_grad()

            outputs_class = outputs_class.squeeze(1)

            loss_explicit = loss_function_class(outputs_class, labels_class)

            loss = loss_explicit
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(Glaucoma_training_loader) + batch_index + 1

            groundtruths.extend(labels_class.data.cpu().numpy())
            predictions.extend(nn.Sigmoid()(outputs_class).data.cpu().numpy())

            loss_class += loss_explicit.item()

        prediction = list(np.around(predictions))
        auc = roc_auc_score(groundtruths, prediction)
        cm = confusion_matrix(groundtruths, prediction)
        sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        acc = accuracy_score(groundtruths, prediction)
        writer.add_scalar('Train/AUC', auc, epoch)
        writer.add_scalar('Train/ACC', acc, epoch)
        writer.add_scalar('Train/SEN', sen, epoch)
        writer.add_scalar('Train/SPEC', spec, epoch)
        writer.add_scalar('Train/loss_class', loss_class / total_number, epoch)

def train_baseline(epoch):
    total_number = len(Glaucoma_training_loader.dataset)
    groundtruths = []
    predictions = []
    loss_class = 0
    loss_map = 0
    for batch_index, (images, labels_seg, labels_class, name) in enumerate(Glaucoma_training_loader):

        images = Variable(images)
        labels_class = Variable(labels_class)
        labels_seg = Variable(labels_seg)

        labels_class = labels_class.cuda(device = GPUdevice)
        labels_class = labels_class.to(torch.float32)
        labels_seg = labels_seg.cuda(device = GPUdevice)
        images = images.cuda(device = GPUdevice)

        #images = torch.cat((images, labels_seg), 1)
        outputs_class = net(images)

        optimizer.zero_grad()

        outputs_class = outputs_class.squeeze(1)

        loss_explicit = loss_function_class(outputs_class, labels_class)

        loss = loss_explicit
        loss.backward()
        optimizer.step()


        n_iter = (epoch - 1) * len(Glaucoma_training_loader) + batch_index + 1

        groundtruths.extend(labels_class.data.cpu().numpy())
        predictions.extend(nn.Sigmoid()(outputs_class).data.cpu().numpy())

        loss_class += loss_explicit.item()

    prediction = list(np.around(predictions))
    auc = roc_auc_score(groundtruths, prediction)
    cm = confusion_matrix(groundtruths, prediction)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction)
    writer.add_scalar('Train/AUC', auc, epoch)
    writer.add_scalar('Train/ACC', acc, epoch)
    writer.add_scalar('Train/SEN', sen, epoch)
    writer.add_scalar('Train/SPEC', spec, epoch)
    writer.add_scalar('Train/loss_class', loss_class/total_number, epoch)

def train_explicit(epoch):
    total_number = len(Glaucoma_training_loader.dataset)
    groundtruths = []
    predictions = []
    loss_class = 0
    loss_map = 0
    index = 1
    #print('total_number', total_number)
    for batch_index, (images, labels_seg, labels_class, img_path) in enumerate(Glaucoma_training_loader):

        images = Variable(images)
        labels_class = Variable(labels_class)
        labels_seg = Variable(labels_seg)

        labels_class = labels_class.cuda(device = GPUdevice)
        labels_class = labels_class.to(torch.float32)
        labels_seg = labels_seg.cuda(device = GPUdevice)
        images = images.cuda(device = GPUdevice)

        [outputs_class, outputs_implicit] = net(images)
        implicit_label = implicitnet(labels_seg,labels_class,images)

        optimizer.zero_grad()
        implicit_optimizer.zero_grad()

        outputs_class = outputs_class.squeeze(1)

        loss_explicit = loss_function_class(outputs_class, labels_class)
        if args.type == 'feature':
            loss_implicit = cka_loss(outputs_implicit, implicit_label)
        else:
            loss_implicit = loss_function_map(outputs_implicit,implicit_label)

        loss = loss_explicit + loss_implicit
        loss.backward()
        optimizer.step()


        n_iter = (epoch - 1) * len(Glaucoma_training_loader) + batch_index + 1

        labels_np = labels_class.data.cpu().numpy()
        out_class_np = nn.Sigmoid()(outputs_class).data.cpu().numpy()
        groundtruths.extend(labels_np)
        predictions.extend(out_class_np)

        if epoch % args.dpool == 0:
            pt = (1-labels_np) * (1 - out_class_np) + labels_np * out_class_np
            alpha = 0.8
            difficulty = 1 - pt
            p = (alpha * (1-labels_np) + labels_np) * np.power(difficulty, 2) * (1 / (1 + np.exp(-16 * (1-pe) + 8)))
            seed = np.random.rand()
            if seed <= p.mean() and index < len(pool):
                for i in range(min(args.b,len(img_path))):
                    pool[-index - i] = (img_path[i], difficulty[i])
                index += args.b
        #print('updating explicit',batch_index)

    prediction = list(np.around(predictions))
    auc = roc_auc_score(groundtruths, predictions)
    cm = confusion_matrix(groundtruths, prediction)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction)
    print('exchange times: %d' % index)
    writer.add_scalar('Train/AUC', auc, epoch)
    writer.add_scalar('Train/ACC', acc, epoch)
    writer.add_scalar('Train/SEN', sen, epoch)
    writer.add_scalar('Train/SPEC', spec, epoch)

def train_explicit_ib(epoch):           # train primary network on data B with only implicit loss

    total_number = len(Glaucoma_implicit_loader.dataset)
    loss_map = 0
    for batch_index, (images, labels_seg, labels_class, name) in enumerate(Glaucoma_implicit_loader):

        images = Variable(images)
        labels_seg = Variable(labels_seg)

        labels_seg = labels_seg.cuda(device = GPUdevice)
        images = images.cuda(device = GPUdevice)

        [outputs_class, outputs_implicit] = net(images)
        outputs_class = outputs_class.squeeze(1)
        sudo_class = outputs_class.data
        implicit_label = implicitnet(labels_seg,sudo_class,images)

        optimizer.zero_grad()
        implicit_optimizer.zero_grad()

        if args.type == 'feature':
            loss_implicit = cka_loss(outputs_implicit, implicit_label)
        else:
            loss_implicit = loss_function_map(outputs_implicit, implicit_label)
        loss = loss_implicit
        loss.backward()
        optimizer.step()

        loss_map += loss_implicit.item()

        #n_iter = (epoch - 1) * len(Glaucoma_implicit_loader) + batch_index + 1

    writer.add_scalar('Train/loss_implicit_converge',loss_map/total_number)

def train_implicit(epoch):
    net_frozen = get_network(args, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = False)              #replica net for preventing distribution problem
    if epoch > 15 and epoch <= 40:
        imp_lr = 1e-4
    elif epoch > 40:
        imp_lr = 1e-5
    elif epoch <= 15:
        imp_lr = 3e-4

    for batch_index, (images, labels_seg, labels_class, name_img) in enumerate(Glaucoma_implicit_loader):
        images = Variable(images)
        labels_seg = Variable(labels_seg)

        labels_seg = labels_seg.cuda(device = GPUdevice)
        images = images.cuda(device = GPUdevice)
        labels_class = labels_class.cuda(device = GPUdevice)

        [outputs_class, outputs_implicit] = net(images)
        outputs_class = outputs_class.squeeze(1)
        sudo_class = outputs_class.data

        implicit_label = implicitnet(labels_seg,sudo_class,images)

        optimizer.zero_grad()
        implicit_optimizer.zero_grad()

        if args.type == 'feature':
            loss_implicit = cka_loss(outputs_implicit, implicit_label)
        else:
            loss_implicit = loss_function_map(outputs_implicit, implicit_label)
        loss_prime = loss_implicit

        frozen_weights = OrderedDict((name, param) for (name, param) in net.named_parameters())

        grads = torch.autograd.grad(loss_prime, net.parameters(), retain_graph = True, create_graph=True, allow_unused = True)


        for ((name, param), grad) in zip(frozen_weights.items(), grads):
            try:
                frozen_weights[name] = param - imp_lr * grad
            except TypeError:
                frozen_weights[name] = param

        for batch_index_ib, (images_ib, labels_seg_ib, labels_class_ib, name_ib) in enumerate(Glaucoma_implicit_baseline_loader):      # increase performance on ib dataset

            images_ib = Variable(images_ib)
            labels_class_ib = Variable(labels_class_ib)

            labels_class_ib = labels_class_ib.cuda(device=GPUdevice)
            labels_class_ib = labels_class_ib.to(torch.float32)
            images_ib = images_ib.cuda(device=GPUdevice)

            outputs_class, outputs_implicit = net_frozen.forward(images_ib, frozen_weights)
            outputs_class = outputs_class.squeeze(1)
            loss_class_imp = loss_function_class(outputs_class, labels_class_ib)
            loss = loss_class_imp

            loss.backward()
            implicit_optimizer.step()
            break

        if args.type != 'feature' and args.vis:
            if (epoch + 1) % 2 == 0:
                num = implicit_label.size(0)
                implicit_label = implicit_label.squeeze(1).cpu().detach().numpy()

                if len(implicit_label.shape) == 4: ## nature image
                    image_np = implicit_label[0,:,:,:]
                    image_np = image_np.transpose(1,2,0)
                else: #map
                    image_np = implicit_label[0, :, :]
                    #image = Image.fromarray(image_np, 'L')
                io.imsave('./map_REFUGE_4b/{:d}_{:s}'.format(epoch+1,name_img[0].split('\\')[-1]),np.uint8(image_np*255))

    print('Updated implicit labels\tepoch: {:d}'.format(epoch))

def train_same_source(epoch):
    net_frozen = get_network(args, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = False)              #replica net for preventing distribution problem
    if epoch > 15 and epoch <= 40:
        imp_lr = 1e-4
    elif epoch > 40:
        imp_lr = 1e-5
    elif epoch <= 15:
        imp_lr = 3e-4
    for batch_index, (images, labels_seg, labels_class, name_img) in enumerate(Glaucoma_training_loader):
        images = Variable(images)
        labels_seg = Variable(labels_seg)
        labels_class = Variable(labels_class)

        labels_class = labels_class.cuda(device = GPUdevice)
        labels_class = labels_class.to(torch.float32)
        labels_seg = labels_seg.cuda(device = GPUdevice)
        images = images.cuda(device = GPUdevice)

        [outputs_class, outputs_implicit] = net(images)
        outputs_class = outputs_class.squeeze(1)
        sudo_class = outputs_class.data
        implicit_label = implicitnet(labels_seg,labels_class,images)

        optimizer.zero_grad()
        implicit_optimizer.zero_grad()

        loss_explicit = loss_function_class(outputs_class, labels_class)
        if args.type == 'feature':
            loss_implicit = cka_loss(outputs_implicit, implicit_label)
        else:
            loss_implicit = loss_function_map(outputs_implicit, implicit_label)
        loss_prime = loss_implicit + loss_explicit

        frozen_weights = OrderedDict((name, param) for (name, param) in net.named_parameters())

        grads = torch.autograd.grad(loss_prime, net.parameters(), retain_graph = True, create_graph=True, allow_unused = True)

        for ((name, param), grad) in zip(frozen_weights.items(), grads):
            try:
                frozen_weights[name] = param - imp_lr * grad
            except TypeError:
                frozen_weights[name] = param


        outputs_class, outputs_implicit = net_frozen.forward(images, frozen_weights)
        outputs_class = outputs_class.squeeze(1)
        loss_class_imp = loss_function_class(outputs_class, labels_class)
        loss = loss_class_imp

        loss.backward()
        implicit_optimizer.step()

        if args.type != 'feature' and args.vis:
            if (epoch + 1) % 2 == 0:
                num = implicit_label.size(0)
                implicit_label = implicit_label.squeeze(1).cpu().detach().numpy()

                if len(implicit_label.shape) == 4: ## nature image
                    image_np = implicit_label[0,:,:,:]
                    image_np = image_np.transpose(1,2,0)
                else: #map
                    image_np = implicit_label[0, :, :]
                    #image = Image.fromarray(image_np, 'L')
                io.imsave('./map_REFUGE_4b/{:d}_{:s}'.format(epoch+1,name_img[0].split('\\')[-1]),np.uint8(image_np*255))

        print('Updating implicit labels\tepoch: {:d}\tbatch: {:d}'.format(epoch, batch_index))

def valuation_training(epoch):

    test_loss_seg = 0.0 # cost function error
    test_loss_class = 0.0
    total_number = len(Glaucoma_valuation_loader.dataset)
    groundtruths = []
    predictions = []
    for batch_index, (images, labels_seg, labels_class, name) in enumerate(Glaucoma_valuation_loader):

        images = Variable(images)
        labels_class = Variable(labels_class)
        labels_seg = Variable(labels_seg)

        labels_class = labels_class.cuda(device = GPUdevice)
        labels_class = labels_class.to(torch.float32)
        labels_seg = labels_seg.cuda(device = GPUdevice)
        images = images.cuda(device = GPUdevice)

        [outputs_class, outputs_implicit] = net(images)
        outputs_class = outputs_class.squeeze(1)

        loss_class = loss_function_class(outputs_class, labels_class)
        test_loss_class += loss_class.item()

        groundtruths.extend(labels_class.data.cpu().numpy())
        predictions.extend(nn.Sigmoid()(outputs_class).data.cpu().numpy())

    prediction = list(np.around(predictions))
    auc = roc_auc_score(groundtruths, predictions)
    cm = confusion_matrix(groundtruths, prediction)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction)
    #add informations to tensorboard
    writer.add_scalar('valuation/Average class loss', test_loss_class / total_number, epoch)
    writer.add_scalar('valuation/AUC', auc, epoch)
    writer.add_scalar('valuation/ACC', acc, epoch)
    writer.add_scalar('valuation/SEN', sen, epoch)
    writer.add_scalar('valuation/SPEC', spec, epoch)
    print('\t Test auc: %.4f' %auc)
    print('\t Test acc: %.4f' % acc)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    return auc

def valuation_training_ib(epoch):

    test_loss_seg = 0.0 # cost function error
    test_loss_class = 0.0
    total_number = len(Glaucoma_implicit_loader.dataset)
    groundtruths = []
    predictions = []
    for batch_index, (images, labels_seg, labels_class, name) in enumerate(Glaucoma_implicit_loader):

        images = Variable(images)
        labels_class = Variable(labels_class)
        labels_seg = Variable(labels_seg)

        labels_class = labels_class.cuda(device = GPUdevice)
        labels_class = labels_class.to(torch.float32)
        labels_seg = labels_seg.cuda(device = GPUdevice)
        images = images.cuda(device = GPUdevice)

        [outputs_class, outputs_implicit] = net(images)
        outputs_class = outputs_class.squeeze(1)

        loss_class = loss_function_class(outputs_class, labels_class)
        test_loss_class += loss_class.item()

        groundtruths.extend(labels_class.data.cpu().numpy())
        predictions.extend(nn.Sigmoid()(outputs_class).data.cpu().numpy())

    prediction = list(np.around(predictions))
    auc = roc_auc_score(groundtruths, predictions)
    cm = confusion_matrix(groundtruths, prediction)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction)
    #add informations to tensorboard
    writer.add_scalar('valuation_ib/Average class loss', test_loss_class / total_number, epoch)
    writer.add_scalar('valuation_ib/AUC', auc, epoch)
    writer.add_scalar('valuation_ib/ACC', acc, epoch)
    writer.add_scalar('valuation_ib/SEN', sen, epoch)
    writer.add_scalar('valuation_ib/SPEC', spec, epoch)

    return auc

def eval_training(epoch):

    test_loss_seg = 0.0 # cost function error
    test_loss_class = 0.0
    total_number = len(Glaucoma_test_loader.dataset)
    groundtruths = []
    predictions = []
    for batch_index, (images, labels_seg, labels_class, name) in enumerate(Glaucoma_test_loader):

        images = Variable(images)
        labels_class = Variable(labels_class)
        labels_seg = Variable(labels_seg)

        labels_class = labels_class.cuda(device = GPUdevice)
        labels_class = labels_class.to(torch.float32)
        labels_seg = labels_seg.cuda(device = GPUdevice)
        images = images.cuda(device = GPUdevice)

        [outputs_class, outputs_implicit] = net(images)
        outputs_class = outputs_class.squeeze(1)

        loss_class = loss_function_class(outputs_class, labels_class)
        test_loss_class += loss_class.item()

        groundtruths.extend(labels_class.data.cpu().numpy())
        predictions.extend(nn.Sigmoid()(outputs_class).data.cpu().numpy())

    prediction = list(np.around(predictions))
    auc = roc_auc_score(groundtruths, predictions)
    cm = confusion_matrix(groundtruths, prediction)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    acc = accuracy_score(groundtruths, prediction)
    #add informations to tensorboard
    writer.add_scalar('Test/Average class loss', test_loss_class / total_number, epoch)
    writer.add_scalar('Test/AUC', auc, epoch)
    writer.add_scalar('Test/ACC', acc, epoch)
    writer.add_scalar('Test/SEN', sen, epoch)
    writer.add_scalar('Test/SPEC', spec, epoch)
    print('\t Test auc: %.4f' %auc)
    print('\t Test acc: %.4f' % acc)
    print('\t Confusion Matrix:\n %s\n' % str(cm))

    return auc

def ini_test_paper_pool():
    pool = []
    summary = 0
    for batch_index, (images, labels_seg, labels_class, img_path) in enumerate(Glaucoma_training_loader):
        seed = np.random.rand()
        if seed <= 0.3:
            images = Variable(images)
            labels_class = Variable(labels_class)
            labels_seg = Variable(labels_seg)

            labels_class = labels_class.cuda(device=GPUdevice)
            labels_class = labels_class.to(torch.float32)
            labels_seg = labels_seg.cuda(device=GPUdevice)
            images = images.cuda(device=GPUdevice)

            [outputs_class, outputs_implicit] = net(images)
            outputs_class = outputs_class.squeeze(1)

            labels_np = labels_class.data.cpu().numpy()
            out_class_np = nn.Sigmoid()(outputs_class).data.cpu().numpy()

            pt = (1 - labels_np) * (1 - out_class_np) + labels_np * out_class_np
            alpha = 0.7
            difficulty = 1 - pt

            for i in range(min(args.b,len(img_path))):
                pool.append((img_path[i], difficulty[i]))
        else:
            continue
    pe = 0.5
    pool.sort(key=lambda x: x[1], reverse=True)
    return pool,pe

def update_test_paper_pool(pool):
    summary = 0
    pool_dict = dict(pool)
    for batch_index, (images, labels_seg, labels_class, img_path) in enumerate(Glaucoma_pool_loader):

        images = Variable(images)
        labels_class = Variable(labels_class)
        labels_seg = Variable(labels_seg)

        labels_class = labels_class.cuda(device = GPUdevice)
        labels_class = labels_class.to(torch.float32)
        labels_seg = labels_seg.cuda(device = GPUdevice)
        images = images.cuda(device = GPUdevice)

        [outputs_class, outputs_implicit] = net(images)
        outputs_class = outputs_class.squeeze(1)

        labels_np = labels_class.data.cpu().numpy()
        out_class_np = nn.Sigmoid()(outputs_class).data.cpu().numpy()

        pt = (1 - labels_np) * (1 - out_class_np) + labels_np * out_class_np
        alpha = 0.7
        difficulty = 1 - pt

        for i in range(min(args.b, len(img_path))):
            pool_dict[img_path[i]] = difficulty[i]

        summary += sum(difficulty)

    pe = summary/len(pool)
    pool = sorted(pool_dict.items(), key = lambda x:x[1],reverse = True)

    print('average difficulty %.4f' % pe)
    return pool,pe


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-type', type=str, default='map', help='net type')
    parser.add_argument('-vis', type=bool, default=False, help='visualization')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=3e-4, help='initial learning rate')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-implicitweights', type=str, default=0, help='the weights file you want to test')
    parser.add_argument('-dpool', type=int, default=10, help='do we need dynamic pool')
    parser.add_argument('-train_data', type=list, default=['RIM-ONEv3_SIL'], help='do we need dynamic pool')
    parser.add_argument('-test_data', type=list, default=['RIM-ONEv3_SIL','RIM-ONEv3_SIR'], help='do we need dynamic pool')
    parser.add_argument('-implicit_data', type=list, default=['MESSIDOR'], help='do we need dynamic pool')
    parser.add_argument('-pool_data', type=list, default=['LAG_test'], help='do we need dynamic pool')
    parser.add_argument('-val_data', type=list, default=['LAG_test'], help='do we need dynamic pool')
    args = parser.parse_args()

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = False)

    implicitnet = EfficientNet.from_name('efficientnet-b4', args.type)
    implicitnet = implicitnet.cuda(device=GPUdevice)

    # discriminator = discriminator(GPUdevice, 'map')
    # discriminator = discriminator.cuda(device=GPUdevice)



    if args.weights != 0:
        net.load_state_dict(torch.load(args.weights), args.gpu)
    if args.implicitweights != 0:
        implicitnet.load_state_dict(torch.load(args.implicitweights), args.gpu)

    #data preprocessing:

    Glaucoma_training_loader = get_training_dataloader(
        args.train_data,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    Glaucoma_training_loader_batch1 = get_training_dataloader(
        args.train_data,
        num_workers=args.w,
        batch_size=1,
        shuffle=args.s
    )

    Glaucoma_implicit_loader = get_implicit_dataloader(
        args.implicit_data,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )



    Glaucoma_implicit_baseline_loader = get_ib_dataloader(
        args.pool_data,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    Glaucoma_valuation_loader = get_valuation_dataloader(
        args.val_data,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    Glaucoma_test_loader = get_test_dataloader(
        args.test_data,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
    loss_function_class = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_function_map = torch.nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer.add_param_group({'params': discriminator.parameters()})
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

    implicit_optimizer = optim.Adam(implicitnet.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #implicit_optimizer.add_param_group({'params': discriminator.parameters()})
    implicit_scheduler = optim.lr_scheduler.StepLR(implicit_optimizer, step_size=10, gamma=0.5)

    imp_lr = args.imp_lr

    iter_per_epoch = len(Glaucoma_training_loader)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    #input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
    #writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(args.epoch_ini, settings.EPOCH):
        if epoch == args.epoch_ini :
            time_start = time.time()
            pool, pe = ini_test_paper_pool()
            Glaucoma_pool_loader = get_pool_dataloader(
                args.train_data,
                pool,
                num_workers=args.w,
                batch_size=args.b,
                shuffle=args.s,
                p = False
            )
            time_end = time.time()
            print('time_for_ini_pool ', time_end - time_start)
        net.train()
        #train_implicit(epoch)
        time_start = time.time()
        train_explicit_ib(epoch)
        time_end = time.time()
        print('time_for_explicit_ib ', time_end - time_start)
        #train_baseline(epoch)
        # time_start = time.time()
        # Glaucoma_training_loader = get_pool_reverse_dataloader(
        #     args.train_data,
        #     pool,
        #     num_workers=args.w,
        #     batch_size=args.b,
        #     shuffle=args.s,
        # )
        # time_end = time.time()
        # print('time_for_reverse_pool ', time_end - time_start)
        time_start = time.time()
        train_explicit(epoch)
        time_end = time.time()
        print('time_for_explicit ', time_end - time_start)
        # if epoch % args.dpool == 0:
        #     time_start = time.time()
        #     pool,pe = update_test_paper_pool(pool)
        #     Glaucoma_pool_loader = get_pool_dataloader(
        #         args.train_data,
        #         pool,
        #         num_workers=args.w,
        #         batch_size=args.b,
        #         shuffle=args.s,
        #         p =False
        #     )
        #     time_end = time.time()
        #     print('time_for_update_pool', time_end - time_start)
        # print('start_train_implicit')
        time_start = time.time()
        train_implicit(epoch)
        time_end = time.time()
        print('time_for_implicit ', time_end - time_start)
        net.eval()
        _ = valuation_training(epoch)
        #_ = valuation_training_ib(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            torch.save(implicitnet.state_dict(), checkpoint_path.format(net = 'implicitnet',epoch = epoch, type = 'best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            print('Saving regular checkpoint')
            print(checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
            torch.save(implicitnet.state_dict(), checkpoint_path.format(net = 'implicitnet',epoch = epoch, type = 'regular'))

    writer.close()
