""" helper function

author baiyu
"""

import sys

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import  GlaucomaTrain, GlaucomaTest, Dataset_FullImg, Dataset_DiscRegion, Dataset_FullImage_pool, Dataset_FullImage_pool_reverse
import math
import PIL


#from dataset import CIFAR100Train, CIFAR100Test
path = '/extracephonline/medai_data1/jundewu/data/GlaucomaClassification'
def get_network(args, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(gpu_device)
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'implicit':
        from models.implicitnet import implicitnet
        net = implicitnet()
    elif args.net == 'efficientnet':
        from models.efficientnet import EfficientNet
        net = EfficientNet.from_name('efficientnet-b4',gpu_device,args.type)


    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        #net = net.cuda(device = gpu_device)
        if distribution:
            net = torch.nn.DataParallel(net)
            net = net.cuda(device=gpu_device)
        else:
            net = net.cuda(device=gpu_device)

    return net


def get_training_dataloader(data, batch_size=16, num_workers=2, shuffle=True):
    """ training primary backbone
    """

    transform_train = transforms.Compose([
        transforms.Resize(((356, 268))),
        # transforms.RandomCrop((img_size, img_size)),  # padding=10
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_seg = transforms.Compose([
        transforms.Resize((356, 268)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
    ])
    if data[0] == 'LAG':
        prob = [2/3, 1/3]  # probability of class 0 = 1/11, of 1 = 1/10

        Glaucoma_training = Dataset_FullImg(path,data,transform = transform_train, transform_seg = transform_seg)
        reciprocal_weights = []
        for index in range(len(Glaucoma_training)):
            _,_,label,_ = Glaucoma_training.__getitem__(index)
            reciprocal_weights.append(prob[label])

        weights = (1 / torch.Tensor(reciprocal_weights))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(Glaucoma_training))

        Glaucoma_training_loader = DataLoader(
            Glaucoma_training, num_workers=num_workers, batch_size=batch_size, sampler = sampler)
    elif data[0] == 'REFUGETrain' or data[0] == 'REFUGEVal'or data[0] == 'REFUGETest':
        prob = [10 / 11, 1 / 11]  # probability of class 0 = 1/11, of 1 = 1/10
        Glaucoma_training = Dataset_FullImg(path, data, transform=transform_train, transform_seg=transform_seg)
        reciprocal_weights = []
        for index in range(len(Glaucoma_training)):
            _, _, label, _ = Glaucoma_training.__getitem__(index)
            reciprocal_weights.append(prob[label])

        weights = (1 / torch.Tensor(reciprocal_weights))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(Glaucoma_training))

        Glaucoma_training_loader = DataLoader(
            Glaucoma_training, num_workers=num_workers, batch_size=batch_size, sampler = sampler)
    elif data[0] == 'Tongren1st':
        prob = [4/7, 3/7]  # probability of class 0 = 1/11, of 1 = 1/10

        Glaucoma_training = Dataset_FullImg(path,data,transform = transform_train, transform_seg = transform_seg)
        reciprocal_weights = []
        for index in range(len(Glaucoma_training)):
            _,_,label,_ = Glaucoma_training.__getitem__(index)
            reciprocal_weights.append(prob[label])

        weights = (1 / torch.Tensor(reciprocal_weights))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(Glaucoma_training))

        Glaucoma_training_loader = DataLoader(
            Glaucoma_training, num_workers=num_workers, batch_size=batch_size, sampler = sampler)
    else:
        Glaucoma_training = Dataset_FullImg(path, data, transform=transform_train, transform_seg=transform_seg)
        Glaucoma_training_loader = DataLoader(
            Glaucoma_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Glaucoma_training_loader

def get_pool_dataloader(data, pool, batch_size=16, num_workers=2, shuffle=True, p = False):
    """ training primary backbone
    """

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomCrop((img_size, img_size)),  # padding=10
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_seg = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
    ])
    alpha = 0.6
    Glaucoma_pool = Dataset_FullImage_pool(path,data,pool, transform = transform_train, transform_seg = transform_seg)
    reciprocal_weights = []
    if p:
        for (name, difficulty) in pool:
            reciprocal_weights.append(numpy.power(difficulty, alpha))

        weights = (torch.Tensor(reciprocal_weights))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(pool))

        Glaucoma_pool_loader = DataLoader(
            Glaucoma_pool, num_workers=num_workers, batch_size=batch_size, sampler = sampler)
    else:
        Glaucoma_pool_loader = DataLoader(
            Glaucoma_pool, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Glaucoma_pool_loader

def get_pool_reverse_dataloader(data, pool, batch_size=16, num_workers=2, shuffle=True):
    """ training primary backbone
    """

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomCrop((img_size, img_size)),  # padding=10
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_seg = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
    ])
    Glaucoma_pool = Dataset_FullImage_pool_reverse(path,data,pool, transform = transform_train, transform_seg = transform_seg)
    Glaucoma_training_loader = DataLoader(
        Glaucoma_pool, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Glaucoma_training_loader

def get_implicit_dataloader(data, batch_size=16, num_workers=2, shuffle=True):
    """ Extra data for generating implicit labels
    """

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomCrop((img_size, img_size)),  # padding=10
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_seg = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
    ])
    # if data[0] == 'REFUGETrain' or data[0] == 'REFUGEVal'or data[0] == 'REFUGETest':
    #     prob = [10 / 11, 1 / 11]  # probability of class 0 = 1/11, of 1 = 1/10
    #     Glaucoma_training = Dataset_FullImg(path, data, transform=transform_train, transform_seg=transform_seg)
    #     reciprocal_weights = []
    #     for index in range(len(Glaucoma_training)):
    #         _, _, label, _ = Glaucoma_training.__getitem__(index)
    #         reciprocal_weights.append(prob[label])
    #
    #     weights = (1 / torch.Tensor(reciprocal_weights))
    #     sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(Glaucoma_training))
    #
    #     Glaucoma_implicit_loader = DataLoader(
    #         Glaucoma_training, num_workers=num_workers, batch_size=batch_size, sampler = sampler)
    # else:
    Glaucoma_training = Dataset_FullImg(path, data, transform=transform_train, transform_seg=transform_seg)
    Glaucoma_implicit_loader = DataLoader(
            Glaucoma_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Glaucoma_implicit_loader

def get_ib_dataloader(data, batch_size=16, num_workers=2, shuffle=True):
    """ unkonwn training pool of primary backbone
    """

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomCrop((img_size, img_size)),  # padding=10
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_seg = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
    ])

    Glaucoma_training = Dataset_FullImg(path,data,transform = transform_train, transform_seg = transform_seg)
    Glaucoma_implicit_loader = DataLoader(
        Glaucoma_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Glaucoma_implicit_loader

def get_valuation_dataloader(data, batch_size=16, num_workers=2, shuffle=True):
    """ valuationg data from the center of training
    """

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_seg = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


    Glaucoma_training = Dataset_FullImg(path,data,transform = transform_test, transform_seg = transform_seg)
    Glaucoma_training_loader = DataLoader(
        Glaucoma_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Glaucoma_training_loader

def get_test_dataloader(data, batch_size=16, num_workers=2, shuffle=True):
    """ test data
    """

    transform_test = transforms.Compose([
        transforms.Resize((356, 268)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_seg = transforms.Compose([
        transforms.Resize((356, 268)),
        transforms.ToTensor(),
    ])


    Glaucoma_Test = Dataset_FullImg(path,data,transform = transform_test, transform_seg = transform_seg)
    Glaucoma_test_loader = DataLoader(
        Glaucoma_Test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Glaucoma_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

def cka_loss(gram_featureA, gram_featureB):

    scaled_hsic = torch.dot(torch.flatten(gram_featureA),torch.flatten(gram_featureB))
    normalization_x = gram_featureA.norm()
    normalization_y = gram_featureB.norm()
    return scaled_hsic / (normalization_x * normalization_y)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)




