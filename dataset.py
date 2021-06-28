""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd

class GlaucomaTrain(Dataset):


    def __init__(self, path, names, transform=None, transform_seg_label = None, concat = None, multi = None):
        # names for multi datasets
        name_path = []
        name_path_seg = []
        self.Glaucoma_path = []
        self.NonGlaucoma_path = []
        self.Glaucoma_list = []
        self.NonGlaucoma_list = []
        self.Glaucoma_len = []
        self.NonGlaucoma_len = []
        self.length_Glau = 0
        self.length_Non = 0
        self.Glaucoma_path_seg = []
        self.NonGlaucoma_path_seg = []
        self.Glaucoma_list_seg = []
        self.NonGlaucoma_list_seg = []
        for idx, dataset_name in enumerate(names):
                name_path.append(os.path.join(os.path.join(path, 'Images'), dataset_name))
                name_path_seg.append(os.path.join(os.path.join(path, 'CupDiscMasks'), dataset_name))
                self.Glaucoma_path.append(os.path.join(name_path[idx], 'Glaucoma'))
                self.NonGlaucoma_path.append(os.path.join(name_path[idx], 'NonGlaucoma'))
                Glaucoma_list_temp = os.listdir(self.Glaucoma_path[idx])
                self.Glaucoma_len.append(len(Glaucoma_list_temp))
                self.Glaucoma_list.extend(Glaucoma_list_temp)
                NonGlaucoma_list_temp = os.listdir(self.NonGlaucoma_path[idx])
                self.NonGlaucoma_len.append(len(NonGlaucoma_list_temp))
                self.NonGlaucoma_list.extend(NonGlaucoma_list_temp)


                self.Glaucoma_path_seg.append(os.path.join(name_path_seg[idx], 'Glaucoma'))
                self.NonGlaucoma_path_seg.append(os.path.join(name_path_seg[idx], 'NonGlaucoma'))


        self.transform = transform
        self.transform_seg_label = transform_seg_label
        self.concat = concat
        self.multi = multi

    def __len__(self):

        return len(self.Glaucoma_list) + len(self.NonGlaucoma_list)

    def __getitem__(self, index):
        length_Glau = 0
        length_Non = 0
        if self.concat or self.multi:
            if index < len(self.Glaucoma_list): # Glau data
                label_class = 1
                for i in range(len(self.Glaucoma_len)):
                    length_Glau += self.Glaucoma_len[i]
                    if index < length_Glau:     #which dataset
                        image = io.imread(os.path.join(self.Glaucoma_path[i], self.Glaucoma_list[index]))
                        image_seg = io.imread(os.path.join(self.Glaucoma_path_seg[i], '.'.join((self.Glaucoma_list[index].split('.',1)[0] , 'png' ))))
                        img_name = self.Glaucoma_list[index].split('.',1)[0]   # seg image has the same name
                        image_seg_concat = image_seg[:, :, np.newaxis]
                        if self.concat:
                            try:
                                image_concat = np.concatenate((image,image_seg_concat),axis=2)
                            except:
                                while image.shape[0] != image_seg.shape[0] or image.shape[1] != image_seg.shape[1]:
                                    H, W = image_seg.shape[0], image_seg.shape[1]
                                    image = image[:H, :W, :]
                                image_concat = np.concatenate((image, image_seg_concat), axis=2)
                            image_concat = Image.fromarray(image_concat)
                            image = image_concat
                            image = self.transform(image)
                            image_seg = Image.fromarray(image_seg)
                            image_seg = self.transform_seg_label(image_seg)
                            return image, image_seg, label_class, img_name
                        else: #multi
                            image = Image.fromarray(image)
                            image_seg = Image.fromarray(image_seg)
                            image_seg = self.transform_seg_label(image_seg)
                            image = self.transform(image)
                            return image, image_seg, label_class, img_name
                    else:
                        continue
            else: # Non Glau data
                non_index = index - len(self.Glaucoma_list)
                label_class = 0
                for i in range(len(self.NonGlaucoma_len)):
                    length_Non += self.NonGlaucoma_len[i]
                    if non_index < length_Non:
                        image = io.imread(os.path.join(self.NonGlaucoma_path[i], self.NonGlaucoma_list[non_index]))
                        image_seg = io.imread(os.path.join(self.NonGlaucoma_path_seg[i], '.'.join((self.NonGlaucoma_list[non_index].split('.',1)[0] , 'png' ))))
                        img_name = self.NonGlaucoma_list[non_index].split('.',1)[0]  # seg image has the same name
                        image_seg_concat = image_seg[:, :, np.newaxis]
                        if self.concat:
                            try:
                                image_concat = np.concatenate((image,image_seg_concat),axis=2)
                            except:
                                while image.shape[0] != image_seg.shape[0] or image.shape[1] != image_seg.shape[1]:
                                    H, W = image_seg.shape[0], image_seg.shape[1]
                                    image = image[:H, :W, :]
                                image_concat = np.concatenate((image, image_seg_concat), axis=2)
                            image_concat = Image.fromarray(image_concat)
                            image = image_concat
                            image = self.transform(image)
                            image_seg = Image.fromarray(image_seg)
                            image_seg = self.transform_seg_label(image_seg)
                            return image, image_seg, label_class, img_name
                        else: #multi
                            image = Image.fromarray(image)
                            image_seg = Image.fromarray(image_seg)
                            image_seg = self.transform_seg_label(image_seg)
                            image = self.transform(image)
                            return image, image_seg, label_class, img_name
                    else:
                        continue

        else:   #pure no concat or multi (no loading segmentation data)
            if index < len(self.Glaucoma_list): # Glau data
                label = 1
                for i in range(len(self.Glaucoma_len)):
                    length_Glau += self.Glaucoma_len[i]
                    if index < length_Glau:
                        image = io.imread(os.path.join(self.Glaucoma_path[i], self.Glaucoma_list[index]))
                        image = Image.fromarray(image)
                        if self.transform:
                            image = self.transform(image)
                        return image, label
                    else:
                        continue
            else: # Non Glau data
                non_index = index - len(self.Glaucoma_list)
                label = 0
                for i in range(len(self.NonGlaucoma_len)):
                    length_Non += self.NonGlaucoma_len[i]
                    if non_index < length_Non:
                        image = io.imread(os.path.join(self.NonGlaucoma_path[i], self.NonGlaucoma_list[non_index]))
                        image = Image.fromarray(image)
                        if self.transform:
                            image = self.transform(image)
                        return image, label
                    else:
                        continue

class GlaucomaTest(Dataset):


    def __init__(self, path, names, transform=None, transform_seg_label = None, concat = None, multi = None):
        # names for multi datasets
        if concat or multi:
            name_path = []
            name_path_seg = []
            self.Glaucoma_path = []
            self.NonGlaucoma_path = []
            self.Glaucoma_list = []
            self.NonGlaucoma_list = []
            self.Glaucoma_len = []
            self.NonGlaucoma_len = []
            self.length_Glau = 0
            self.length_Non = 0
            self.Glaucoma_path_seg = []
            self.NonGlaucoma_path_seg = []
            self.Glaucoma_list_seg = []
            self.NonGlaucoma_list_seg = []
            for idx, dataset_name in enumerate(names):
                    name_path.append(os.path.join(os.path.join(path, 'Images'), dataset_name))
                    name_path_seg.append(os.path.join(os.path.join(path, 'CupDiscMasks'), dataset_name))
                    self.Glaucoma_path.append(os.path.join(name_path[idx], 'Glaucoma'))
                    self.NonGlaucoma_path.append(os.path.join(name_path[idx], 'NonGlaucoma'))
                    Glaucoma_list_temp = os.listdir(self.Glaucoma_path[idx])
                    self.Glaucoma_len.append(len(Glaucoma_list_temp))
                    self.Glaucoma_list.extend(Glaucoma_list_temp)
                    NonGlaucoma_list_temp = os.listdir(self.NonGlaucoma_path[idx])
                    self.NonGlaucoma_len.append(len(NonGlaucoma_list_temp))
                    self.NonGlaucoma_list.extend(NonGlaucoma_list_temp)


                    self.Glaucoma_path_seg.append(os.path.join(name_path_seg[idx], 'Glaucoma'))
                    self.NonGlaucoma_path_seg.append(os.path.join(name_path_seg[idx], 'NonGlaucoma'))
        else:
            name_path = []
            self.Glaucoma_path = []
            self.NonGlaucoma_path = []
            self.Glaucoma_list = []
            self.NonGlaucoma_list = []
            self.Glaucoma_len = []
            self.NonGlaucoma_len = []
            self.length_Glau = 0
            self.length_Non = 0
            for idx, dataset_name in enumerate(names):
                    name_path.append(os.path.join(path, dataset_name))
                    self.Glaucoma_path.append(os.path.join(name_path[idx], 'Glaucoma'))
                    self.NonGlaucoma_path.append(os.path.join(name_path[idx], 'NonGlaucoma'))
                    Glaucoma_list_temp = os.listdir(self.Glaucoma_path[idx])
                    self.Glaucoma_len.append(len(Glaucoma_list_temp))
                    self.Glaucoma_list.extend(Glaucoma_list_temp)
                    NonGlaucoma_list_temp = os.listdir(self.NonGlaucoma_path[idx])
                    self.NonGlaucoma_len.append(len(NonGlaucoma_list_temp))
                    self.NonGlaucoma_list.extend(NonGlaucoma_list_temp)


        self.transform = transform
        self.transform_seg_label = transform_seg_label
        self.concat = concat
        self.multi = multi

    def __len__(self):

        return len(self.Glaucoma_list) + len(self.NonGlaucoma_list)

    def __getitem__(self, index):
        length_Glau = 0
        length_Non = 0
        if self.concat or self.multi:
            if index < len(self.Glaucoma_list): # Glau data
                label_class = 1
                for i in range(len(self.Glaucoma_len)):
                    length_Glau += self.Glaucoma_len[i]
                    if index < length_Glau:
                        image = io.imread(os.path.join(self.Glaucoma_path[i], self.Glaucoma_list[index]))
                        image_seg = io.imread(os.path.join(self.Glaucoma_path_seg[i], '.'.join((self.Glaucoma_list[index].split('.',1)[0] , 'png' ))))
                        img_name = self.Glaucoma_list[index]   # seg image has the same name

                        if self.concat:
                            try:
                                image_seg = image_seg[:, :, np.newaxis]
                                image_concat = np.concatenate((image,image_seg),axis=2)
                            except:
                                while image.shape[0] != image_seg.shape[0] or image.shape[1] != image_seg.shape[1]:
                                    H, W = image_seg.shape[0], image_seg.shape[1]
                                    image = image[:H, :W, :]
                                image_concat = np.concatenate((image, image_seg), axis=2)
                            image_concat = Image.fromarray(image_concat)
                            image = image_concat
                            image = self.transform(image)
                            image_seg = Image.fromarray(image_seg)
                            image_seg = self.transform_seg_label(image_seg)
                            return image, image_seg, label_class, img_name
                        else: #multi
                            image = Image.fromarray(image)
                            image_seg = Image.fromarray(image_seg)
                            image_seg = self.transform_seg_label(image_seg)
                            image = self.transform(image)
                            return image, label_class, image_seg, img_name
                    else:
                        continue
            else: # Non Glau data
                non_index = index - len(self.Glaucoma_list)
                label_class = 0
                for i in range(len(self.NonGlaucoma_len)):
                    length_Non += self.NonGlaucoma_len[i]
                    if non_index < length_Non:
                        image = io.imread(os.path.join(self.NonGlaucoma_path[i], self.NonGlaucoma_list[non_index]))
                        image_seg = io.imread(os.path.join(self.NonGlaucoma_path_seg[i], '.'.join((self.NonGlaucoma_list[non_index].split('.',1)[0] , 'png' ))))
                        img_name = self.Glaucoma_list[index]  # seg image has the same name
                        if self.concat:
                            try:
                                image_seg = image_seg[:, :, np.newaxis]
                                image_concat = np.concatenate((image,image_seg),axis=2)
                            except:
                                while image.shape[0] != image_seg.shape[0] or image.shape[1] != image_seg.shape[1]:
                                    H, W = image_seg.shape[0], image_seg.shape[1]
                                    image = image[:H, :W, :]
                                image_concat = np.concatenate((image, image_seg), axis=2)
                            image_concat = Image.fromarray(image_concat)
                            image = image_concat
                            image = self.transform(image)
                            image_seg = Image.fromarray(image_seg)
                            image_seg = self.transform_seg_label(image_seg)
                            return image, image_seg, label_class, img_name
                        else: #multi
                            image = Image.fromarray(image)
                            image_seg = Image.fromarray(image_seg)
                            image_seg = self.transform_seg_label(image_seg)
                            image = self.transform(image)
                            return image, label_class, image_seg, img_name
                    else:
                        continue

        else:   #pure no concat or multi (no loading segmentation data)
            if index < len(self.Glaucoma_list): # Glau data
                label = 1
                for i in range(len(self.Glaucoma_len)):
                    length_Glau += self.Glaucoma_len[i]
                    if index < length_Glau:
                        image = io.imread(os.path.join(self.Glaucoma_path[i], self.Glaucoma_list[index]))
                        image = Image.fromarray(image)
                        if self.transform:
                            image = self.transform(image)
                        return image, label
                    else:
                        continue
            else: # Non Glau data
                non_index = index - len(self.Glaucoma_list)
                label = 0
                for i in range(len(self.NonGlaucoma_len)):
                    length_Non += self.NonGlaucoma_len[i]
                    if non_index < length_Non:
                        image = io.imread(os.path.join(self.NonGlaucoma_path[i], self.NonGlaucoma_list[non_index]))
                        image = Image.fromarray(image)
                        if self.transform:
                            image = self.transform(image)
                        return image, label
                    else:
                        continue

class Dataset_DiscRegion(Dataset):
    def __init__(self, data_path, DF, transform = None, transform_seg = None):
        if DF[0] == 'train' or DF[0] == 'val' or DF[0] == 'imp' or DF[0] == 'ORIGAtrain' or DF[0] == 'ORIGAimp' or DF[0] == 'ORIGAtest':
            self.DF = pd.read_csv(data_path + '/' + 'Glaucoma' + '_' + DF[0] + '.csv', encoding='gbk')
        elif DF[0] == 'LAG_train' or DF[0] == 'LAG_test':
            self.DF = pd.read_csv(data_path + '/' + DF[0] + '.csv',encoding='gbk')
        elif DF[0] == 'REFUGETrain' or DF[0] == 'REFUGEVal' or DF[0] == 'REFUGETest':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        else:
            DF_all = pd.read_csv(data_path + '/' + 'GlaucomaLabels_6Center_191206_cleaned.csv', encoding='gbk')
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = DF_all.loc[DF_all['center'] == split]
                DF_this = DF_this.reset_index(drop=True)
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        self.DF.index = range(0, len(self.DF))
        self.data_path = data_path
        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, index):

        """Get the images"""
        imgName = self.DF.loc[index, 'imgName']
        data_path = self.data_path + '/' + 'Images'
        fullPathName = os.path.join(data_path, imgName)
        fullPathName = fullPathName.replace('\\', '/')

        Img0 = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)
        Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)

        discFlag = self.DF.loc[index, 'discFlag']
        if discFlag == 1:
            xmin = self.DF.loc[index, 'xmin']
            ymin = self.DF.loc[index, 'ymin']
            xmax = self.DF.loc[index, 'xmax']
            ymax = self.DF.loc[index, 'ymax']
            width = self.DF.loc[index, 'width']
            height = self.DF.loc[index, 'height']

            discHeight = ymax - ymin
            discWidth = xmax - xmin
            discX = int((xmax + xmin) / 2)
            discY = int((ymax + ymin) / 2)

            cropRadius1 = 1.5 * np.maximum(discHeight, discWidth)
            cropRadius2 = 0.2 * (np.maximum(height, width))
            # cropRadius = int(np.maximum(cropRadius1, cropRadius2))
            cropRadius = int(cropRadius1)
            # print(cropRadius, cropRadius1, cropRadius2)

            borderWidth = int(1.3 * cropRadius)
            ImgPad = cv2.copyMakeBorder(Img, borderWidth, borderWidth, borderWidth, borderWidth, cv2.BORDER_CONSTANT,value=0)

            disturbance = np.random.randint(-int(0.15 * cropRadius), int(0.15 * cropRadius), 4)
            xmin_crop = discX + borderWidth - cropRadius + disturbance[0]
            ymin_crop = discY + borderWidth - cropRadius + disturbance[1]
            xmax_crop = discX + borderWidth + cropRadius + disturbance[2]
            ymax_crop = discY + borderWidth + cropRadius + disturbance[3]

            DiscCrop = ImgPad[ymin_crop:ymax_crop, xmin_crop:xmax_crop, :]

        else:

            DiscCrop = Img

        DiscCrop = transforms.ToPILImage()(DiscCrop)
        if self.transform is not None:
            DiscCrop = self.transform(DiscCrop)

        label = self.DF.loc[index, 'label']

        data_path = self.data_path
        maskName = self.DF.loc[index, 'maskName']
        fullPathName = os.path.join(data_path, maskName)
        fullPathName = fullPathName.replace('\\', '/')

        Img = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)

        discFlag = self.DF.loc[index, 'discFlag']
        if discFlag == 1:

            ImgPad = cv2.copyMakeBorder(Img, borderWidth, borderWidth, borderWidth, borderWidth, cv2.BORDER_CONSTANT,
                                        value=0)
            Seg_DiscCrop = ImgPad[ymin_crop:ymax_crop, xmin_crop:xmax_crop]

        else:

            Seg_DiscCrop = Img

        Seg_DiscCrop = transforms.ToPILImage()(Seg_DiscCrop)
        if self.transform_seg is not None:
            Seg_DiscCrop = self.transform_seg(Seg_DiscCrop)

        return DiscCrop, Seg_DiscCrop, label, imgName


    def __len__(self):
        return len(self.DF)

class Dataset_FullImage_pool(Dataset):
    def __init__(self, data_path, DF,pool, transform = None, transform_seg = None):
        if DF[0] == 'train' or DF[0] == 'val' or DF[0] == 'imp' or DF[0] == 'ORIGAtrain' or DF[0] == 'ORIGAimp' or DF[0] == 'ORIGAtest':
            self.DF = pd.read_csv(data_path + '/' + 'Glaucoma' + '_' + DF[0] + '.csv', encoding='gbk',index_col = 0)
        elif DF[0] == 'LAG_train' or DF[0] == 'LAG_test':
            self.DF = pd.read_csv(data_path + '/' + DF[0] + '.csv',encoding='gbk',index_col = 0)
        elif DF[0] == 'REFUGETrain' or DF[0] == 'REFUGEVal' or DF[0] == 'REFUGETest':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        else:
            DF_all = pd.read_csv(data_path + '/' + 'GlaucomaLabels_6Center_191206_cleaned.csv', encoding='gbk', index_col = 0)
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = DF_all.loc[DF_all['center'] == split]
                DF_this = DF_this.reset_index(drop=True)
                self.DF = pd.concat([self.DF, DF_this])
        DF_all = self.DF
        self.DF = pd.DataFrame(
            columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                     'discFlag'])
        for (name,dif) in pool:
            DF_this = DF_all.loc[DF_all['imgName'] == name]
            DF_this = DF_this.reset_index(drop=True)
            self.DF = pd.concat([self.DF, DF_this])
        self.DF.index = range(0, len(self.DF))
        self.data_path = data_path
        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, index):

        """Get the images"""
        imgName = self.DF.loc[index, 'imgName']
        data_path = self.data_path + '/' + 'Images'
        fullPathName = os.path.join(data_path, imgName)
        fullPathName = fullPathName.replace('\\', '/')

        # Img0 = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)
        # Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
        # Img = transforms.ToPILImage()(Img)

        Img = Image.open(fullPathName).convert('RGB')
        if self.transform is not None:
            Img = self.transform(Img)

        """Get the segmentation images"""
        data_path = self.data_path
        maskName = self.DF.loc[index, 'maskName']
        fullPathName = os.path.join(data_path, maskName)
        fullPathName = fullPathName.replace('\\', '/')

        # Img0 = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)
        # Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
        # Img = transforms.ToPILImage()(Img)

        Seg = Image.open(fullPathName)
        if self.transform_seg is not None:
            Seg = self.transform_seg(Seg)

        label = self.DF.loc[index, 'label']

        return Img, Seg, label, imgName


    def __len__(self):
        return len(self.DF)

class Dataset_FullImage_pool_reverse(Dataset):
    def __init__(self, data_path, DF,pool, transform = None, transform_seg = None):
        if DF[0] == 'train' or DF[0] == 'val' or DF[0] == 'imp' or DF[0] == 'ORIGAtrain' or DF[0] == 'ORIGAimp' or DF[0] == 'ORIGAtest':
            self.DF = pd.read_csv(data_path + '/' + 'Glaucoma' + '_' + DF[0] + '.csv', encoding='gbk',index_col = 0)
        elif DF[0] == 'LAG_train' or DF[0] == 'LAG_test':
            self.DF = pd.read_csv(data_path + '/' + DF[0] + '.csv',encoding='gbk',index_col = 0)
        elif DF[0] == 'REFUGETrain' or DF[0] == 'REFUGEVal' or DF[0] == 'REFUGETest':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        else:
            DF_all = pd.read_csv(data_path + '/' + 'GlaucomaLabels_6Center_191206_cleaned.csv', encoding='gbk', index_col = 0)
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = DF_all.loc[DF_all['center'] == split]
                DF_this = DF_this.reset_index(drop=True)
                self.DF = pd.concat([self.DF, DF_this])
        DF_this = self.DF
        self.DF = pd.DataFrame(
            columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                     'discFlag'])
        for (name,dif) in pool:
            DF_this = DF_this.loc[DF_this['imgName'] != name]
            DF_this = DF_this.reset_index(drop=True)
        self.DF = pd.concat([self.DF, DF_this])
        self.DF.index = range(0, len(self.DF))
        self.data_path = data_path
        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, index):

        """Get the images"""
        imgName = self.DF.loc[index, 'imgName']
        data_path = self.data_path + '/' + 'Images'
        fullPathName = os.path.join(data_path, imgName)
        fullPathName = fullPathName.replace('\\', '/')

        # Img0 = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)
        # Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
        # Img = transforms.ToPILImage()(Img)

        Img = Image.open(fullPathName).convert('RGB')
        if self.transform is not None:
            Img = self.transform(Img)

        """Get the segmentation images"""
        data_path = self.data_path
        maskName = self.DF.loc[index, 'maskName']
        fullPathName = os.path.join(data_path, maskName)
        fullPathName = fullPathName.replace('\\', '/')

        # Img0 = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)
        # Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
        # Img = transforms.ToPILImage()(Img)

        Seg = Image.open(fullPathName)
        if self.transform_seg is not None:
            Seg = self.transform_seg(Seg)

        label = self.DF.loc[index, 'label']

        return Img, Seg, label, imgName


    def __len__(self):
        return len(self.DF)

class Dataset_FullImg(Dataset):
    def __init__(self, data_path, DF, transform = None, transform_seg = None):
        if DF[0] == 'train' or DF[0] == 'val' or DF[0] == 'imp' or DF[0] == 'ORIGAtrain' or DF[0] == 'ORIGAimp' or DF[0] == 'ORIGAtest':
            self.DF = pd.read_csv(data_path + '/' + 'Glaucoma' + '_' + DF[0] + '.csv', encoding='gbk')
        elif DF[0] == 'LAG_train' or DF[0] == 'LAG_test':
            self.DF = pd.read_csv(data_path + '/' + DF[0] + '.csv',encoding='gbk')
        elif DF[0] == 'REFUGETrain' or DF[0] == 'REFUGEVal' or DF[0] == 'REFUGETest':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        elif DF[0] == 'RIM-ONEv3_SIL' or DF[0] == 'RIM-ONEv3_SIR':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        elif DF[0] == 'DRIGHTI_train' or DF[0] == 'DRIGHTI_test':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        elif DF[0] == 'BinRushed' or DF[0] == 'Magrabia' or DF[0] == 'MESSIDOR':
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = pd.read_csv(data_path + '/' + 'Glaucoma_seg_' + split + '.csv', encoding='gbk')
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        else:
            DF_all = pd.read_csv(data_path + '/' + 'GlaucomaLabels_6Center_191206_cleaned.csv', encoding='gbk')
            self.DF = pd.DataFrame(
                columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height',
                         'discFlag'])
            for split in DF:
                DF_this = DF_all.loc[DF_all['center'] == split]
                DF_this = DF_this.reset_index(drop=True)
                DF_this = DF_this.drop('Unnamed: 0', 1)
                self.DF = pd.concat([self.DF, DF_this])
        self.DF.index = range(0, len(self.DF))
        self.data_path = data_path
        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, index):

        """Get the images"""
        imgName = self.DF.loc[index, 'imgName']
        data_path = self.data_path + '/' + 'Images'
        fullPathName = os.path.join(data_path, imgName)
        fullPathName = fullPathName.replace('\\', '/')

        # Img0 = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)
        # Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
        # Img = transforms.ToPILImage()(Img)

        Img = Image.open(fullPathName).convert('RGB')
        if self.transform is not None:
            Img = self.transform(Img)

        """Get the segmentation images"""
        data_path = self.data_path
        maskName = self.DF.loc[index, 'maskName']
        fullPathName = os.path.join(data_path, maskName)
        fullPathName = fullPathName.replace('\\', '/')

        # Img0 = cv2.imdecode(np.fromfile(fullPathName, dtype=np.uint8), -1)
        # Img = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
        # Img = transforms.ToPILImage()(Img)

        Seg = Image.open(fullPathName).convert('L')
        if self.transform_seg is not None:
            Seg = self.transform_seg(Seg)

        label = self.DF.loc[index, 'label']

        return Img, Seg, label, imgName


    def __len__(self):
        return len(self.DF)