import os
import numpy as np
from skimage import io
from PIL import Image


path = 'E:\GlaucomaClassification'
name_path = []
name_path_seg = []
Glaucoma_path = []
NonGlaucoma_path = []
Glaucoma_path_seg = []
NonGlaucoma_path_seg = []
Glaucoma_list = []
NonGlaucoma_list = []
Glaucoma_list_seg = []
NonGlaucoma_list_seg = []
Glaucoma_len = []
NonGlaucoma_len = []
all_list = []
length_Glau = 0
names = ['Tongren_DrCheng']
for idx, dataset_name in enumerate(names):
    name_path.append(os.path.join(os.path.join(path, 'Images'), dataset_name))
    name_path_seg.append(os.path.join(os.path.join(path, 'CupDiscMasks'), dataset_name))
    Glaucoma_path.append(os.path.join(name_path[idx], 'Glaucoma'))
    NonGlaucoma_path.append(os.path.join(name_path[idx], 'NonGlaucoma'))
    Glaucoma_path_seg.append(os.path.join(name_path_seg[idx], 'Glaucoma'))
    NonGlaucoma_path_seg.append(os.path.join(name_path_seg[idx], 'NonGlaucoma'))
    Glaucoma_list_temp = os.listdir(Glaucoma_path[idx])
    for i, image_name in enumerate(Glaucoma_list_temp):
        Glaucoma_list.append(os.path.join(Glaucoma_path[idx], image_name))
        Glaucoma_list_seg.append(os.path.join(Glaucoma_path_seg[idx], '.'.join((image_name.split('.',1)[0],'png'))))
    NonGlaucoma_list_temp = os.listdir(NonGlaucoma_path[idx])
    for i, image_name in enumerate(NonGlaucoma_list_temp):
        NonGlaucoma_list.append(os.path.join(NonGlaucoma_path[idx], image_name))
        NonGlaucoma_list_seg.append(os.path.join(NonGlaucoma_path_seg[idx], '.'.join((image_name.split('.',1)[0],'png'))))

all_list = Glaucoma_list
all_list.extend(NonGlaucoma_list)
all_list_seg = Glaucoma_list_seg
all_list_seg.extend(NonGlaucoma_list_seg)

for i, name in enumerate(all_list):
    image = io.imread(name)
    image_seg = io.imread(all_list_seg[i])
    if image.shape[0] == image_seg.shape[0] and image.shape[1] == image_seg.shape[1]:
        continue
    else:
        print('natural name: %s' % name)
        print('seg name: %s' % all_list_seg[i])
