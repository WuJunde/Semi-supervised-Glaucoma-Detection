import os
import numpy as np
from skimage import io
from PIL import Image


path = '/extracephonline/medai_data1/jundewu/data/GlaucomaClassification'
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
        Glaucoma_list_seg.append(os.path.join(Glaucoma_path_seg[idx], image_name))
    NonGlaucoma_list_temp = os.listdir(NonGlaucoma_path[idx])
    for i, image_name in enumerate(NonGlaucoma_list_temp):
        NonGlaucoma_list.append(os.path.join(NonGlaucoma_path[idx], image_name))
        NonGlaucoma_list_seg.append(os.path.join(NonGlaucoma_path_seg[idx], image_name))

all_list = Glaucoma_list
all_list.extend(NonGlaucoma_list)
all_list_seg = Glaucoma_list_seg
all_list_seg.extend(NonGlaucoma_list_seg)
for i, image_path in enumerate(all_list):
    data = np.dstack([io.imread(image_path)[:, :] ])
mean = np.mean(data)
std = np.std(data)
print(mean)
print(std)