# -*- coding: utf-8 -*-

import sys
import os
import torch
import numpy as np
import random
import csv
import os
import random
from random import shuffle
from random import choice
from os import listdir
from os.path import join
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import matplotlib.pyplot as plt

import skimage.io
import skimage.transform
import skimage.color
import skimage
from PIL import Image

image_size = 128

def is_tiff(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.tiff'])

class coma_Dataset_n(Dataset):
    def __init__(self,image_dir,labeled=True,train_model=True,transform=None,random_shuffle=False):
        self.images_dir = image_dir
        self.images_coma = image_dir + '/coma1'
        self.images_coma1 = image_dir + '/coma1_d'

        self.image_files = np.array([file_name for file_name in sorted(listdir(self.images_coma)) if file_name.endswith('.tiff')])
        if random_shuffle:
            shuffle(self.image_files)
        
        self.nums = len(self.image_files)
        self.labeled = labeled
        self.transform = transform
        
    def __len__(self):
        return self.nums
    
    
    def __getitem__(self, index):
        image,label = self.read_image(index)
        sample = {'image':image,'coma1':label[0],'coma2':label[1],'coma3':label[2],'coma4':label[3],'coma5':label[4]}
        if self.transform:
            sample = self.transform(sample)
        return sample
        
    def read_image(self,index):
        file_name = self.image_files[index]
        k1 = file_name.find('_coma1_')
        k2 = file_name.find('_coma2_')
        k3 = file_name.find('_ast1_')
        k4 = file_name.find('_ast2_')
        k5 = file_name.find('_sph_')
        k6 = file_name.find('.tif')
        coma1 = float(file_name[k1+7:k2])
        coma2 = float(file_name[k2+7:k3])
        coma3 = float(file_name[k3+6:k4])
        coma4 = float(file_name[k4+6:k5])
        #coma5 = float(file_name[k4+6:k5])
        coma5 = float(file_name[k5+5:k6])
        
        
        
        label = [self.read_label(coma1),self.read_label(coma2),self.read_label(coma3),self.read_label(coma4),self.read_label(coma5)]
        
        img1= skimage.io.imread(self.images_coma + '/' + file_name)
        img2= skimage.io.imread(self.images_coma1 + '/' + file_name)
        img1[img1 > 1000] = 1000
        img1[img1 < 100] = 100
        img2[img2 > 1000] = 1000
        img2[img2 < 100] = 100
       

        x = random.randint(200,img1.shape[1]-200 - image_size)
        y = random.randint(200,img1.shape[0]-200- image_size)
        img1 = img1[y:y+image_size,x:x+image_size]
        img2 = img2[y:y+image_size,x:x+image_size]
        #img = np.expand_dims(img1,axis = 2)
        img = np.dstack((img1,img2))
        
        
        if np.random.rand() > 0.95:
            scale = np.random.uniform(0.98,1.02)
            img = img*scale
        return img,label
    
    def read_label(self,x):
        if x <=-0.175 and x > -0.225:
            return 0
        elif x <=-0.125 and x > -0.175:
            return 1
        elif x <=-0.075 and x > -0.125:
            return 2
        elif x <=-0.025 and x > -0.075:
            return 3
        elif x <=0.025 and x > -0.025:
            return 4
        elif x <=0.075 and x > 0.025:
            return 5
        elif x <=0.125 and x > 0.075:
            return 6
        elif x <=0.175 and x > 0.125:
            return 7
        elif x <=0.225 and x > 0.175:
            return 8
        
class Normalizer(object):
    def __init__(self,mean=None,std=None):
        self.mean = np.array([[[100]]])
        self.std = np.array([[[900]]])
    def __call__(self,sample):
        img = sample['image']
        sample = {'image':((img)),'coma1':sample['coma1'],'coma2':sample['coma2'],'coma3':sample['coma3'],'coma4':sample['coma4'],'coma5':sample['coma5']}
        return sample
    
class Augmenter(object):
    def __call__(self,sample):
        image = sample['image'].astype(np.float64)
        coma1 = np.asarray(sample['coma1'])
        coma2 = np.asarray(sample['coma2'])
        coma3 = np.asarray(sample['coma3'])
        coma4 = np.asarray(sample['coma4'])
        coma5 = np.asarray(sample['coma5'])
        
        return {'image':torch.from_numpy(image.copy()).type(torch.FloatTensor).permute(2,0,1),'coma1':torch.from_numpy(coma1.copy()).type(torch.FloatTensor),'coma2':torch.from_numpy(coma2.copy()).type(torch.FloatTensor),'coma3':torch.from_numpy(coma3.copy()).type(torch.FloatTensor),'coma4':torch.from_numpy(coma4.copy()).type(torch.FloatTensor),'coma5':torch.from_numpy(coma5.copy()).type(torch.FloatTensor)}
    
    
    
    
if __name__ == '__main__':
    print('../data4/Dm_coma/train')
    dir_name = './data4/Dm_coma/train'
    a = np.array([file_name for file_name in sorted(listdir(dir_name + '/coma1')) if file_name.endswith('.tiff')])
    train_dataset = coma_Dataset_n(dir_name,True,True,transforms.Compose([Normalizer(),Augmenter()]),True)
    
    

        
        
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    






























