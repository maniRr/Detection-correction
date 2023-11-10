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
from os import listdir
from os.path import join
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class Res_block(nn.Module):
    def __init__(self,in_channels = 1,mid_channels = 32,out_channels = 64):
        super(Res_block,self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels,mid_channels,kernel_size = 3, padding = 1,padding_mode = 'replicate'), nn.BatchNorm2d(mid_channels),nn.ReLU())
        self.conv_block2 = nn.Sequential(nn.Conv2d(mid_channels,mid_channels,kernel_size = 3, padding = 1,padding_mode = 'replicate'), nn.BatchNorm2d(mid_channels),nn.ReLU())
        self.conv_block3 = nn.Sequential(nn.Conv2d(mid_channels,out_channels,kernel_size = 3, padding = 1,padding_mode = 'replicate'), nn.BatchNorm2d(out_channels))
        self.short_cut = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size = 1, padding = 0),nn.BatchNorm2d(out_channels))
        self.maxPool = nn.MaxPool2d(2)
        
    def forward(self,x):
        out2 = self.conv_block1(x)
        out2 = self.conv_block2(out2)
        out2 = self.conv_block3(out2)
        out1 = self.short_cut(x)
        #print(out1.shape)
        out1 += out2
        out = F.relu(out1)
        out = self.maxPool(out)
        return out
    
class Res_block_nopool(nn.Module):
    def __init__(self,in_channels = 1,mid_channels = 32,out_channels = 64):
        super(Res_block_nopool,self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels,mid_channels,kernel_size = 5, padding = 0,padding_mode = 'replicate'), nn.BatchNorm2d(mid_channels),nn.ReLU())
        self.conv_block2 = nn.Sequential(nn.Conv2d(mid_channels,mid_channels,kernel_size = 5, padding = 0,padding_mode = 'replicate'), nn.BatchNorm2d(mid_channels),nn.ReLU())
        self.conv_block3 = nn.Sequential(nn.Conv2d(mid_channels,out_channels,kernel_size = 5, padding = 0,padding_mode = 'replicate'), nn.BatchNorm2d(out_channels))
        self.short_cut = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size = 1, padding = 0),nn.BatchNorm2d(out_channels))
        self.maxPool = nn.MaxPool2d(2)
        
    def forward(self,x):
        out2 = self.conv_block1(x)
        out2 = self.conv_block2(out2)
        out2 = self.conv_block3(out2)

        #print(out1.shape)
        out = F.relu(out2)
        return out
    
    
class decoder(nn.Module):
        def __init__(self,in_channels,mid_channels,out_channels):
            super(decoder,self).__init__()
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear',align_corners = True)
            
            self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels,mid_channels,kernel_size = 3, padding = 1,padding_mode = 'replicate'), nn.BatchNorm2d(mid_channels),nn.ReLU())
            self.conv_block2 = nn.Sequential(nn.Conv2d(mid_channels,mid_channels,kernel_size = 3, padding = 1,padding_mode = 'replicate'), nn.BatchNorm2d(mid_channels),nn.ReLU())
            self.conv_block3 = nn.Sequential(nn.Conv2d(mid_channels,out_channels,kernel_size = 3, padding = 1,padding_mode = 'replicate'), nn.BatchNorm2d(out_channels))
        
        def forward(self,x1,x2):
            #x2 = self.up(x2)
            x = torch.cat([x1,x2],dim=1)
            x = self.up(x)
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = F.relu(x)
            return x
        
    
class Network_generator(nn.Module):
    def __init__(self):
        super(Network_generator,self).__init__()
        self.downSample1 = Res_block(1,32,64)
        self.downSample2 = Res_block(64,96,128)
        self.downSample3 = Res_block(128,192,256)
        self.downSample4 = Res_block(256,384,512)
        
        self.bot_block = nn.Sequential(nn.Conv2d(512,512,kernel_size = 3, padding = 1,padding_mode = 'replicate'), nn.BatchNorm2d(512),nn.ReLU())
        
        self.upSample4 = decoder(1024,640,256)
        self.upSample3 = decoder(512,320,128)
        self.upSample2 = decoder(256,160,64)
        self.upSample1 = decoder(128,80,32)
        
        self.conv_final = nn.Sequential(nn.Conv2d(32,1,kernel_size = 1, padding = 0,padding_mode = 'replicate'), nn.BatchNorm2d(1),nn.ReLU())
        
        
        
        self.conv1 = nn.Conv2d(32,48,5,2,padding = 0)
        self.conv2 = nn.Conv2d(48,56,5,2,padding = 0)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(15*15*4,32)
        self.fc2 = nn.Linear(32,2)
        self.fc_coma1 = nn.Linear(56*15*15,256)
        self.fc_coma11 = nn.Linear(256,9)
        self.fc_coma2 = nn.Linear(56*15*15,256)
        self.fc_coma22 = nn.Linear(256,9)
        self.dropout = nn.Dropout()
    def forward(self,x):
        
        x1 = self.downSample1(x) # 64 128 128
        x2 = self.downSample2(x1) # 128 64 64
        x3 = self.downSample3(x2) # 256 32 32
        x4 = self.downSample4(x3) # 512 16 16
        x5 = self.bot_block(x4) # 512 16 16
        
        x6 = self.upSample4(x4,x5) # 256 32 32
        x7 = self.upSample3(x3,x6) # 128 64 64
        x8 = self.upSample2(x2,x7) # 64 128 128
        x9 = self.upSample1(x1,x8) # 32 256 256
        
        x10 = F.relu(self.maxpool(self.conv1(x9)))
        x11 = F.relu(self.maxpool(self.conv2(x10)))
        
        x12 = x11.view(-1,56*15*15)
        #x10 = self.conv_final(x9)
        
        #x = x10.view(-1,1*256*256)
        #o = torch.sigmoid(x10)
        x_coma1 = F.relu(self.fc_coma1(x12))
        x_coma1 = self.dropout(x_coma1)
        x_coma1 = self.fc_coma11(x_coma1)
        
        x_coma2 = F.relu(self.fc_coma2(x12))
        x_coma2 = self.dropout(x_coma2)
        x_coma2 = self.fc_coma22(x_coma2)
        return {'coma0':x_coma1, 'coma1':x_coma2}

class Network_resnet(nn.Module):
    def __init__(self):
        super(Network_resnet,self).__init__()
        self.downSample1 = Res_block(2,32,64)
        self.downSample2 = Res_block(64,96,128)
        self.downSample3 = Res_block(128,256,256)
        self.downSample4 = Res_block(256,256,384)
        self.downSample5 = Res_block(512,512,512)
        self.maxPool = nn.MaxPool2d(2)
        
        self.fc_coma1 = nn.Linear(16384,256)
        self.fc_coma11 = nn.Linear(256,9)
        self.fc_coma2 = nn.Linear(16384,256)
        self.fc_coma22 = nn.Linear(256,9)
        self.fc_coma3 = nn.Linear(16384,256)
        self.fc_coma33 = nn.Linear(256,9)
        self.fc_coma4 = nn.Linear(16384,256)
        self.fc_coma44 = nn.Linear(256,9)
        self.fc_coma5 = nn.Linear(16384,256)
        self.fc_coma55 = nn.Linear(256,9)
        self.dropout = nn.Dropout()
        
    def forward(self,x):
        
        x1 = self.downSample1(x) # 64 128 128
        x2 = self.downSample2(x1) # 128 64 64
        x3 = self.downSample3(x2) # 256 32 
        #x4 = self.downSample4(x3) # 512 16 16
        #x5 = self.downSample5(x4) # 512 16 16
        x5 = self.maxPool(x3)

        x6 = x5.view(-1,16384)
        x_coma1 = F.relu(self.fc_coma1(x6))
        x_coma1 = self.dropout(x_coma1)
        x_coma1 = self.fc_coma11(x_coma1)
        
        x_coma2 = F.relu(self.fc_coma2(x6))
        x_coma2 = self.dropout(x_coma2)
        x_coma2 = self.fc_coma22(x_coma2)
        
        x_coma3 = F.relu(self.fc_coma3(x6))
        x_coma3 = self.dropout(x_coma3)
        x_coma3 = self.fc_coma33(x_coma3)
        
        x_coma4 = F.relu(self.fc_coma4(x6))
        x_coma4 = self.dropout(x_coma4)
        x_coma4 = self.fc_coma44(x_coma4)
        
        x_coma5 = F.relu(self.fc_coma5(x6))
        x_coma5 = self.dropout(x_coma5)
        x_coma5 = self.fc_coma55(x_coma5)
        
        
        return {'coma1':x_coma1,'coma2':x_coma2,'coma3':x_coma3,'coma4':x_coma4,'coma5':x_coma5}

class Network_resnet_regression(nn.Module):
    def __init__(self):
        super(Network_resnet_regression,self).__init__()
        self.downSample1 = Res_block_nopool(1,32,64)
        self.downSample2 = Res_block_nopool(64,96,128)
        self.downSample3 = Res_block_nopool(128,192,256)
        self.downSample4 = Res_block_nopool(256,256,384)
        self.downSample5 = Res_block_nopool(512,512,512)
        self.maxPool = nn.MaxPool2d(2)
        
        self.fc_coma1 = nn.Linear(384*40*40,256)
        self.fc_coma11 = nn.Linear(256,1)
        self.fc_coma2 = nn.Linear(384*40*40,256)
        self.fc_coma22 = nn.Linear(256,1)
        self.dropout = nn.Dropout()
        
    def forward(self,x):
        
        x1 = self.downSample1(x) # 64 128 128
        x2 = self.downSample2(x1) # 128 64 64
        x3 = self.downSample3(x2) # 256 32 
        x4 = self.downSample4(x3) # 512 16 16
        #x5 = self.downSample5(x4) # 512 16 16
        x5 = self.maxPool(x4)

        x6 = x5.view(-1,384*40*40)
        x_coma1 = F.relu(self.fc_coma1(x6))
        x_coma1 = self.dropout(x_coma1)
        x_coma1 = self.fc_coma11(x_coma1)
        
        x_coma2 = F.relu(self.fc_coma2(x6))
        x_coma2 = self.dropout(x_coma2)
        x_coma2 = self.fc_coma22(x_coma2)
        return {'coma0':x_coma1, 'coma1':x_coma2}
    


class Network_discriminator(nn.Module):
    def __init__(self):
        self.a = 0
    def forward(x):
        return 0
    
    
    
    
    
if __name__ == '__main__':
    x = torch.randn(4,1,128,128)
    model = Network_resnet_regression()
    out = model(x)
    print(out['coma0'].shape)
    
    
    
    
    
    
    
    
