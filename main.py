# -*- coding: utf-8 -*-
import os
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms,utils,models
import torchvision.utils as vutils
from torchvision import utils,models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from pytorchcv.model_provider import get_model as ptcv_get_model
from scipy.interpolate import CubicSpline
import numpy as np
import timeit
import time
import random
from random import shuffle
import logging
import argparse
from datetime import datetime
import skimage.io
import skimage.transform
import skimage.color
import skimage

from dataset_load import coma_Dataset_n, Normalizer, Augmenter
from model import Network_generator, Network_resnet
logging.getLogger().setLevel(logging.INFO)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Configuration
parser = argparse.ArgumentParser(description='Deformable mirror coma estimation.')
parser.add_argument('--dataset', default = 'DM_coma')
parser.add_argument('--data_path', default = '../data')
parser.add_argument('---workers', default=8,type=int)
parser.add_argument('--epochs', default=2000,type=int)
parser.add_argument('--batch_size', default = 8, type=int)
parser.add_argument('--model_name', default='Mani_coma_net',type=str)


def main():
    args = parser.parse_args()
    print("PYTORCH VERSION", torch.__version__)
    args.data_dir = args.data_path
    args.start_epoch = 0
    args.model_name = '{}_{}'.format(args.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(args.model_name)
    makedirs('./results')
    args.res_dir = os.path.join('./results', args.model_name)
    makedirs(args.res_dir)
    logging.info('New train: loading data')

    train_dir = './train'
    test_dir = './test'
    
    train_dataset = coma_Dataset_n(train_dir,True,True,transforms.Compose([Normalizer(),Augmenter()]),True)
    test_dataset = coma_Dataset_n(test_dir,True,True,transforms.Compose([Normalizer(),Augmenter()]),True)
    #return len(train_dataset)
    
    train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size,shuffle=True,num_workers =4,pin_memory=True)
    test_dataloader = DataLoader(test_dataset,batch_size = args.batch_size,shuffle=True,num_workers =4,pin_memory=True)
    
    model = Network_resnet()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model,device_ids = [1,0])
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.00001,betas=(0.5,0.99))
    weights = [9.8, 68.0, 5.3, 3.5, 10.8, 1.1, 1.4,1,10]
    class_weights = torch.FloatTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    epochs = args.epochs
    train_losses = []
    test_losses = []
    
    for epoch in range(1,epochs + 1):
        train_loss = 0.0
        test_loss = 0.0
        run_loss = 0.0
        model.train()
        optimizer.zero_grad()
        # train
        for index,data in enumerate(train_dataloader):
            # if index > 1:
            #     return 0
            if index % 50 == 0:
                print(index)
            optimizer.zero_grad()
            output = model(data['image'].to(device))
            #return data['image']
            coma1 = data['coma1'].type(dtype=torch.long).to(device)
            coma2 = data['coma2'].type(dtype=torch.long).to(device)
            coma3 = data['coma3'].type(dtype=torch.long).to(device)
            coma4 = data['coma4'].type(dtype=torch.long).to(device)
            coma5 = data['coma5'].type(dtype=torch.long).to(device)
           
            #print(coma0.shape)
            #print(output['coma0'].shape)
            loss_coma1 = criterion(output['coma1'], coma1)
            loss_coma2 = criterion(output['coma2'], coma2)
            loss_coma3 = criterion(output['coma3'], coma3)
            loss_coma4 = criterion(output['coma4'], coma4)
            loss_coma5 = criterion(output['coma5'], coma5)
            
            loss = loss_coma1+ loss_coma2+ loss_coma3+ loss_coma4+ loss_coma5
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            run_loss += loss.item()
            if index % 50 == 0:
                logging.info('\n Epoch: {}/{} Loss: {} \n'.format(epoch,epochs,run_loss/100))
                run_loss = 0
        
        # test
        with torch.no_grad():
            model.eval()
            for index,data in enumerate(test_dataloader):
                output = model(data['image'].to(device))
                coma1 = data['coma1'].type(dtype=torch.long).to(device)
                coma2 = data['coma2'].type(dtype=torch.long).to(device)
                coma3 = data['coma3'].type(dtype=torch.long).to(device)
                coma4 = data['coma4'].type(dtype=torch.long).to(device)
                coma5 = data['coma5'].type(dtype=torch.long).to(device)
                
                loss_coma1 = criterion(output['coma1'], coma1)
                loss_coma2 = criterion(output['coma2'], coma2)
                loss_coma3 = criterion(output['coma3'], coma3)
                loss_coma4 = criterion(output['coma4'], coma4)
                loss_coma5 = criterion(output['coma5'], coma5)
               
                loss = loss_coma1+ loss_coma2+ loss_coma3+ loss_coma4+ loss_coma5
                test_loss += loss.item()
                
        if epoch % 50 == 0:
            print('Epoch: {} is saved.'.format(epoch))
            torch.save(model.state_dict(),args.res_dir + '/' + 'net_{}.pt'.format(epoch))
            
        train_losses.append(train_loss/len(train_dataloader))
        test_losses.append(test_loss/len(test_dataloader))
        np.savetxt(args.res_dir + '/' + 'train_losses.txt',train_losses,delimiter=',')
        np.savetxt(args.res_dir + '/' + 'test_losses.txt',test_losses,delimiter=',')
        
    return 0
    
    
    
    



def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
if __name__ == '__main__':
    a = main()

