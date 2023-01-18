
import torch
import torch.nn as nn

import numpy as np
import os
import random
import math

import sys
sys.path.append("..")
import audio_dataset


class F_EarlyStopping_val:

    def __init__(self, patience=10, path=None):
      
        self.patience = patience
        self.counter = 0
        self.val_min_acc = 0.
        self.early_stop = False
        self.path = path

    def __call__(self, val_acc, model):

 
        if  val_acc < self.val_min_acc:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.val_min_acc = val_acc
            self.save_checkpoint(model)
            self.counter = 0
       
    def save_checkpoint(self, model):
    
        torch.save(model.state_dict(), self.path)


def F_split_dataset(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool):

    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    val_idx = []

    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])
    
    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        val_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, val_idx)



def F_AUDIO_aug(data):

    off = random.randint(-15, 15)
    data = np.roll(data, shift=off, axis=2)

    return data


def F_audio_datasets(data_name,data_path,T,aug):

    if data_name == 'SHD':
        
        train_set_pth = os.path.join(data_path+data_name+'/cache/', f'train_set_{T}_aug_{aug}.pt')
        test_set_pth = os.path.join(data_path+data_name+'/cache/', f'test_set_{T}_aug_{aug}.pt')

        if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
            trainset = torch.load(train_set_pth)
            testset = torch.load(test_set_pth)

        else:
            if aug == False:
                trainset = audio_dataset.AUDIO_datasets(file_name=data_path+data_name+'/origin/shd_train.h5',T=T)
            else:
                trainset = audio_dataset.AUDIO_datasets(file_name=data_path+data_name+'/origin/shd_train.h5',T=T,transform=F_AUDIO_aug)
            testset = audio_dataset.AUDIO_datasets(file_name=data_path+data_name+'/origin/shd_test.h5',T=T)

            if not os.path.exists(os.path.join(data_path+data_name+'/cache/')):
                os.makedirs(os.path.join(data_path+data_name+'/cache/'))
            torch.save(trainset, train_set_pth)
            torch.save(testset, test_set_pth)

    elif data_name == 'SSC':
        
        train_set_pth = os.path.join(data_path+data_name+'/cache/', f'train_set_{T}_aug_{aug}.pt')
        val_set_pth = os.path.join(data_path+data_name+'/cache/', f'val_set_{T}_aug_{aug}.pt')
        test_set_pth = os.path.join(data_path+data_name+'/cache/', f'test_set_{T}_aug_{aug}.pt')

        if os.path.exists(train_set_pth) and os.path.exists(test_set_pth) and os.path.exists(val_set_pth):
            trainset = torch.load(train_set_pth)
            valset = torch.load(val_set_pth)
            testset = torch.load(test_set_pth)

        else:
            if aug == False:
                trainset = audio_dataset.AUDIO_datasets(file_name=data_path+data_name+'/origin/ssc_train.h5',T=T)
                valset = audio_dataset.AUDIO_datasets(file_name=data_path+data_name+'/origin/ssc_valid.h5',T=T)
            else:
                trainset = audio_dataset.AUDIO_datasets(file_name=data_path+data_name+'/origin/ssc_train.h5',T=T,transform=F_AUDIO_aug)
                valset = audio_dataset.AUDIO_datasets(file_name=data_path+data_name+'/origin/ssc_valid.h5',T=T,transform=F_AUDIO_aug)
            testset = audio_dataset.AUDIO_datasets(file_name=data_path+data_name+'/origin/ssc_test.h5',T=T)

            if not os.path.exists(os.path.join(data_path+data_name+'/cache/')):
                os.makedirs(os.path.join(data_path+data_name+'/cache/'))
            torch.save(trainset, train_set_pth)
            torch.save(valset,val_set_pth)
            torch.save(testset, test_set_pth)


    if data_name == 'SSC':
        return trainset,valset,testset
    else:
        return trainset,testset


def F_init_seed(seed):

    print('\nseed:',seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def F_reset_all(net: nn.Module):
    
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()

