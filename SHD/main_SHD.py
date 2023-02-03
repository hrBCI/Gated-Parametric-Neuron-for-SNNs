import os
import time
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("..")
from core import tools
from core import neurons
from core import surrogate
from core import losses
import model


def Parser():

    parser = argparse.ArgumentParser(description='adopted from spikingjelly')

    parser.add_argument('-net', default=model.S_channels
                        ,help='snn net')
    parser.add_argument('-neuron_func', default=neurons.GPN
                        ,help='snn neuron')
    parser.add_argument('-loss_func', default=losses.CE_mean
                        ,help='loss function')     
    
    parser.add_argument('-aug', default=True, type=bool, help='data augmentation')
    parser.add_argument('-T', default=20, type=int, help='simulating time-steps')
    parser.add_argument('-batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-epochs', default=150, type=int, metavar='N',help='number of total epochs to run')
    parser.add_argument('-lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('-neuron_kwargs', default={"surrogate_function": surrogate.ATan()},  help='snn neuron_kwrgs')
        
    parser.add_argument('-data_path', default='/mnt/data1/hrwang1/DATASETS/dynamics/', type=str, help='root path of dataset')
    parser.add_argument('-data_name', default='SHD', type=str, help='dataset name')
    parser.add_argument('-out_dir', type=str, default='/mnt/data1/hrwang1/SNN/BP/2_outputs/', help='root dir for saving logs and checkpoint')

    args = parser.parse_args()

    return args


def main_val(args):

    device=torch.device('cuda')
    net=args.net(T=args.T,single_step_neuron=args.neuron_func,**args.neuron_kwargs).to(device) 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_func = args.loss_func
    
    if not Path(args.data_path).is_dir():
        args.data_path = '/mnt/ssd2/hrwang1/DATASETS/dynamics/'
        args.out_dir = '/mnt/ssd2/hrwang1/SNN/BP/2_outputs/'

    train_dataset,test_dataset = tools.F_audio_datasets(args.data_name,args.data_path,args.T,args.aug)
    train_split,val_split = tools.F_split_dataset(train_ratio=0.75,origin_dataset=train_dataset,num_classes=20,random_split=True)
    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True,drop_last=False,num_workers=1,pin_memory=True)
    val_loader = DataLoader(val_split, batch_size=args.batch_size, shuffle=False,drop_last=False,num_workers=1,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=1,pin_memory=True)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    print('\nval mode')
    print('\ndataset:%s  augmentation:%s'%(args.data_name,str(args.aug)))
    print('\nnet:%s  neuron:%s  loss:%s'%(type(net).__name__,args.neuron_func.__name__,args.loss_func.__name__))
    print('\nLR:%s  T:%d  batch:%d  epochs:%d'%(str(args.lr),args.T,args.batch_size,args.epochs))

    out_dir = os.path.join(args.out_dir, str(int(time.time())))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'\nMake dir {out_dir}\n')
    

    start_epoch = 0
    writer = SummaryWriter(os.path.join(out_dir), purge_step=start_epoch)

    early_stopping = tools.F_EarlyStopping_val(patience=10, path=out_dir+'/checkpoint.pt')

    tools.F_reset_all(net)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        train_batch = 0


        for frame, label in train_loader:
      
            frame = frame.float().to(device)
            label = label.to(device)
            
            out_rec_TNO = net(frame)
    
            loss = loss_func(out_rec_TNO,label,20)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            out_rec_NO = torch.mean(out_rec_TNO, dim=0)

            train_batch += 1
            train_samples += label.numel()
            _, idx = out_rec_NO.max(1)
            train_acc += np.sum((label == idx).cpu().numpy())
            train_loss += loss.cpu().item()
            
            tools.F_reset_all(net)
            
        train_loss /= train_batch
        train_acc /= train_samples
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        lr_scheduler.step()
        



        net.eval()
        val_loss = 0
        val_acc = 0
        val_samples = 0
        val_batch = 0

        with torch.no_grad():
            for frame, label in val_loader:
 
                frame = frame.float().to(device)
                label = label.to(device)
        
                out_rec_TNO = net(frame)
    
                loss = loss_func(out_rec_TNO,label,20)

                out_rec_NO = torch.mean(out_rec_TNO, dim=0)

                val_batch += 1
                val_samples += label.numel()
                _, idx = out_rec_NO.max(1)
                val_acc += np.sum((label == idx).cpu().numpy())
                val_loss += loss.cpu().item()

                tools.F_reset_all(net)
                
        val_loss /= val_batch
        val_acc /= val_samples
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)


        print("epoch:%d  train_loss:%.4f  train_acc:%.3f  val_loss:%.4f  val_acc:%.3f  time:%.1fs"%(epoch,train_loss,100*train_acc,val_loss,100*val_acc,time.time()-start_time))

        if args.epochs - epoch < 50:
            early_stopping(val_acc, net)
            if early_stopping.early_stop:
                print("\n\nEarly stopping at %d epoch"%(epoch))
                break

    
    net.load_state_dict(torch.load(out_dir+'/checkpoint.pt'))
    net.eval()
    test_acc = 0
    test_samples = 0

    with torch.no_grad():
        for frame, label in test_loader:
         
            frame = frame.float().to(device)
            label = label.to(device)
    
            out_rec_TNO = net(frame)
            out_rec_NO = torch.mean(out_rec_TNO, dim=0)

            test_samples += label.numel()
            _, idx = out_rec_NO.max(1)
            test_acc += np.sum((label == idx).cpu().numpy())

            tools.F_reset_all(net)
            

    test_acc /= test_samples           
    
    print('\n-----------------')
    print('test acc%.3f'%(100*test_acc))
    print('-----------------\n')

    return test_acc*100



def main_no_val(args):

    device=torch.device('cuda')
    net=args.net(T=args.T,single_step_neuron=args.neuron_func,**args.neuron_kwargs).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_func = args.loss_func
    
    if not Path(args.data_path).is_dir():
        args.data_path = '/mnt/ssd2/hrwang1/DATASETS/dynamics/'
        args.out_dir = '/mnt/ssd2/hrwang1/SNN/BP/2_outputs/'

    train_dataset,test_dataset = tools.F_audio_datasets(args.data_name,args.data_path,args.T,args.aug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=False,num_workers=1,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=1,pin_memory=True)

    print('\ntest mode')
    print('\ndataset:%s  augmentation:%s'%(args.data_name,str(args.aug)))
    print('\nnet:%s  neuron:%s  loss:%s'%(type(net).__name__,args.neuron_func.__name__,args.loss_func.__name__))
    print('\nLR:%s  T:%d  batch:%d  epochs:%d'%(str(args.lr),args.T,args.batch_size,args.epochs))

    out_dir = os.path.join(args.out_dir, str(int(time.time())))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'\nMake dir {out_dir}\n')
    

    start_epoch = 0
    train_loss_list=[]
    train_acc_list=[]  
    test_acc_list=[]

    writer = SummaryWriter(os.path.join(out_dir), purge_step=start_epoch)

    tools.F_reset_all(net)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        train_epoch = 0

        for frame, label in train_loader:
   
            frame = frame.float().to(device)
            label = label.to(device)
            
            out_rec_TNO = net(frame)

            loss = loss_func(out_rec_TNO,label,20)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            out_rec_NO = torch.mean(out_rec_TNO, dim=0)
                 
            train_epoch += 1
            train_samples += label.numel()
            _, idx = out_rec_NO.max(1)
            train_acc += np.sum((label == idx).cpu().numpy())
            train_loss += loss.cpu().item()
            
            tools.F_reset_all(net)

        lr_scheduler.step()         
        
        train_loss /= train_epoch
        train_acc /= train_samples
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
    

        net.eval()
        test_acc = 0
        test_samples = 0
        
        with torch.no_grad():
            for frame, label in test_loader:
    
                frame = frame.float().to(device)
                label = label.to(device)
        
                out_rec_TNO = net(frame)
                out_rec_NO = torch.mean(out_rec_TNO, dim=0)

                test_samples += label.numel()
                _, idx = out_rec_NO.max(1)
                test_acc += np.sum((label == idx).cpu().numpy())

                tools.F_reset_all(net)
                
        test_acc /= test_samples
        writer.add_scalar('test_acc', test_acc, epoch)
        test_acc_list.append(test_acc)

        print("epoch:%d  loss:%.4f  train_acc:%.3f  test_acc:%.3f  time:%.1fs"%(epoch,train_loss,100*train_acc,100*test_acc,time.time()-start_time))
    
    np.savez(out_dir+'/output.npz', arr_train_loss=np.array(train_loss_list), arr_train_acc=np.array(train_acc_list), arr_test_acc=np.array(test_acc_list))
    
    return 100*test_acc


    
if __name__ == '__main__':

    args = Parser()

    acc = []

    tools.F_init_seed(2022)
    acc.append(main_val(args))
    tools.F_init_seed(2023)
    acc.append(main_val(args))
    tools.F_init_seed(2024)
    acc.append(main_val(args))

    mean = np.mean(np.array(acc))
    std = np.std(np.array(acc))

    print('\n\n\n=========================')
    print('val early stop')
    print('mean:%.2f  std:%.2f'%(mean,std))
    print('=========================\n\n\n')

    tools.F_init_seed(2024)
    acc_no_val=main_no_val(args)
    
    print('\n\n\n=========================')
    print('test acc:%.2f'%(acc_no_val))

    
    
