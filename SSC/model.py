import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

from core import methods


#snn
class S_base(nn.Module):
    def __init__(self, T, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        self.T = T

        self.fc1 = nn.Linear(700, 512)
        self.sn1 = single_step_neuron(**kwargs)
        self.dp1 = methods.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.sn2 = single_step_neuron(**kwargs)
        self.dp2 = methods.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.sn3 = single_step_neuron(**kwargs)
        self.dp3 = methods.Dropout(0.5)

        self.fc4 = nn.Linear(128, 35)
   

    def forward(self, x_input: torch.Tensor):
        
        out_rec=[]

        x_input = x_input.permute(1, 0, 2)


        for t in range(self.T):

            x = self.fc1(x_input[t])
            x = self.sn1(x)
            x = self.dp1(x)

            x = self.fc2(x)
            x = self.sn2(x)
            x = self.dp2(x)

            x = self.fc3(x)
            x = self.sn3(x)
            x = self.dp3(x)

            x = self.fc4(x)

            out_rec.append(x)

        out_rec_TBdims = torch.stack(out_rec, dim=0)
     
        return out_rec_TBdims

#snn input_channels in kwargs
class S_channels(nn.Module):
    def __init__(self, T, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        self.T = T

        self.fc1 = nn.Linear(700, 512)
        self.sn1 = single_step_neuron(input_channels=512)
        self.dp1 = methods.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.sn2 = single_step_neuron(input_channels=256)
        self.dp2 = methods.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.sn3 = single_step_neuron(input_channels=128)
        self.dp3 = methods.Dropout(0.5)

        self.fc4 = nn.Linear(128, 35)
   

    def forward(self, x_input: torch.Tensor):
        
        out_rec=[]

        x_input = x_input.permute(1, 0, 2)


        for t in range(self.T):

            x = self.fc1(x_input[t])
            x = self.sn1(x)
            x = self.dp1(x)

            x = self.fc2(x)
            x = self.sn2(x)
            x = self.dp2(x)

            x = self.fc3(x)
            x = self.sn3(x)
            x = self.dp3(x)

            x = self.fc4(x)

            out_rec.append(x)

        out_rec_TBdims = torch.stack(out_rec, dim=0)
     
        return out_rec_TBdims

#lstm
class P_ann_lstm(nn.Module):
    def __init__(self, T, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        self.sn = nn.LSTM(input_size=700,hidden_size=128,num_layers=3)
        self.fc = nn.Linear(in_features=128,out_features=35)
  

    def forward(self, x_input: torch.Tensor):
        
        x = x_input.permute(1,0,2)
        
        x,_ = self.sn(x)
       
        x = self.fc(x)

        return x

#gru
class P_ann_gru(nn.Module):
    def __init__(self, T, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        self.sn = nn.GRU(input_size=700,hidden_size=128,num_layers=3)
        self.fc = nn.Linear(in_features=128,out_features=35)
  

    def forward(self, x_input: torch.Tensor):
        
        x = x_input.permute(1,0,2)
        
        x,_ = self.sn(x)
       
        x = self.fc(x)

        return x

#rnn
class P_ann_rnn(nn.Module):
    def __init__(self, T, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        self.sn = nn.RNN(input_size=700,hidden_size=128,num_layers=3)
        self.fc = nn.Linear(in_features=128,out_features=35)
  

    def forward(self, x_input: torch.Tensor):
        
        x = x_input.permute(1,0,2)
        
        x,_ = self.sn(x)
       
        x = self.fc(x)

        return x

#cnn
class P_cnn(nn.Module):
    def __init__(self, T, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        self.T = T

        self.conv1 = nn.Conv1d(in_channels=700,out_channels=512,kernel_size=7,stride=1,padding=3)
        self.bn1 = nn.BatchNorm1d(512)
        self.sn1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512,out_channels=256,kernel_size=5,stride=1,padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.sn2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.sn3 = nn.ReLU()

        self.fc4 = nn.Linear(128*self.T, 35)
   

    def forward(self, x_input: torch.Tensor):
        
        x = x_input.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.sn3(x)

        x = x.flatten(1)

        x = self.fc4(x)
     
        out_rec_TBdims = x.unsqueeze(0)
     
        return out_rec_TBdims

