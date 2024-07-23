import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import MinMaxScaler

batch_size = 512
input_size = 30
hidden_size = 16
hidden_size2 = 32
num_layers = 3
output_size = 1
num_epochs = 900
# 学习率
learning_rate = 0.002
# betas参数
betas = (0.95, 0.999)
# eps参数
eps = 1e-8
# 权重衰减
weight_decay = 0
# 是否使用AMSGrad
amsgrad = True

class MultiInputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiInputLSTM, self).__init__()
        self.CMD_LSTM = nn.LSTM(input_size, 64, 5, batch_first=True,dropout=0.1)
        self.bn1 = nn.BatchNorm1d(64)

        self.Real_LSTM = nn.LSTM(input_size, 64, 5, batch_first=True,dropout=0.1)
        self.bn2 = nn.BatchNorm1d(64)

        self.concat_LSTM = nn.LSTM(496, 512, 5, batch_first=True,dropout=0.2)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv_layer1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=6,padding=2))
        self.conv_layer2 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=6,padding=2))

        self.conv_layer3 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,1)))
        self.max_pooling = nn.MaxPool1d(3, stride=2)
        self.relu =  nn.ReLU(inplace=False)

        self.fc = nn.Linear(512, output_size)

    def forward(self, x1,x2):
        x1 = x1.view(-1,1,input_size)
        real_out, _ = self.Real_LSTM(x1)
        real_out_bn = self.bn1(real_out[:, -1, :]) 

        x2 = x2.view(-1,1,input_size)
        cmd_out, _ = self.CMD_LSTM(x2) 
        cmd_out_bn = self.bn2(cmd_out[:, -1, :]) 
        
        stack_data = torch.stack((real_out_bn,cmd_out_bn),dim=0).permute(1, 0, 2)
        stack_data = stack_data.view(-1,1,2,64)
        stack_data = self.relu(self.conv_layer3(stack_data))
        stack_data = stack_data.view(-1,16,64)
        out = self.max_pooling(stack_data)    
        out = out.view(-1,1,496)

        out,_ = self.concat_LSTM(out)
        out = self.fc(out[:, -1, :]) # 全连接层
        return out

