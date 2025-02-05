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

class DoubleLstmInputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiInputLSTM, self).__init__()
        self.CMD_LSTM = nn.LSTM(input_size, 64, 5, batch_first=True, dropout=0.3)
        self.bn1 = nn.BatchNorm1d(64)

        self.Real_LSTM = nn.LSTM(input_size, 64, 5, batch_first=True, dropout=0.3)
        self.bn2 = nn.BatchNorm1d(64)

        self.concat_LSTM = nn.LSTM(496, 512, 5, batch_first=True, dropout=0.3)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 1))
        )
        self.max_pooling = nn.MaxPool1d(3, stride=2)
        self.relu = nn.ReLU(inplace=False)

        self.fc = nn.Linear(512, output_size)
        self.input_size = input_size

    def forward(self, x1, x2):
        x1 = x1.view(-1, 1, self.input_size)
        real_out, _ = self.Real_LSTM(x1)
        real_out_bn = self.bn1(real_out[:, -1, :])

        x2 = x2.view(-1, 1, self.input_size)
        cmd_out, _ = self.CMD_LSTM(x2)
        cmd_out_bn = self.bn2(cmd_out[:, -1, :])

        stack_data = torch.stack((real_out_bn, cmd_out_bn), dim=0).permute(1, 0, 2)
        stack_data = stack_data.view(-1, 1, 2, 64)
        stack_data = self.relu(self.conv_layer3(stack_data))
        stack_data = stack_data.view(-1, 16, 64)
        out = self.max_pooling(stack_data)
        out = out.view(-1, 1, out.shape[1] * out.shape[2])

        out, _ = self.concat_LSTM(out)
        out = self.fc(out[:, -1, :])  # 全连接层
        return out


class WeightClassify(nn.Module):
    def __init__(self, *args, **kwargs):
        super(WeightClassify,self).__init__(*args, **kwargs)
        
    def forward(self):
        pass    
        

class MultiInputLSTM(nn.Module):
    def __init__(self, input_size,input_num, output_num):
        super(MultiInputLSTM, self).__init__()
        self.input_size = input_size #每个输入的数据的长度
        
        self.input_num = input_num # 输入数据的个数
        self.output_num = output_num #输出数据的个数
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(input_num, 1)),
        )
     
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.LeakyReLU(0.1, inplace=False)

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.LeakyReLU(0.1, inplace=False)

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
        )
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.LeakyReLU(0.1, inplace=False)

        # self.max_pooling = nn.MaxPool2d((128, 3), stride=2)
        self.concat_LSTM = nn.LSTM(32 * self.input_size, 128, 5, batch_first=True, dropout=0.1,bidirectional=True)

        self.fc1 = nn.Linear(128*2, 64)
        self.fc1_1 = nn.Linear(64, 1)
        
        self.fc2 = nn.Linear(32* self.input_size, 64)
        self.fc2_1 = nn.Linear(64, 10)

        self.fc3 = nn.Linear(128*2, 64)
        self.fc3_1 = nn.Linear(64, 1)        
        
        self.classification_layer = nn.Linear(10, 3)  # 输出维度为2，对应二分类
        self.s = nn.Softmax(dim=1)

                        
        

    def forward(self, *inputs):
        if len(inputs) == 2:
            stack_data = torch.stack((inputs[0], inputs[1]), dim=1).permute(0, 2, 1, 3)
        elif len(inputs) == 3:
            stack_data = torch.stack((inputs[0], inputs[1],inputs[2]), dim=1).permute(0, 2, 1, 3)
        elif len(inputs) == 4:
            stack_data = torch.stack((inputs[0], inputs[1],inputs[2],inputs[3]), dim=1).permute(0, 2, 1, 3)
        elif len(inputs) == 5:
            stack_data = torch.stack((inputs[0], inputs[1],inputs[2],inputs[3],inputs[4]), dim=1).permute(0, 2, 1, 3)
        stack_data = self.conv_layer1(stack_data)
        stack_data = self.bn1(stack_data)
        stack_data = self.relu1(stack_data)

        stack_data = self.conv_layer2(stack_data.view(-1, 128, self.input_size))
        stack_data = self.bn2(stack_data)
        stack_data = self.relu2(stack_data)

        stack_data = self.conv_layer3(stack_data)
        stack_data = self.bn3(stack_data)
        stack_data = self.relu3(stack_data)
        
        stack_data = stack_data.view(-1, 1, 32 * self.input_size)
        # stack_data = self.max_pooling(stack_data)
        out, _ = self.concat_LSTM(stack_data)

        out = out[:, -1, :]        
        if self.output_num == 1:
            out = self.fc1(out)  # 全连接层
            out = self.fc1_1(out)  # 全连接层
        elif self.output_num == 2: 
            pred1, pred2 = self.fc1(out), self.fc2(stack_data)
            pred1, pred2 = self.fc1_1(pred1), self.fc2_1(pred2)
            pred2 = self.classification_layer(pred2)
            # pred2 = self.s(pred2)
            # out = torch.stack([pred1[:,-1], pred2[:,-1]], dim=0)
            # out = out.permute(1,0)
        elif self.output_num == 3:
            pred1, pred2, pred3 = self.fc1(out), self.fc2(out), self.fc3(out)
            pred1, pred2, pred3 = self.fc1_1(pred1), self.fc2_1(pred2), self.fc3_1(pred3)
            out = torch.stack([pred1[:,-1], pred2[:,-1], pred3[:,-1]], dim=0)
            out = out.permute(1,0)
        return pred1[:,-1],pred2
