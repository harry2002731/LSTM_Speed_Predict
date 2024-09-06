# import torch.nn as nn
# from torch.utils.data import Dataset
# import numpy as np
# from torch.utils.data import DataLoader
# import torch
# import torch.utils.data as data
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline
# from sklearn.preprocessing import MinMaxScaler

# batch_size = 512
# input_size =   25
# hidden_size = 16
# hidden_size2 = 32
# num_layers = 3
# output_size = 1
# num_epochs = 900
# # 学习率
# learning_rate = 0.002
# # betas参数
# betas = (0.95, 0.999)
# # eps参数
# eps = 1e-8
# # 权重衰减
# weight_decay = 0
# # 是否使用AMSGrad
# amsgrad = True


# class MultiInputLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(MultiInputLSTM, self).__init__()
#         self.CMD_LSTM = nn.LSTM(input_size, 64, 5, batch_first=True, dropout=0.3)
#         self.bn1 = nn.BatchNorm1d(64)

#         self.Real_LSTM = nn.LSTM(input_size, 64, 5, batch_first=True, dropout=0.3)
#         self.bn2 = nn.BatchNorm1d(64)

#         self.concat_LSTM = nn.LSTM(496, 512, 5, batch_first=True, dropout=0.3)
#         self.bn3 = nn.BatchNorm1d(512)

#         self.conv_layer3 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 1))
#         )
#         self.max_pooling = nn.MaxPool1d(3, stride=2)
#         self.relu = nn.ReLU(inplace=False)

#         self.fc = nn.Linear(512, output_size)

#     def forward(self, x1, x2):
#         x1 = x1.view(-1, 1, input_size)
#         real_out, _ = self.Real_LSTM(x1)
#         real_out_bn = self.bn1(real_out[:, -1, :])

#         x2 = x2.view(-1, 1, input_size)
#         cmd_out, _ = self.CMD_LSTM(x2)
#         cmd_out_bn = self.bn2(cmd_out[:, -1, :])

#         stack_data = torch.stack((real_out_bn, cmd_out_bn), dim=0).permute(1, 0, 2)
#         stack_data = stack_data.view(-1, 1, 2, 64)
#         stack_data = self.relu(self.conv_layer3(stack_data))
#         stack_data = stack_data.view(-1, 16, 64)
#         out = self.max_pooling(stack_data)
#         out = out.view(-1, 1, out.shape[1] * out.shape[2])

#         out, _ = self.concat_LSTM(out)
#         out = self.fc(out[:, -1, :])  # 全连接层
#         return out


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


output_size = 1


class MultiInputLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiInputLSTM, self).__init__()
        self.input_size = input_size
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 1)),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.1, inplace=False)

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
        )
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.LeakyReLU(0.1, inplace=False)

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1),
        )
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.LeakyReLU(0.1, inplace=False)


        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 1)),
        )
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.LeakyReLU(0.1, inplace=False)
        
        
        # self.max_pooling = nn.MaxPool2d((128, 3), stride=2)
        self.concat_LSTM = nn.LSTM(32 * self.input_size, 128, 3, batch_first=True, dropout=0.1)

        self.fc = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x1, x2):
        # a = x2[:,:,-1]-x2[:,:,-2]
        # temp = torch.tensor([1 if i > 0 else -1  for i in a]).view(-1,1,1).to(torch.device("cuda:0"))
        stack_data = torch.stack((x1, x2), dim=1).permute(0, 2, 1, 3)
        stack_data = self.conv_layer1(stack_data)
        stack_data = self.bn1(stack_data)
        stack_data = self.relu1(stack_data)

        stack_data = self.conv_layer2(stack_data.view(-1, 64, self.input_size))
        stack_data = self.bn2(stack_data)
        stack_data = self.relu2(stack_data)

        stack_data = self.conv_layer3(stack_data)
        stack_data = self.bn3(stack_data)
        stack_data = self.relu3(stack_data)

        stack_data = stack_data.view(-1, 1, 32 * self.input_size)
        # stack_data = self.max_pooling(stack_data)
        out, _ = self.concat_LSTM(stack_data)
        out = self.fc(out[:, -1, :])  # 全连接层
        out = self.fc2(out)  # 全连接层
        
        # a = x2[:,:,-1]-x2[:,:,-2]
        # temp = torch.tensor([1 if i > 0 else -1  for i in a]).view(-1,1,1).to(torch.device("cuda:0"))
        # stack_data = torch.stack((out.view(-1,1,1), temp), dim=1).permute(0, 2, 1, 3)
        # stack_data = self.conv_layer4(stack_data)
        # stack_data = self.bn4(stack_data)
        # stack_data = self.relu4(stack_data).view(-1,32)
        # stack_data = self.fc3(stack_data)  # 全连接层

        # out += temp
        return out
