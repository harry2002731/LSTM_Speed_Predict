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
import math

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



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
    

class MultiInputLSTM(nn.Module):
    def __init__(self, input_size, output_size,input_num, output_num):
        super(MultiInputLSTM, self).__init__()
        self.input_size = input_size #每个输入的数据的长度
        
        self.input_num = input_num # 输入数据的个数
        self.output_num = output_num #输出数据的个数
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(input_num, 1)),
        )
     
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.LeakyReLU(0.1, inplace=False)

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1),
        )
        self.bn2 = nn.BatchNorm1d(1)
        self.relu2 = nn.LeakyReLU(0.1, inplace=False)

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1),
        )
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.LeakyReLU(0.1, inplace=False)

        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(3 * self.input_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=3 * self.input_size, nhead=5, dropout=0.1,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3) 
            
        self.concat_LSTM = nn.LSTM(3 * self.input_size, 128, 2, batch_first=True, dropout=0.1)
        # self.transformer = nn.Transformer(d_model=16 * self.input_size, nhead=2, num_encoder_layers=2, num_decoder_layers=2,batch_first=True)

        self.fc = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.fc3 = nn.Linear(32, output_size)

        self.model_type = 'Transformer'
        
   
        self.decoder = nn.Linear(3 * self.input_size,64)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask                
    def forward(self, *inputs):
        if len(inputs) == 2:
            stack_data = torch.stack((inputs[0], inputs[1]), dim=1).permute(0, 2, 1, 3)
        elif len(inputs) == 3:
            stack_data = torch.cat((inputs[0], inputs[1],inputs[2]), dim=2)
            
        output = self.transformer_encoder(stack_data,self.src_mask)#, self.src_mask)
        out, _ = self.concat_LSTM(output)
        out = self.fc(out[:, -1, :])  # 全连接层
        out = self.fc2(out)  # 全连接层
        
        return out




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
# import math

# class DoubleLstmInputLSTM(nn.Module):
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
#         self.input_size = input_size

#     def forward(self, x1, x2):
#         x1 = x1.view(-1, 1, self.input_size)
#         real_out, _ = self.Real_LSTM(x1)
#         real_out_bn = self.bn1(real_out[:, -1, :])

#         x2 = x2.view(-1, 1, self.input_size)
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



# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, max_len=150):
#         super(PositionalEncoding, self).__init__()       
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         #pe.requires_grad = False
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :,:1].repeat(1,x.shape[1],1)
    
    

# class MultiInputLSTM(nn.Module):
#     def __init__(self, input_size, output_size,input_num, output_num):
#         super(MultiInputLSTM, self).__init__()
#         self.input_size = input_size #每个输入的数据的长度
        
#         self.input_num = input_num # 输入数据的个数
#         self.output_num = output_num #输出数据的个数
        
#         self.conv_layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(input_num, 1)),
#         )
     
#         self.bn1 = nn.BatchNorm2d(1)
#         self.relu1 = nn.LeakyReLU(0.1, inplace=False)

#         self.conv_layer2 = nn.Sequential(
#             nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1),
#         )
#         self.bn2 = nn.BatchNorm1d(1)
#         self.relu2 = nn.LeakyReLU(0.1, inplace=False)

#         self.conv_layer3 = nn.Sequential(
#             nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1),
#         )
#         self.bn3 = nn.BatchNorm1d(32)
#         self.relu3 = nn.LeakyReLU(0.1, inplace=False)

        
#         self.src_mask = None
#         self.pos_encoder = PositionalEncoding(3 * self.input_size)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=3 * self.input_size, nhead=6, dropout=0.1,batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2) 
            
#         self.concat_LSTM = nn.LSTM(3 * self.input_size, 128, 2, batch_first=True, dropout=0.1)
#         # self.transformer = nn.Transformer(d_model=16 * self.input_size, nhead=2, num_encoder_layers=2, num_decoder_layers=2,batch_first=True)

#         # self.fc = nn.Linear(128, 64)
#         # self.fc2 = nn.Linear(64, output_size)
#         # self.fc3 = nn.Linear(32, output_size)

#         self.fc = nn.Linear(3 * self.input_size, 64)
#         self.fc2 = nn.Linear(64, output_size)
#         self.fc3 = nn.Linear(32, output_size)
#         self.model_type = 'Transformer'
        
   
#         self.decoder = nn.Linear(3 * self.input_size,64)
        
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1    
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask                
#     def forward(self, *inputs):
#         if len(inputs) == 2:
#             stack_data = torch.stack((inputs[0], inputs[1]), dim=1).permute(0, 2, 1, 3)
#         elif len(inputs) == 3:
#             stack_data = torch.cat((inputs[0], inputs[1],inputs[2]), dim=2).permute(2,0,1)
#             # stack_data = torch.cat((inputs[0], inputs[1],inputs[2]), dim=2)

#         # stack_data = self.input_embedding(stack_data) # linear transformation before positional embedding
            
#         stack_data = self.pos_encoder(stack_data).permute(1,2,0)
#         output = self.transformer_encoder(stack_data,self.src_mask)#, self.src_mask)

#         # out, _ = self.concat_LSTM(output)
#         # out = self.fc(out[:, -1, :])  # 全连接层
#         out = self.fc(output)  # 全连接层
#         out = self.fc2(out)  # 全连接层
        
#         return out
