import torch
import torch.nn as nn
import numpy as np
import time
import math


# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()       
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         pe = pe.permute(0,2,1)
#         #pe.requires_grad ,= False
#         self.register_buffer('pe', pe)
        
#         # 5000 len 
#         # 1 5000 250 
        
#         # 5000 1 250  -> 5000 250 1 
#         # 80 46 1   -> 46 1 80   //me

#     def forward(self, x):
#         return x + self.pe[:x.size(0),:,:]
#         # return x + self.pe[:x.size(0), :]
    
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=30):
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
       
       

    
class TransAm(nn.Module):
    def __init__(self,input_size,input_num, output_num, feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.input_size = input_size
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
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1),
        )
        self.bn3 = nn.BatchNorm1d(1)
        self.relu3 = nn.LeakyReLU(0.1, inplace=False)
        
        self.src_mask = None
        feature_size = 20
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        # self.init_weights()
        self.fc1 = nn.Linear(20, 1)
        
    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,*inputs):
        
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
        stack_data = stack_data.view(-1, 1,  self.input_size)
        stack_data = stack_data.permute(2,0,1)

        if self.src_mask is None or self.src_mask.size(0) != len(stack_data):
            device = stack_data.device
            mask = self._generate_square_subsequent_mask(len(stack_data)).to(device)
            self.src_mask = mask
        # print(self.src_mask)
        src = self.pos_encoder(stack_data)
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output)
        output = output.permute(1,2,0)

        output = self.fc1(output)  # 全连接层
        
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
