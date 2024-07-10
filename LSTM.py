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
import random

batch_size = 256
input_size = 24
hidden_size = 16
hidden_size2 = 32
num_layers = 3
output_size = 1
num_epochs = 3000
# 学习率
learning_rate = 0.005
# betas参数
betas = (0.95, 0.999)
# eps参数
eps = 1e-8
# 权重衰减
weight_decay = 0
# 是否使用AMSGrad
amsgrad = True

class Mydataset(data.Dataset):
    def __init__(self, x1,x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.idx = list()
        for i,item in enumerate(x1):
            self.idx.append([self.x1[i],self.x2[i]])
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)
    
def randomSortData(path,random_seed):
    random.seed(random_seed)  
    total_data_list = []
    with open(path, "r") as f:
        line = f.readline() # 读取第一行
        while line:
            data_list = line.split(" ")
            data_list.pop()
            data_list = [float(item) for item in data_list]
            total_data_list.append(data_list)
            line = f.readline() # 读取下一行
    random.shuffle(total_data_list)
    return total_data_list

def loadData(path,device,random_state):
    total_data_list = randomSortData(path,35)
    data_label = []
    real_data = []
    cmd_data = []
    for data_list in total_data_list:

        data_list = [float(item) for item in data_list]
        label = data_list[0]
        len_data_list = len(data_list)
        real_list = data_list[1:int((len_data_list-1)/2)+1]
        cmd_list = data_list[int((len_data_list-1)/2)+1:]
        while len(real_list) >input_size:
            real_list.pop(0)
            cmd_list.pop(0)
        if len(real_list) == input_size:
            data_label.append(label)
            real_data.append(np.array(real_list))
            cmd_data.append(np.array(cmd_list))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    real_data = scaler.fit_transform(real_data)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    cmd_data = scaler.fit_transform(cmd_data)
    len_data = int(len(real_data)*(1 - random_state))
    real_data = torch.tensor(real_data).to(device)
    cmd_data = torch.tensor(cmd_data).to(device)
    data_label = torch.tensor(np.array(data_label)).to(device)

    train_list = [real_data[0:len_data],cmd_data[0:len_data],data_label[0:len_data]]
    test_list = [real_data[len_data:-1],cmd_data[len_data:-1],data_label[len_data:-1]]

    return train_list,test_list


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.CMD_LSTM = nn.LSTM(input_size, 32, num_layers, batch_first=True)
        self.Real_LSTM = nn.LSTM(input_size, 32, num_layers, batch_first=True)
        self.concat_LSTM = nn.LSTM(32, 64, 5, batch_first=True)
        self.fc = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.5)

        self.embedding = nn.Embedding(num_embeddings=70, embedding_dim=25)

    def forward(self, x1,x2):

        real_out, _ = self.Real_LSTM(x1)
        cmd_out, _ = self.CMD_LSTM(x2) # LSTM层
        cmd_out = self.dropout(cmd_out)
        real_out = self.dropout(real_out)

        out = (real_out + cmd_out)/2
        out,_ = self.concat_LSTM(out)
        out = self.dropout(out)

        out2 = self.fc(out[:, -1, :]) # 全连接层
        return out2
    
class MultiInputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiInputLSTM, self).__init__()
        self.CMD_LSTM = nn.LSTM(24, 64, 3, batch_first=True,dropout=0.2)
        self.bn1 = nn.BatchNorm1d(64)

        self.Real_LSTM = nn.LSTM(24, 64, 3, batch_first=True,dropout=0.2)
        self.bn2 = nn.BatchNorm1d(64)

        self.concat_LSTM = nn.LSTM(496, 128, 3, batch_first=True,dropout=0.6)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv_layer1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=6,padding=2))
        self.conv_layer2 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=6,padding=2))

        self.conv_layer3 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,1)))
        self.max_pooling = nn.MaxPool1d(3, stride=2)
        self.relu =  nn.ReLU(inplace=False)

        self.fc = nn.Linear(128, output_size)

    def forward(self, x1,x2):
        x1 = x1.view(-1,1,24)
        real_out, _ = self.Real_LSTM(x1)
        real_out_bn = self.bn1(real_out[:, -1, :]) 

        x2 = x2.view(-1,1,24)
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

def train(device):
    train_, test_ = loadData(r"data\train.txt",device,0.2)
    test_data, test_data_2 ,test_labels = test_
    train_data, train_data_2 ,train_labels = train_
    dataset = Mydataset(train_data,train_data_2, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Mydataset(test_data,test_data_2, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 定义损失函数和优化器
    loss_function = nn.MSELoss()

    # 创建模型实例
    model = MultiInputLSTM(input_size, hidden_size, num_layers, output_size).to(device)

    # 创建Adam优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad
    )
    iter = 0
    print('TRAINING STARTED.\n')
    for epoch in range(num_epochs):
        for i, (motor_data, train_labels) in enumerate(dataloader):
            # pass
            model.train()
            real_motor = motor_data[0]                
            cmd_motor = motor_data[1]                
            real_motor = real_motor.view(-1, 1, input_size)
            cmd_motor = cmd_motor.view(-1, 1, input_size)
            model = model.to(device, dtype=torch.float64)

            outputs = model(real_motor,cmd_motor)
            train_labels = train_labels.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
            outputs *= 100
            train_labels *= 100
            loss = loss_function(outputs, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter += 1
            if iter % 500 == 0:
                model.eval()
                loss_ = []
                for i, (images, test_labels) in enumerate(test_loader):
                    test_real_motor = images[0].view(-1, 1, input_size)                
                    test_cmd_motor = images[1].view(-1, 1, input_size)                

                    outputs = model(test_real_motor,test_cmd_motor)
                    test_labels = test_labels.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
                    outputs *= 100
                    test_labels *= 100
                    loss_.append(loss_function(outputs, test_labels).cpu().detach().numpy())
                avg_ = np.mean(np.array(loss_))
                print(f'Epoch: {epoch + 1}/{num_epochs}\t Iteration: {iter}\t Loss: {loss.item():.4f} test_Loss: {avg_.item():.4f}') 
                
    torch.save(model, r"model.pth")

def test(device):
    loss_function = nn.MSELoss()

    model = torch.load(r"model.pth").to(device)
    train_, test_ = loadData(r"data\train.txt",device,0.2)
    test_data, test_data_2 ,test_labels = test_

    dataset = Mydataset(test_data,test_data_2, test_labels)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    iter = 0
    model.eval()
    print('\nCALCULATING ACCURACY...\n')
    with torch.no_grad():
        pred = []
        y = []
        loss_list = []
        i = 0
        # Iterate through test dataset
        for i, (images, test_labels) in enumerate(test_loader):
            imgs1 = images[0]                
            imgs2 = images[1]                
            images = imgs1.view(-1, 1, input_size)
            images2 = imgs2.view(-1, 1, input_size)

            outputs = model(images,images2)
            test_labels = test_labels.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
            outputs *= 100
            test_labels *= 100
            loss = loss_function(outputs, test_labels)
            test_labels = test_labels.cpu()
            outputs = outputs.cpu()
            y.append(test_labels)
            pred.append(outputs)
            loss_list.append(float(loss))
            # _, predicted = torch.max(outputs.data, 1)
            iter += 1
            if iter % 100 == 0:
                print(f'Iteration: {iter}\t Loss: {loss.item():.4f}')
        # pred = pred.cpu()
        n = i
        y, pred = np.array(y), np.array(pred)
        x = [i for i in range(0, n)]
        x_smooth = np.linspace(np.min(x), np.max(x), n)
        plt.plot(x_smooth, y[0:n], c='green', marker='*', ms=1, alpha=0.75, label='true')
        plt.plot(x_smooth, pred[0:n], c='red', marker='o', ms=1, alpha=0.75, label='pred')
        plt.plot(x_smooth, loss_list[0:n], c='blue', marker='o', ms=1, alpha=0.75, label='loss')
        plt.grid(axis='y')
        plt.legend()
        plt.show()
if __name__== "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.is_available())
    train(device)
    test(device)

