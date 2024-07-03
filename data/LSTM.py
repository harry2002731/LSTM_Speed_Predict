import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
input_size = 90
hidden_size = 128
num_layers = 3
output_size = 1
learning_rate = 0.001
num_epochs = 1500

# C:\Projects\Python\speed_control\result.txt
# with open(, "r") as f:
class Mydataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)
    
def loadData(path,device):
    with open(path, "r") as f:
        line = f.readline() # 读取第一行
        data_set = []
        data_label = []

        while line:
            data_list = line.split(" ")
            data_list.pop()
            data_list = [float(item) for item in data_list]
            label = data_list[0]
            while len(data_list)-2>input_size:
                data_list.pop(1)

            data_set.append(data_list[1:-1])
            data_label.append(label)
            line = f.readline() # 读取下一行
        return torch.tensor(np.array(data_set)).to(device), torch.tensor(np.array(data_label)).to(device)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.embedding = nn.Embedding(num_embeddings=70, embedding_dim=25)

    def forward(self, x):
        # embedded = self.embedding(x)
        out, _ = self.lstm(x) # LSTM层
        out = self.dropout(out)
        out = self.fc(out[:, -1, :]) # 全连接层
        return out

def train(device):
    train_data, train_labels = loadData(r"C:\Projects\Python\speed_control\result.txt",device)
    dataset = Mydataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=60, shuffle=False)
    # 创建模型实例
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    # 定义损失函数和优化器
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iter = 0
    print('TRAINING STARTED.\n')
    for epoch in range(num_epochs):
        for i, (images, train_labels) in enumerate(dataloader):
            # pass
            images = images.view(-1, 1, input_size)
            # train_labels = train_labels.to(device)
            model = model.to(device, dtype=torch.float64)

            outputs = model(images)
            train_labels = train_labels.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
            outputs *= 100
            train_labels *= 100
            loss = loss_function(outputs, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter += 1
            if iter % 500 == 0:
                # Calculate Loss
                print(f'Epoch: {epoch + 1}/{num_epochs}\t Iteration: {iter}\t Loss: {loss.item():.4f}')

    torch.save(model, r"C:\Projects\Python\speed_control\\test.pth")

def test(device):
    loss_function = nn.MSELoss()

    model = torch.load(r"C:\Projects\Python\speed_control\\test.pth").to(device)
    test_data, test_labels = loadData(r"C:\Projects\Python\speed_control\test.txt",device)
    dataset = Mydataset(test_data, test_labels)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    iter = 0
    # model.eval()
    print('\nCALCULATING ACCURACY...\n')
    with torch.no_grad():
        pred = []
        y = []
        correct = 0
        total = 0
        progress = tqdm.tqdm(test_loader, total=len(test_loader))
        # Iterate through test dataset
        for i, (images, test_labels) in enumerate(test_loader):

        # for images, labels in progress:
            images = images.view(-1, 1, input_size)

            outputs = model(images)
            test_labels = test_labels.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
            outputs *= 100
            test_labels *= 100
            loss = loss_function(outputs, test_labels)
            test_labels = test_labels.cpu()
            outputs = outputs.cpu()
            y.append(test_labels)
            pred.append(outputs)
            # _, predicted = torch.max(outputs.data, 1)
            iter += 1
            if iter % 10 == 0:
                print(f'Iteration: {iter}\t Loss: {loss.item():.4f}')
        pred = pred.cpu()

        y, pred = np.array(y), np.array(pred)
        x = [i for i in range(0, 100)]
        x_smooth = np.linspace(np.min(x), np.max(x), 100)
        y_smooth = make_interp_spline(x, y[0:100])(x_smooth)
        plt.plot(x_smooth, y[0:100], c='green', marker='*', ms=1, alpha=0.75, label='true')

        y_smooth = make_interp_spline(x, pred[0:100])(x_smooth)
        plt.plot(x_smooth, pred[0:100], c='red', marker='o', ms=1, alpha=0.75, label='pred')
        plt.grid(axis='y')
        plt.legend()
        plt.show()
if __name__== "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print(torch.cuda.is_available())
    # train(device)
    test(device)

