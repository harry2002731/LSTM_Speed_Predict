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
import time
from modelLSTM import MultiInputLSTM
import onnx
import onnxruntime
import MNN

batch_size = 512
input_size = 30
hidden_size = 16
hidden_size2 = 32
num_layers = 3
output_size = 1
num_epochs = 2000
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
    
def randomSortData(path,random_seed,if_random):
    if if_random:
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
    else:
        total_data_list = []
        with open(path, "r") as f:
            line = f.readline() # 读取第一行
            while line:
                data_list = line.split(" ")
                data_list.pop()
                data_list = [float(item) for item in data_list]
                total_data_list.append(data_list)
                line = f.readline() # 读取下一行
    return total_data_list

def loadData(path,device,random_state,if_random):
    total_data_list = randomSortData(path,35,if_random)
    data_label = []
    real_data = []
    cmd_data = []
    for data_list in total_data_list:

        data_list = [float(item) for item in data_list]
        label = data_list[0]
        len_data_list = len(data_list)
        real_list = data_list[1:int((len_data_list-1)/2)+1]
        cmd_list = data_list[int((len_data_list-1)/2)+1:]
        #可能导致比较大的问题
        while len(real_list) >input_size:
            real_list.pop(0)
        while len(cmd_list) >input_size:
            cmd_list.pop(0)
        if len(real_list) == input_size:
            data_label.append(label)
            real_data.append(np.array(real_list))
            cmd_data.append(np.array(cmd_list))

    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # real_data = scaler.fit_transform(real_data)
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # cmd_data = scaler.fit_transform(cmd_data)
    len_data = int(len(real_data)*(1 - random_state))
    real_data = torch.tensor(real_data).to(device)
    cmd_data = torch.tensor(cmd_data).to(device)
    data_label = torch.tensor(np.array(data_label)).to(device)

    train_list = [real_data[0:len_data],cmd_data[0:len_data],data_label[0:len_data]]
    test_list = [real_data[len_data:-1],cmd_data[len_data:-1],data_label[len_data:-1]]

    return train_list,test_list

def train(device):
    start_time = time.time()
    train_, test_ = loadData(r"data/train.txt",device,0.5,True)
    test_data, test_data_2 ,test_labels = test_
    train_data, train_data_2 ,train_labels = train_
    dataset = Mydataset(train_data,train_data_2, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Mydataset(test_data,test_data_2, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 定义损失函数和优化器
    loss_function = nn.MSELoss()

    # 创建模型实例
    model = MultiInputLSTM(input_size, hidden_size, num_layers, output_size).to(device, dtype=torch.float64)

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
            start_time = time.time()

            # pass
            model.train()
            real_motor = motor_data[0]                
            cmd_motor = motor_data[1]                
            real_motor = real_motor.view(-1, 1, input_size).to(dtype=torch.float64)
            cmd_motor = cmd_motor.view(-1, 1, input_size).to(dtype=torch.float64)

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
    print(time.time()-start_time,f"num_epochs为{num_epochs}")


def test(device):
    loss_function = nn.MSELoss()
    model = torch.load(r"model.pth").to(device)

    train_, test_ = loadData(r"data/train.txt",device,1.0,False)
    test_data, test_data_2 ,test_labels = test_
    dataset = Mydataset(test_data,test_data_2, test_labels)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    iter = 0
    model.eval()
    print('\nCALCULATING ACCURACY...\n')
    with torch.no_grad():
        pred, loss_list, y = [], [], []
        i = 0
        # Iterate through test dataset
        for i, (images, test_labels) in enumerate(test_loader):
            imgs1 = images[0]                
            imgs2 = images[1]                
            images = imgs1.view(-1, 1, input_size).to(dtype=torch.float64)
            images2 = imgs2.view(-1, 1, input_size).to(dtype=torch.float64)
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
        print(f"平均值 {sum(loss_list)/len(loss_list)} 最大loss{max(loss_list)}")

        # n = i
        # y, pred = np.array(y), np.array(pred)
        # x = [i for i in range(0, n)]
        # x_smooth = np.linspace(np.min(x), np.max(x), n)
        # plt.plot(x_smooth, y[0:n], c='green', marker='*', ms=1, alpha=0.75, label='true')
        # plt.plot(x_smooth, pred[0:n], c='red', marker='o', ms=1, alpha=0.75, label='pred')
        # plt.plot(x_smooth, loss_list[0:n], c='blue', marker='o', ms=1, alpha=0.75, label='loss')
        # plt.grid(axis='y')
        # plt.legend()
        # plt.show()

def convert2ONNX():
    print('\CONVERTING TORCH TO ONNX...\n')
    model = torch.load(r"model.pth").to(device)
    model.eval()

    torch_input = torch.randn(batch_size, 1, input_size).to(device=device).to(dtype=torch.float64)
    torch_input2 = torch.randn(batch_size, 1, input_size).to(device=device).to(dtype=torch.float64)
    torch.onnx.export(model, (torch_input, torch_input2),"model.onnx",                  
                  export_params=True,
                  opset_version=17,  # 指定 ONNX 的操作集版本
                  input_names=["input1", "input2"],  # 可以为每个输入指定名称
                  output_names=["output"],  # 输出名称
                  dynamic_axes={"input1": {0: "batch_size"},  # 指定可变长度的维度
                                "input2": {0: "batch_size"}}
                )
    


def ONNXRuntime():
    train_, test_ = loadData(r"data/train.txt",device,1.0,False)
    test_data, test_data_2 ,test_labels = test_

    dataset = Mydataset(test_data,test_data_2, test_labels)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ort_session = onnxruntime.InferenceSession("model.onnx",providers=["CUDAExecutionProvider"])

    # 将张量转化为ndarray格式
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    for i, (images, test_labels) in enumerate(test_loader):
        imgs1 = images[0]                
        imgs2 = images[1]                
        images = imgs1.view(-1, 1, input_size)
        images2 = imgs2.view(-1, 1, input_size)
        # 构建输入的字典和计算输出结果
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)
                    ,ort_session.get_inputs()[1].name: to_numpy(images2)}
        
        ort_outs = ort_session.run(None, ort_inputs)
        print(ort_outs)
        if i ==5:
            break
# def MNNRuneTime():


if __name__== "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # print(torch.cuda.is_available())
    # train(device)
    # test(device)
    # convert2ONNX()
    ONNXRuntime()
    # ort_session = onnxruntime.InferenceSession("model.onnx",providers=["CUDAExecutionProvider"])







