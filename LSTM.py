import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time
from modelLSTM import MultiInputLSTM
import onnx
import onnxruntime
import MNN
from DataProcess import *
import subprocess
batch_size = 8192

input_size = 50
hidden_size = 16
hidden_size2 = 32
num_layers = 3
output_size = 2
num_epochs = 1200
# 学习率
learning_rate = 0.001
# betas参数
betas = (0.9, 0.999)
# eps参数
eps = 1e-8
# 权重衰减
weight_decay = 0
# 是否使用AMSGrad
amsgrad = True

def genDataloader(data_path, device , input_size, random_state, if_random , if_torch):
    train_, test_ = loadData(data_path, device , input_size, random_state, if_random , if_torch)
    x1_train_data, x2_train_data,x3_train_data ,train_labels,train_labels2, train_file_name = train_
    x1_test_data, x2_test_data, x3_test_data, test_labels, test_labels2,test_file_name = test_
    dataset = Mydataset(x1_train_data, x2_train_data, x3_train_data,
                        train_labels,train_labels2, train_file_name)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Mydataset(x1_test_data, x2_test_data,x3_test_data ,
                             test_labels,test_labels2, test_file_name)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

def genDataloader2(data_path, device , input_size, random_state, if_random , if_torch):
    train_, test_ = loadData2(data_path, device , input_size, random_state, if_random , if_torch)
    x1_train_data, x2_train_data ,train_labels, train_file_name = train_
    x1_test_data, x2_test_data, test_labels,test_file_name = test_
    dataset = Mydataset2(x1_train_data, x2_train_data,
                        train_labels, train_file_name)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Mydataset2(x1_test_data, x2_test_data ,
                             test_labels, test_file_name)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

def genDataloader3(data_path, device , input_size, random_state, if_random , if_torch):
    train_, test_ = loadData3(data_path, device , input_size, random_state, if_random , if_torch)
    x1_train_data, x2_train_data,x3_train_data ,train_labels, train_file_name = train_
    x1_test_data, x2_test_data, x3_test_data, test_labels,test_file_name = test_
    dataset = Mydataset3(x1_train_data, x2_train_data,x3_train_data,
                        train_labels, train_file_name)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Mydataset3(x1_test_data, x2_test_data ,x3_test_data,
                             test_labels, test_file_name)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

class Mydataset(data.Dataset):
    def __init__(self, x1, x2, x3,  y, y2, file_name):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y = y
        self.y2 = y2
        self.file_name = file_name
        self.idx = list()
        self.idy = list()
        for i, item in enumerate(x1):
            self.idx.append([self.x1[i], self.x2[i], self.x3[i]])
        for i, item in enumerate(y):
            self.idy.append([self.y[i], self.y2[i]])
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.idy[index]
        file_name = self.file_name[index]
        return input_data, target, file_name

    def __len__(self):
        return len(self.idx)

class Mydataset2(data.Dataset):
    def __init__(self, x1, x2, y, file_name):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.file_name = file_name
        self.idx = list()
        self.idy = list()
        for i, item in enumerate(x1):
            self.idx.append([self.x1[i], self.x2[i]])
        for i, item in enumerate(y):
            self.idy.append(self.y[i])
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.idy[index]
        file_name = self.file_name[index]
        return input_data, target, file_name

    def __len__(self):
        return len(self.idx)

class Mydataset3(data.Dataset):
    def __init__(self, x1, x2, x3,  y, file_name):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y = y
        self.file_name = file_name
        self.idx = list()
        self.idy = list()
        for i, item in enumerate(x1):
            self.idx.append([self.x1[i], self.x2[i], self.x3[i]])
        for i, item in enumerate(y):
            self.idy.append(self.y[i])
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.idy[index]
        file_name = self.file_name[index]
        return input_data, target, file_name

    def __len__(self):
        return len(self.idx)
    

def train(device, input_num, output_num, debug = False,pretrained=False):
    start_time = time.time()
    if input_num == 2:
        train_loader, test_loader = genDataloader2(r"data/train.txt", device, input_size, 0.1, True, True)
    elif input_num == 3 and output_num == 1:
        train_loader, test_loader = genDataloader3(r"data/train.txt", device, input_size, 0.1, True, True)
    elif input_num == 3 and output_num == 2:
        train_loader, test_loader = genDataloader(r"data/train.txt", device, input_size, 0.1, True, True)
        
    # 定义损失函数和优化器
    loss_function = nn.MSELoss()

    # 创建模型实例
    if pretrained:
        print("Load pretrained model \n")
        model = torch.load(r"model/model.pth").to(device)
    else:
        model = MultiInputLSTM(input_size, 1, input_num, output_num).to(device, dtype=torch.float64)

    # 创建Adam优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    iter = 0
    min_loss = 20
    print("TRAINING STARTED.\n")
    start_time = time.time()
    for epoch in range(num_epochs):
        model_saved = False
        for i, (train_datum, train_labels, _) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()  # 清空之前的梯度信息
            train_model_inputs = [i.view(-1, 1,input_size).to(dtype=torch.float64) for i in train_datum]
            train_labels = train_labels.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
            if input_num == 3 and output_num == 1:
                outputs = model(train_model_inputs[0], train_model_inputs[1],train_model_inputs[2])
            elif input_num == 2 and output_num == 1:
                outputs = model(train_model_inputs[0], train_model_inputs[1])

            outputs *= 1000
            train_labels *= 1000
            loss = loss_function(outputs, train_labels)

            loss.backward()
            optimizer.step()

            iter += 1
            if iter % 200 == 0:
                model.eval()
                loss_ = []
                file_name_list = []
                file_error_list = []
                for i, (test_datum, test_labels, test_file_name) in enumerate(test_loader):
                    test_model_inputs = [i.view(-1, 1,input_size).to(dtype=torch.float64) for i in test_datum]
                    test_labels = test_labels.unsqueeze(1)
                                       
                    if input_num == 3 and output_num == 1:
                        outputs = model(test_model_inputs[0], test_model_inputs[1],test_model_inputs[2])
                    elif input_num == 2 and output_num == 1:
                        outputs = model(test_model_inputs[0], test_model_inputs[1])
                    outputs *= 1000
                    test_labels *= 1000
                    l = loss_function(
                        outputs, test_labels).cpu().detach().numpy()

                    # 监视有问题的数据来源的文件名称
                    loss_.append(l)
                    if l > 50 and debug:
                        file_error_list.append([test_file_name[0], test_datum,outputs/1000,test_labels/1000, l])
                        file_name_list.append(test_file_name[0])
                        
                avg_ = np.mean(np.array(loss_))
                if debug:
                    for name in set(file_name_list):
                        print(name, file_name_list.count(name))
                        
                        # if (name == "robokit_2024-09-02_17-46-38.1.log.txt"):
                        for error in file_error_list:
                            if error[0] == name:
                                # print(error[0],error[2].cpu().detach().numpy(),error[3].cpu().detach().numpy(),list(error[1][2].cpu().detach().numpy()),error[-1],"|",list(error[1][0].cpu().detach().numpy()),list(error[1][1].cpu().detach().numpy()))
                                print(error[0],error[2].cpu().detach().numpy(),error[3].cpu().detach().numpy(),error[-1],"|",list(error[1][0].cpu().detach().numpy()),list(error[1][1].cpu().detach().numpy()))
                                break

                if loss < min_loss:
                    min_loss = loss
                    model_saved = True
                    torch.save(model, r"model/model.pth")
                print(f'Epoch: {epoch + 1}/{num_epochs}\t Loss: {loss.item():.4f} test_Loss: {avg_.item():.4f} model_saved:{model_saved}')

    print(time.time()-start_time, f"num_epochs为{num_epochs}")

def test(device, input_num=3,output_num=1):
    loss_function = nn.MSELoss()
    model = torch.load(r"model/model.pth").to(device, dtype=torch.float64)
    model.eval()

    # 打印权重信息
    # for name in model.state_dict():
    #     print(name, model.state_dict()[name])
    
    
    if input_num == 2 and output_num == 1:
        _, test_loader = genDataloader2(r"data/train.txt", device, input_size, 1.0, False, True)
    elif input_num == 3 and output_num == 1:
        _, test_loader = genDataloader3(r"data/train.txt", device, input_size, 1.0, False, True)

    iter = 0

    print("\nCALCULATING ACCURACY...\n")
    with torch.no_grad():
        pred, loss_list, y = [], [], []
        i = 0
        # Iterate through test dataset
        for i, (test_datum, test_labels, test_file_name) in enumerate(test_loader):

            test_model_inputs = [i.view(-1, 1,input_size).to(dtype=torch.float64) for i in test_datum]
            if input_num == 3 and output_num == 1:
                outputs = model(test_model_inputs[0], test_model_inputs[1],test_model_inputs[2])
            elif input_num == 2 and output_num == 1:
                outputs = model(test_model_inputs[0], test_model_inputs[1])
            outputs *= 1000
            test_labels *= 1000
            loss = loss_function(outputs, test_labels).cpu().detach().numpy()

            test_labels = test_labels.unsqueeze(1)
            test_labels = test_labels[0].to(dtype=torch.float64).cpu()  # 将目标的形状从[2]变为[2, 1]
            outputs = outputs.cpu()

            y.append(test_labels)
            pred.append(outputs[0][0])
            loss_list.append(float(loss))
            iter += 1
            if iter % 100 == 0:
                print(f"Iteration: {iter}\t Loss: {loss.item():.4f}")
                print(f"*************************************************************************")
                # break
        print(f"平均值 {sum(loss_list)/len(loss_list)} 最大loss{max(loss_list)}")

        n = i
        y, pred = np.array(y), np.array(pred)
        # pred = pred[:, 0]
        # y = y[:, 0]
        pred = pred[:]
        y = y[:]
        x = [i for i in range(0, n)]
        x_smooth = np.linspace(np.min(x), np.max(x), n)
        plt.plot(
            x_smooth, y[0:n], c="green", marker="*", ms=1, alpha=0.75, label="true"
        )
        plt.plot(
            x_smooth, pred[0:n], c="red", marker="o", ms=1, alpha=0.75, label="pred"
        )
        plt.plot(
            x_smooth,
            loss_list[0:n],
            c="blue",
            marker="o",
            ms=1,
            alpha=0.75,
            label="loss",
        )
        plt.grid(axis="y")
        plt.legend()
        plt.show()


def convert2ONNX(input_num,output_num,model_name):
    print("\CONVERTING TORCH TO ONNX...\n")
    model = torch.load(r"model/"+model_name).to(device, dtype=torch.float64)
    model.eval()

    torch_input = (
        torch.randn(batch_size, 1, input_size).to(
            device=device).to(dtype=torch.float64)
    )
    torch_input2 = (
        torch.randn(batch_size, 1, input_size).to(
            device=device).to(dtype=torch.float64)
    )
    torch_input3 = (
        torch.randn(batch_size, 1, input_size).to(
            device=device).to(dtype=torch.float64)
    )
    if input_num == 2:
        torch.onnx.export(
            model,
            (torch_input, torch_input2),
            "model/model.onnx",
            export_params=True,
            opset_version=17,  # 指定 ONNX 的操作集版本
            input_names=["input1", "input2"],  # 可以为每个输入指定名称
            output_names=["output"],  # 输出名称
            dynamic_axes={
                "input1": {0: "batch_size"},  # 指定可变长度的维度
                "input2": {0: "batch_size"},
            },
        )
    elif input_num == 3 :
            torch.onnx.export(
            model,
            (torch_input, torch_input2,torch_input3),
            "model/model.onnx",
            export_params=True,
            opset_version=17,  # 指定 ONNX 的操作集版本
            input_names=["input1", "input2","input3"],  # 可以为每个输入指定名称
            output_names=["output"],  # 输出名称
            dynamic_axes={
                "input1": {0: "batch_size"},  # 指定可变长度的维度
                "input2": {0: "batch_size"},
                "input3": {0: "batch_size"},
            },
    )


def ONNXRuntime(): 
    train_loader, test_loader = genDataloader(r"data/train.txt", device, input_size, 1.0, False, True)

    ort_session = onnxruntime.InferenceSession(
        "model/model.onnx", providers=["CUDAExecutionProvider"]
    )

    # 将张量转化为ndarray格式
    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    for i, (test_datum, test_labels, test_file_name) in enumerate(test_loader):
        real_speed = test_datum[0]
        cmd_speed = test_datum[1]
        test_datum = real_speed.view(-1, 1, input_size)
        images2 = cmd_speed.view(-1, 1, input_size)
        # 构建输入的字典和计算输出结果
        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(test_datum),
            ort_session.get_inputs()[1].name: to_numpy(images2),
        }

        ort_outs = ort_session.run(None, ort_inputs)
        print(ort_outs)
        if i == 5:
            break


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.is_available())

    # train(device,input_num=3,output_num=1, debug=True , pretrained=False)#3输入1输出
    test(device,input_num=3,output_num=1)
    # convert2ONNX(input_num=3,output_num=1,model_name="model.pth")
    # subprocess.run(['bash', 'convert.sh','model'])
    
    
    # train(device,input_num=2,output_num=1, debug=True , pretrained=False)#2输入1输出
    # convert2ONNX(input_num=2,output_num=1,model_name="model.pth")
    # subprocess.run(['bash', 'convert.sh',"model"])
    
    # ONNXRuntime()
    # ort_session = onnxruntime.InferenceSession("model.onnx",providers=["CUDAExecutionProvider"])
# 当前位置、期望目标位置、期望下发速度
