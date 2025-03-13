import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time
from modelLSTM import MultiInputLSTM2
import onnx
import MNN
import onnxruntime
from DataProcess import *
import subprocess
batch_size = 4096

input_size = 20
hidden_size = 16
hidden_size2 = 32
num_layers = 3
output_size = 2
num_epochs = 2000
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


def genDataloader(data_path, device, input_size, input_param, output_param, random_state, if_random, if_torch):
    train_, test_ = loadData(data_path, device, input_size,
                             random_state, input_param, output_param, if_random, if_torch)

    dataset = Mydataset(input_param, output_param, train_)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Mydataset(input_param, output_param, test_)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


class Mydataset(data.Dataset):
    def __init__(self, input_size, output_size, *x):
        self.file_name = x[0][-1]
        self.idx = list()
        self.idy = list()
        for i, item in enumerate(x[0][0]):
            self.idx.append([x[0][index][i] for index in range(input_size)])

        for i, item in enumerate(x[0][0]):
            self.idy.append([x[0][index+input_size][i]
                            for index in range(output_size)])

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.idy[index]
        # target = self.idy[index][0]
        file_name = self.file_name[index]
        return input_data, target, file_name

    def __len__(self):
        return len(self.idx)


def train(device, input_num, output_num, debug=False, pretrained=False):
    start_time = time.time()

    train_loader, test_loader = genDataloader(
        r"data/train.txt", device, input_size, 3, output_num, 0.2, True, True)

    # 定义损失函数和优化器
    loss_function = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()  # 分类损失函数

    # 创建模型实例
    if pretrained:
        print("Load pretrained model \n")
        model = torch.load(r"model/model.pth").to(device, dtype=torch.float64)
    else:
        model = MultiInputLSTM2(input_size, input_num, output_num).to(
            device, dtype=torch.float64)

    # 创建Adam优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    correct = 0
    total_num = 0
    min_loss = 500
    print("TRAINING STARTED.\n")
    start_time = time.time()
    for epoch in range(1,num_epochs):
        model_saved = False
        for iter, (train_datum, train_labels, _) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()  # 清空之前的梯度信息
            train_model_inputs = [
                i.view(-1, 1, input_size).to(dtype=torch.float64) for i in train_datum]

            tensor_unsqueezed = train_labels[1].unsqueeze(-1).unsqueeze(-1)  # 形状变为 [4096, 1, 1]
            tensor_expanded = tensor_unsqueezed.repeat(1, 1, input_size)  # 形状变为 [4096, 1, 20]

            train_model_inputs.append(tensor_expanded)
            train_labels = torch.stack(train_labels).T
            outputs = model(train_model_inputs)

            loss_reg = loss_function(
                outputs * 1000, train_labels[:, 0] * 1000)
            total_loss = loss_reg  # 将两个损失相加

            total_loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            # print(f"Loss: {total_loss.item():.4f}",loss_reg.item(),loss_cls.item())
            true_list, pred_list, loss_list, test_acc, file_error_list = test(
                model, test_loader, input_num, output_num, train_mode=True)
            avg_ = np.mean(np.array(loss_list))
            if total_loss < min_loss:
                min_loss = total_loss
                model_saved = True
                torch.save(model, r"model/model.pth")
            # print(f'Epoch: {epoch + 1}/{num_epochs}\t Loss: {loss.item():.4f} test_Loss: {avg_.item():.4f} model_saved:{model_saved}')
            # print(f'Epoch: {epoch + 1}/{num_epochs}\t || Train: Loss: {total_loss.item():.4f} Classify_ACC {correct/total_num} || Test: Loss: {avg_}  Classify_ACC {test_acc} || Model_Saved:{model_saved}')
            print(f'Epoch: {epoch + 1}/{num_epochs}\t || Train: Loss: {total_loss.item():.4f}  || Test: Loss: {avg_}  || Model_Saved:{model_saved}')

    print(time.time()-start_time, f"num_epochs为{num_epochs}")


# 测试loader中的所有数据
def test(model, test_loader, input_num, output_num, train_mode=False):
    model.eval()
    loss_function = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()  # 分类损失函数

    # 打印权重信息
    # for name in model.state_dict():
    #     print(name, model.state_dict()[name])
    iter = 0
    correct = 0
    print("\nCALCULATING ACCURACY...\n")
    with torch.no_grad():
        pred_list, loss_list, true_list = [], [], []
        for i, (test_datum, test_labels, test_file_name) in enumerate(test_loader):
            test_labels = torch.stack(test_labels).T

            test_model_inputs = [
                i.view(-1, 1, input_size).to(dtype=torch.float64) for i in test_datum]
            
            tensor_unsqueezed = test_labels[0][1].unsqueeze(-1).unsqueeze(-1)  # 形状变为 [4096, 1, 1]
            tensor_expanded = tensor_unsqueezed.repeat(1, 1, input_size)  # 形状变为 [4096, 1, 20]
            test_model_inputs.append(tensor_expanded)
            
            outputs = model(test_model_inputs)

            loss_reg = loss_function(
                outputs * 1000, test_labels[:, 0] * 1000)
            loss = loss_reg  # 简单地将两个损失相加


            loss_list.append(float(loss.cpu().detach().numpy()))
            iter += 1

        # print(f"平均值 {sum(loss_list)/len(loss_list)} 最大损失 {max(loss_list)} 分类准确度 {correct/len(loss_list)}")

    # return true_list, pred_list, loss_list, file_name_list, file_error_list
    return [], [], loss_list, 0, []


def convert2ONNX(input_num, output_num, model_name, batch_size, input_size, device="cpu", opset_version=17):
    print("\nCONVERTING TORCH TO ONNX...\n")
    # 加载模型
    model = torch.load(f"model/{model_name}").to(device, dtype=torch.float64)
    model.eval()

    # 动态生成输入张量
    torch_inputs = [
        torch.randn(batch_size, 1, input_size).to(
            device=device).to(dtype=torch.float64)
        for _ in range(input_num)
    ]

    # 动态设置输入名称和动态轴
    input_names = [f"input{i+1}" for i in range(input_num)]
    dynamic_axes = {input_names[i]: {0: "batch_size"}
                    for i in range(input_num)}

    # 动态设置输出名称
    output_names = [f"output{i+1}" for i in range(output_num)]

    # 导出为ONNX模型
    torch.onnx.export(
        model,
        tuple(torch_inputs),  # 输入
        f"model/model.onnx",  # 输出文件名
        export_params=True,
        opset_version=opset_version,  # 指定ONNX的操作集版本
        input_names=input_names,  # 输入名称
        output_names=output_names,  # 输出名称
        dynamic_axes=dynamic_axes,  # 动态轴
    )

    print(f"Model converted to ONNX and saved as model/model.onnx")


def ONNXRuntime():
    train_loader, test_loader = genDataloader(
        r"data/train.txt", device, input_size, 1.0, False, True)

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
    # mode = input("mode:")
    mode = "1"
    if mode == "train" or mode == "1":
        train(device, input_num=4, output_num=2,
              debug=True, pretrained=True)  # 3输入1输
        # test(device,input_num=3,output_num=1)
        convert2ONNX(
            input_num=4,
            output_num=2,
            model_name="model.pth",
            batch_size=batch_size,
            input_size=input_size,
            device=device,
            opset_version=17
        )
        subprocess.run(['bash', 'convert.sh', 'model'])
    elif mode == "convert" or mode == "2":
        convert2ONNX(
            input_num=4,
            output_num=2,
            model_name="model.pth",
            batch_size=batch_size,
            input_size=input_size,
            device=device,
            opset_version=17
        )
        subprocess.run(['bash', 'convert.sh', 'model'])
