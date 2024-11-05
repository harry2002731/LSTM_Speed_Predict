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



def genDataloader(data_path, device , input_size, input_param, output_param, random_state, if_random , if_torch):
    train_, test_ = loadData(data_path, device , input_size, random_state, input_param, output_param, if_random , if_torch)

    dataset = Mydataset(input_param, output_param,train_)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Mydataset(input_param, output_param,test_)
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
            self.idy.append([x[0][index+input_size][i] for index in range(output_size)])


    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.idy[index][0]
        file_name = self.file_name[index]
        return input_data, target, file_name

    def __len__(self):
        return len(self.idx)
    
    
    
    
def train(device, input_num, output_num, debug = False,pretrained=False):
    start_time = time.time()

    train_loader, test_loader = genDataloader(r"data/train.txt", device, input_size, input_num, output_num, 0.1, True, True)
        
    # 定义损失函数和优化器
    loss_function = nn.MSELoss()

    # 创建模型实例
    if pretrained:
        print("Load pretrained model \n")
        model = torch.load(r"model/model.pth").to(device, dtype=torch.float64)
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
    min_loss = 50
    print("TRAINING STARTED.\n")
    start_time = time.time()
    for epoch in range(num_epochs):
        model_saved = False
        for i, (train_datum, train_labels, _) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()  # 清空之前的梯度信息
            train_model_inputs = [i.view(-1, 1,input_size).to(dtype=torch.float64) for i in train_datum]
            train_labels = train_labels.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
            # train_labels = train_labels.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
            if input_num == 3 and output_num == 1:
                outputs = model(train_model_inputs[0], train_model_inputs[1],train_model_inputs[2])
            elif input_num == 2 and output_num == 1:
                outputs = model(train_model_inputs[0], train_model_inputs[1])
            elif input_num == 4 and output_num == 1:
                outputs = model(train_model_inputs[0], train_model_inputs[1],train_model_inputs[2],train_model_inputs[3])
            elif input_num == 5 and output_num == 1:
                outputs = model(train_model_inputs[0], train_model_inputs[1],train_model_inputs[2],train_model_inputs[3],train_model_inputs[4])            
            outputs *= 1000
            train_labels *= 1000
            loss = loss_function(outputs, train_labels)

            loss.backward()
            optimizer.step()

            iter += 1
            if iter % 100 == 0:

                true_list, pred_list, loss_list, file_name_list, file_error_list = testAllData(model,test_loader,input_num,output_num,train_mode = True)
                avg_ = np.mean(np.array(loss_list))
                # avg_ = 0
                if loss < min_loss:
                    min_loss = loss
                    model_saved = True
                    torch.save(model, r"model/model.pth")
                # print(f'Epoch: {epoch + 1}/{num_epochs}\t Loss: {loss.item():.4f} test_Loss: {avg_.item():.4f} model_saved:{model_saved}')
                print(f'Epoch: {epoch + 1}/{num_epochs}\t Loss: {loss.item():.4f} test_Loss: {avg_} model_saved:{model_saved}')

    print(time.time()-start_time, f"num_epochs为{num_epochs}")

def test(device, input_num=3,output_num=1):
    model = torch.load(r"model/model.pth").to(device, dtype=torch.float64)
    model.eval()

    # 打印权重信息
    # for name in model.state_dict():
    #     print(name, model.state_dict()[name])
    
    _, test_loader = genDataloader(r"data/test.txt", device, input_size, input_num, output_num, 1.0, False, True)

    iter = 0

    print("\nCALCULATING ACCURACY...\n")
    true_list, pred_list, loss_list, file_name_list, file_error_list = testAllData(model,test_loader,input_num,output_num,train_mode = True)

    n = len(true_list)
    true_array, pred_array = np.array(true_list), np.array(pred_list)

    x = [i for i in range(0, n)]
    x_smooth = np.linspace(np.min(x), np.max(x), n)
    plt.plot(
        x_smooth, true_array[0:n], c="green", marker="*", ms=1, alpha=0.75, label="true"
    )
    plt.plot(
        x_smooth, pred_array[0:n], c="red", marker="o", ms=1, alpha=0.75, label="pred"
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


        
# 测试loader中的所有数据
def testAllData(model,test_loader,input_num,output_num,train_mode = False):
    model.eval()
    loss_function = nn.MSELoss()

    # 打印权重信息
    # for name in model.state_dict():
    #     print(name, model.state_dict()[name])
    iter = 0
    file_name_list = []
    file_error_list = []
    print("\nCALCULATING ACCURACY...\n")
    with torch.no_grad():
        pred_list, loss_list, true_list = [], [], []
        i = 0
        for i, (test_datum, test_labels, test_file_name) in enumerate(test_loader):

            test_model_inputs = [i.view(-1, 1,input_size).to(dtype=torch.float64) for i in test_datum]
            if input_num == 3 and output_num == 1:
                outputs = model(test_model_inputs[0], test_model_inputs[1],test_model_inputs[2])
            elif input_num == 2 and output_num == 1:
                outputs = model(test_model_inputs[0], test_model_inputs[1])
            elif input_num == 4 and output_num == 1:
                outputs = model(test_model_inputs[0], test_model_inputs[1],test_model_inputs[2],test_model_inputs[3])
            elif input_num == 5 and output_num == 1:
                outputs = model(test_model_inputs[0], test_model_inputs[1],test_model_inputs[2],test_model_inputs[3],test_model_inputs[4])                
            outputs *= 100
            test_labels *= 100
            loss = loss_function(outputs[0], test_labels).cpu().detach().numpy()

            test_labels = test_labels.unsqueeze(1)
            # test_labels = test_labels.unsqueeze(1)
            test_labels = test_labels[0].to(dtype=torch.float64).cpu()  # 将目标的形状从[2]变为[2, 1]
            outputs = outputs.cpu()

            true_list.append(test_labels)
            pred_list.append(outputs[0])
            loss_list.append(float(loss))
            iter += 1
            if train_mode:
                # print(test_file_name[0], loss)
                # print(test_datum, outputs/1000, test_labels/1000)
                file_error_list.append([test_file_name[0], test_datum, outputs/1000, test_labels/1000, loss])
                file_name_list.append(test_file_name[0])
            # if iter % 100 == 0:
                
                
                
                # print(f"Iteration: {iter}\t Loss: {loss.item():.4f}")
                # print(f"*************************************************************************")
                # break
        # if train_mode:
        #     losses = [item[-1] for item in file_error_list]
        #     labels = [item[3] for item in file_error_list]
        #     mean_loss = np.mean(losses)
        #     std_loss = np.std(losses)
        #     threshold_multiplier = 1  # 通常可以选择 2 或 3，倍数越大，对异常值的判断越严格
        #     upper_threshold = mean_loss + threshold_multiplier * std_loss
            # for item in file_error_list:
            #     loss = item[-1]
            #     label = item[3]
            #     if loss > upper_threshold:
            #         slope = find_max_slope(item[1][1][0])
            #         print(f"Abnormal loss found in list: {item[-1]} "+ str(item[1][1][0][-1])+ " "+ str(slope)   )
            #         for index,l in enumerate(labels):
            #             if l == label:
            #                 slope = find_max_slope(file_error_list[index][1][1][0])
            #                 print(f"like abnormal data: {file_error_list[index][-1]} "+ str(file_error_list[index][1][1][0][-1]) + " "+str(slope)   )
            #                 break
            #         print(f"*************************************************************************")

                
        # print(f"平均值 {sum(loss_list)/len(loss_list)} 最大loss{max(loss_list)}")
    return true_list, pred_list, loss_list, file_name_list, file_error_list

def find_max_slope(data_list):
    if len(data_list) < 2:
        return 0
    max_slope = 0
    for i in range(len(data_list) - 1):
        for j in range(i + 1, len(data_list)):
            slope = (data_list[j] - data_list[i]) / (j - i)
            if abs(slope) > abs(max_slope):
                max_slope = slope
    return max_slope

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
    # print(torch.cuda.is_available())1
    mode = input("mode:")
    if mode == "1":
        train(device,input_num=3,output_num=1, debug=True , pretrained=False)#3输入1输
        # test(device,input_num=3,output_num=1)
        convert2ONNX(input_num=3,output_num=1,model_name="model.pth")
        subprocess.run(['bash', 'convert.sh','model'])
    elif mode == "2":
        convert2ONNX(input_num=3,output_num=1,model_name="model.pth")
        subprocess.run(['bash', 'convert.sh','model1'])
    
    # train(device,input_num=2,output_num=1, debug=True , pretrained=False)#2输入1输出
    # convert2ONNX(input_num=2,output_num=1,model_name="model.pth")
    # subprocess.run(['bash', 'convert.sh',"model"])
    
    # ONNXRuntime()
    # ort_session = onnxruntime.InferenceSession("model.onnx",providers=["CUDAExecutionProvider"])
# 当前位置、期望目标位置、期望下发速度
