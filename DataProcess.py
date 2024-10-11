import torch
import numpy as np
import random

def randomSortData(data_path, random_seed, if_random):
    total_data_list = []
    with open(data_path, "r") as f:
        line = f.readline()  # 读取第一行
        while line:
            data_list = line.split(" ")
            data_list.pop()
            data_l = [float(item) for item in data_list[1:]]
            data_l.insert(0, data_list[0])
            total_data_list.append(data_l)
            line = f.readline()  # 读取下一行
                
    if if_random:
        random.seed(random_seed)
        random.shuffle(total_data_list)
    return total_data_list


def loadData(data_path, device, input_size,  random_state, if_random , if_torch):
    """
    加载和处理数据。

    参数:
    - data_path: 数据文件的路径。
    - device: 数据处理所使用的设备（如CPU或GPU）。
    - random_state: 用于划分训练集和测试集的随机状态参数。
    - if_random: 是否对数据进行随机排序。

    返回:
    - train_list: 包含训练集real数据、cmd数据和标签的列表。
    - test_list: 包含测试集real数据、cmd数据和标签的列表。
    """
    total_data_list = randomSortData(data_path, 64, if_random)
    total_file_name = []
    total_label = []
    total_label2 = []
    total_real = []
    total_cmd = []
    total_postion = []
    for data_list in total_data_list:

        file_name = data_list.pop(0)
        label = data_list[0]
        label2 = data_list[1]
        len_data_list = len(data_list)
        x1_list = data_list[2 : int((len_data_list - 2) / 3) + 2]
        x2_list = data_list[int((len_data_list - 2) / 3) + 2 :int(2*(len_data_list - 2) / 3) + 2 ]
        x3_list = data_list[int(2*(len_data_list - 2) / 3) + 2:]
        while len(x1_list) > input_size:
            x1_list.pop(0)
        while len(x2_list) > input_size:
            x2_list.pop(0)
        while len(x3_list) > input_size:
            x3_list.pop(0)
        if len(x1_list) == input_size and len(x2_list) == input_size and len(x3_list) == input_size:
            total_file_name.append(file_name)
            total_label.append(label)
            total_label2.append(label2)
            total_real.append(x1_list)
            total_cmd.append(x2_list)
            total_postion.append(x3_list)
    total_label = np.array(total_label)
    total_label2 = np.array(total_label2)
    total_real = np.array(total_real)
    total_cmd = np.array(total_cmd)
    total_postion = np.array(total_postion)
    len_data = int(len(total_real) * (1 - random_state))
    if if_torch:
        total_real = torch.tensor(total_real).to(device)
        total_cmd = torch.tensor(total_cmd).to(device)
        total_postion = torch.tensor(total_postion).to(device)
        total_label = torch.tensor(total_label).to(device)
        total_label2 = torch.tensor(total_label2).to(device)
    # 分割数据集
    train_list = [total_real[0:len_data], total_cmd[0:len_data], total_postion[0:len_data], total_label[0:len_data],total_label2[0:len_data],total_file_name[0:len_data]]
    test_list = [total_real[len_data:-1], total_cmd[len_data:-1], total_postion[len_data:-1],total_label[len_data:-1],total_label2[len_data:-1],total_file_name[len_data:-1]]

    return train_list, test_list



def loadData2(data_path, device, input_size,  random_state, if_random , if_torch):
    """
    加载和处理数据。

    参数:
    - data_path: 数据文件的路径。
    - device: 数据处理所使用的设备（如CPU或GPU）。
    - random_state: 用于划分训练集和测试集的随机状态参数。
    - if_random: 是否对数据进行随机排序。

    返回:
    - train_list: 包含训练集real数据、cmd数据和标签的列表。
    - test_list: 包含测试集real数据、cmd数据和标签的列表。
    """
    total_data_list = randomSortData(data_path, 64, if_random)
    total_file_name = []
    total_label = []
    total_label2 = []
    total_real = []
    total_cmd = []
    total_postion = []
    for data_list in total_data_list:

        file_name = data_list.pop(0)
        label = data_list[0]
        len_data_list = len(data_list)
        x1_list = data_list[1 : int((len_data_list - 1) / 2) + 1]
        x2_list = data_list[int((len_data_list - 1) / 2) + 1 : ]
        while len(x1_list) > input_size:
            x1_list.pop(0)
        while len(x2_list) > input_size:
            x2_list.pop(0)

        if len(x1_list) == input_size and len(x2_list) == input_size :
            total_file_name.append(file_name)
            total_label.append(label)
            total_real.append(x1_list)
            total_cmd.append(x2_list)
    total_label = np.array(total_label)
    total_label2 = np.array(total_label2)
    total_real = np.array(total_real)
    total_cmd = np.array(total_cmd)
    total_postion = np.array(total_postion)
    len_data = int(len(total_real) * (1 - random_state))
    if if_torch:
        total_real = torch.tensor(total_real).to(device)
        total_cmd = torch.tensor(total_cmd).to(device)
        total_postion = torch.tensor(total_postion).to(device)
        total_label = torch.tensor(total_label).to(device)
        total_label2 = torch.tensor(total_label2).to(device)
    # 分割数据集
    train_list = [total_real[0:len_data], total_cmd[0:len_data],  total_label[0:len_data],total_file_name[0:len_data]]
    test_list = [total_real[len_data:-1], total_cmd[len_data:-1],total_label[len_data:-1],total_file_name[len_data:-1]]

    return train_list, test_list


def loadData3(data_path, device, input_size,  random_state, if_random , if_torch):
    """
    加载和处理数据。

    参数:
    - data_path: 数据文件的路径。
    - device: 数据处理所使用的设备（如CPU或GPU）。
    - random_state: 用于划分训练集和测试集的随机状态参数。
    - if_random: 是否对数据进行随机排序。

    返回:
    - train_list: 包含训练集real数据、cmd数据和标签的列表。
    - test_list: 包含测试集real数据、cmd数据和标签的列表。
    """
    total_data_list = randomSortData(data_path, 64, if_random)
    total_file_name = []
    total_label = []
    total_label2 = []
    total_real = []
    total_cmd = []
    total_postion = []
    for data_list in total_data_list:

        file_name = data_list.pop(0)
        label = data_list[0]
        len_data_list = len(data_list)
        x1_list = data_list[1 : int((len_data_list - 1) / 3) + 1]
        x2_list = data_list[int((len_data_list - 1) / 3) + 1 :int(2*(len_data_list - 1) / 3) + 1 ]
        x3_list = data_list[int(2*(len_data_list - 1) / 3) + 1:]
        while len(x1_list) > input_size:
            x1_list.pop(0)
        while len(x2_list) > input_size:
            x2_list.pop(0)
        while len(x3_list) > input_size:
            x3_list.pop(0)
        if len(x1_list) == input_size and len(x2_list) == input_size and len(x3_list) == input_size:
            total_file_name.append(file_name)
            total_label.append(label)
            total_real.append(x1_list)
            total_cmd.append(x2_list)
            total_postion.append(x3_list)
    total_label = np.array(total_label)
    total_real = np.array(total_real)
    total_cmd = np.array(total_cmd)
    total_postion = np.array(total_postion)
    len_data = int(len(total_real) * (1 - random_state))
    if if_torch:
        total_real = torch.tensor(total_real).to(device)
        total_cmd = torch.tensor(total_cmd).to(device)
        total_postion = torch.tensor(total_postion).to(device)
        total_label = torch.tensor(total_label).to(device)
    # 分割数据集
    train_list = [total_real[0:len_data], total_cmd[0:len_data], total_postion[0:len_data], total_label[0:len_data],total_file_name[0:len_data]]
    test_list = [total_real[len_data:-1], total_cmd[len_data:-1], total_postion[len_data:-1],total_label[len_data:-1],total_file_name[len_data:-1]]

    return train_list, test_list


# def loadData(data_path, device, input_size, random_state, data_num = 3 , label_num = 1,   if_random = True , if_torch = True):
#     """
#     加载和处理数据。

#     参数:
#     - data_path: 数据文件的路径。
#     - device: 数据处理所使用的设备（如CPU或GPU）。
#     - random_state: 用于划分训练集和测试集的随机状态参数。
#     - if_random: 是否对数据进行随机排序。

#     返回:
#     - train_list: 包含训练集real数据、cmd数据和标签的列表。
#     - test_list: 包含测试集real数据、cmd数据和标签的列表。
#     """
#     total_data_list = randomSortData(data_path, 64, if_random)
#     total_file_name = []
#     total_label = []
#     total_label2 = []
#     total_real = []
#     total_cmd = []
#     total_postion = []
#     for data_list in total_data_list:
#         file_name = data_list.pop(0)
#         labels = [[data_list[i]] for i in range(label_num)]
        
#         len_data_list = len(data_list)
        
#         x1_list = data_list[label_num : int((len_data_list - label_num) / data_num) + label_num]
#         x2_list = data_list[int((len_data_list - 2) / 3) + 2 :int(2*(len_data_list - 2) / 3) + 2 ]
#         x3_list = data_list[int(2*(len_data_list - 2) / 3) + 2:]
#         while len(x1_list) > input_size:
#             x1_list.pop(0)
#         while len(x2_list) > input_size:
#             x2_list.pop(0)
#         while len(x3_list) > input_size:
#             x3_list.pop(0)
#         if len(x1_list) == input_size and len(x2_list) == input_size and len(x3_list) == input_size:
#             total_file_name.append(file_name)
#             total_label.append(label)
#             total_label2.append(label2)
#             total_real.append(x1_list)
#             total_cmd.append(x2_list)
#             total_postion.append(x3_list)
#     total_label = np.array(total_label)
#     total_label2 = np.array(total_label2)
#     total_real = np.array(total_real)
#     total_cmd = np.array(total_cmd)
#     total_postion = np.array(total_postion)
#     len_data = int(len(total_real) * (1 - random_state))
#     if if_torch:
#         total_real = torch.tensor(total_real).to(device)
#         total_cmd = torch.tensor(total_cmd).to(device)
#         total_postion = torch.tensor(total_postion).to(device)
#         total_label = torch.tensor(total_label).to(device)
#         total_label2 = torch.tensor(total_label2).to(device)
#     # 分割数据集
#     train_list = [total_real[0:len_data], total_cmd[0:len_data], total_postion[0:len_data], total_label[0:len_data],total_label2[0:len_data],total_file_name[0:len_data]]
#     test_list = [total_real[len_data:-1], total_cmd[len_data:-1], total_postion[len_data:-1],total_label[len_data:-1],total_label2[len_data:-1],total_file_name[len_data:-1]]

#     return train_list, test_list