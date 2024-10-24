from matplotlib import pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from LSTM import *
import pandas as pd
from scipy import stats
from fileProcessor import *
from scipy.signal import butter, lfilter
from genTrainData import generateTrainData
from BasicFun import *
import shutil

# 创建实时绘制横纵轴变量
x = []
y = []
predict_y = []
pause = False
delete = False
close = False

def on_delete_press(event):
    global delete
    global close
    global file_name

    if event.key == 'd':  # 检查是否按下了 'd' 键
        delete = True
        print(file_name)
    elif event.key == 'c':
        # 源文件路径
        src_path = '/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/' + file_name
        # 目标文件路径
        dst_path = '/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/need_data/' + file_name
        # 确保目标路径的目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        # 复制文件
        shutil.copy(src_path, dst_path)
        print(f'文件已从 {src_path} 复制到 {dst_path}')
        close = True
        
    elif event.key == '1':  # 训练集
        file_name = file_name.replace(".txt","")
        src_path = '/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/' + file_name         # 源文件路径
        # 目标文件路径
        dst_path = '/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/train/' + file_name
        # 确保目标路径的目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        # 复制文件
        shutil.copy(src_path, dst_path)
        print(f'train 文件已从 {src_path} 复制到 {dst_path}')
        close = True   
    elif event.key == '2': # 测试集
        file_name = file_name.replace(".txt","")
        src_path = '/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/' + file_name         # 源文件路径
        # 目标文件路径
        dst_path = '/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/test/' + file_name
        # 确保目标路径的目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        # 复制文件
        shutil.copy(src_path, dst_path)
        print(f'test 文件已从 {src_path} 复制到 {dst_path}')
        close = True    
        
        
def loadDataSet(file_path="", last_file=False):
    total_data_list = []
    if last_file:
        last_file_name = find_latest_file(
            "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/")+".txt"
        latest_file_path = os.path.join(
            "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train", last_file_name)
        file_path = latest_file_path
        print(last_file_name)
    with open(file_path, "r") as f:
        line = f.readline()  # 读取第一行
        while line:
            line = line.replace(' \n', '').replace(' \r', '')
            data_list = line.split(" ")
            if " " in data_list:
                data_list.remove(" ")
            if '' in data_list:
                data_list.remove('')
            data_list = [float(item) for item in data_list]
            total_data_list.append(data_list)
            line = f.readline()  # 读取下一行

    total_data_array = np.array(total_data_list).T
    return total_data_array


# 可视化数据集
def visualizeDataset(dataset,title_name="", visual_indexs=[], compare_indexs=[], name_list=[]):
    global delete, close
    fig, ax = plt.subplots(figsize=(10, 8))
    calCorrelation(dataset[compare_indexs[0]], dataset[compare_indexs[1]])
    acc_list = calACC(dataset[compare_indexs[0]],
                      dataset[compare_indexs[1]])  # 计算重合度
    colors = ["xkcd:blue",
                "xkcd:grass green",
                "xkcd:goldenrod",
                "xkcd:forest green",
                "xkcd:sky blue",
                "xkcd:light red",
                "xkcd:bright pink",
                "xkcd:lavender",
                "xkcd:ocean blue",
                "xkcd:mud",
                "xkcd:eggplant",
                "xkcd:cyan",
                "xkcd:slate blue",
                "xkcd:peach",
                "xkcd:coral"]
    while not delete and not close:
        plt.title(title_name,y=0,loc='right')

        fig.canvas.mpl_disconnect(
            fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册
        fig.canvas.mpl_connect('key_press_event', on_delete_press)
        x = np.linspace(0, len(dataset[visual_indexs[0]]), len(
            dataset[visual_indexs[0]]))
        for index in visual_indexs:
            ax.plot(x, dataset[index], linewidth=3.0,color=colors[index])
        for item in acc_list:
            plt.plot(item[0], 0, 'o')  # 'o' 表示用圆圈标记数据点
            plt.text(item[0], 0, f'{np.round(item[2],2)}',
                     ha='right', va='bottom')
        plt.legend(name_list)
        plt.waitforbuttonpress()
        plt.pause(0.001)
    ax.cla()  # 清除图形
    plt.close()  # 关闭画图窗口
    delete = False
    close = False
    
    
    
def inference(*data):
    input_num = len(data)
    input_data_list =  [torch.tensor(np.array(d)).to(device).view(-1, 1, input_size) for d in data]
    if input_num == 2:
        outputs = model(input_data_list[0], input_data_list[1])
    elif input_num == 3:
        outputs = model(input_data_list[0], input_data_list[1], input_data_list[2])
    return float(np.round(convertTensor2Numpy(outputs), 3))

# 速度空间下进行搜索
def searchSpeedSpace(data1, data2, data3, model):
    speed_find_list = np.arange(-0.02,0.03,0.01)
    output_list = [[] for i in speed_find_list]
    data2_last = data2[-1]
    for index,speed in enumerate(speed_find_list):
        data2[-1] = data2_last + speed
        predict = inference(data1, data2, data3)
        output_list[index].append(predict)
    return output_list

if __name__ == "__main__":
    # 模型加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(r"model/model.pth").to(device)
    model.eval()


    # # 查看所有数据
    orin_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train/"
    for root, dirs, files in os.walk(orin_path):
        global file_name
        for file_name in files:
            print(file_name)
            motor_info = ["motor_t", "motor_real", "motor_cmd", "motor_expect", "motor_height_gap", "motor_speed_gap","orin_cmd","predict_expect","real_gap","cmd_gap","expect_gap"]
            # dataset = loadDataSet(orin_path+file_name, True)
            p = orin_path+file_name
            # p = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train/robokit_2024-10-23_17-27-40.0.log.txt"
            # p = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train/robokit_2024-09-27_15-48-20.2.log.txt"
            # p = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/test/robokit_2024-10-17_16-16-06.1.log.txt"
            # p = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/test/robokit_2024-10-17_16-36-02.0.log.txt"
            # p = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/test/robokit_2024-10-17_16-45-33.0.log.txt"
            dataset = loadDataSet(p, False)
            visual_indexs = [1,2,3,4,5,7]
            visualizeDataset(dataset, title_name = file_name, visual_indexs=visual_indexs, compare_indexs=[1,3], name_list=[motor_info[i] for i in visual_indexs])
            # print("****************************************************")
            
            
            # motor_info = ["-0.02", "-0.01", "0.0", "0.01", "0.02", "motor_speed_gap","orin_cmd","predict_expect","real_gap","cmd_gap","expect_gap"]

            # data_list = generateTrainData(
            #     p,
            #     p,
            #     1.4,
            #     interp_interval=0.05,
            #     repetition_rate=0.96,
            #     need_input_data=["real_temp",
            #                     "cmd_temp", "height_gap_temp"],
            #     need_output_data=["real_label"],
            #     interp=True,
            #     difference=False,
            #     save_file=False
            # )
            # dataset = []
            # speed_ = [[],[],[],[],[],[]]
            # for data_ in data_list:
            #     speed_list = searchSpeedSpace([float(i) for i in data_[1].split(" ")],[float(i) for i in data_[2].split(" ")],[float(i) for i in data_[3].split(" ")],model)
            #     speed_[0].append(speed_list[0])
            #     speed_[1].append(speed_list[1])
            #     speed_[2].append(speed_list[2])
            #     speed_[3].append(speed_list[3])
            #     speed_[4].append(speed_list[4])
            #     speed_[5].append(float(data_[0]))
                
            # dataset.append([i[0] for i in speed_[0]])
            # dataset.append([i[0] for i in speed_[1]])
            # dataset.append([i[0] for i in speed_[2]])
            # dataset.append([i[0] for i in speed_[3]])
            # dataset.append([i[0] for i in speed_[4]])
            # dataset.append([i for i in speed_[5]])
            # # visual_indexs = [0,1,2,3,4,5]
            # visual_indexs = [0,2,4,5]
            # visualizeDataset(dataset, title_name = file_name, visual_indexs=visual_indexs, compare_indexs=[0,1], name_list=[motor_info[i] for i in visual_indexs])
            # print("****************************************************")

