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
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator

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
def visualizeDataset(dataset , title_name="", visual_indexs=[], compare_indexs=[], name_list=[], nrow = 0):
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

# 可视化数据集
def visualizeDataset2(title_name, fig, axs):
    global delete, close
    while not delete and not close:
        plt.title(title_name,y=0,loc='right')
        fig.canvas.mpl_disconnect(
            fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册
        fig.canvas.mpl_connect('key_press_event', on_delete_press)
        plt.waitforbuttonpress()
        plt.pause(0.001)
    # plt.show()

    plt.close()  # 关闭画图窗口
    delete = False
    close = False
    
def findAllHeight(dataset):
    height_list = []
    expect_speed = dataset[3]
    height_gap = dataset[4]
    start_record = False
    dataset2 = [[] for i in range(len(dataset))]
    for index in range(len(expect_speed)-1):
        if not start_record:
            if expect_speed[index] == 0 and expect_speed[index+1] != 0:
                height = round(height_gap[index+1],1)
                if  height not in height_list:
                    height_list.append(height)
    print(height_list)
    return height_list
def findHeightData(dataset,expect_height):
    real_speed = dataset[1]
    expect_speed = dataset[3]
    height_gap = dataset[4]
    start_record = False
    dataset2 = [[] for i in range(len(dataset))]
    for index in range(len(expect_speed)-1):
        if not start_record:
            if expect_speed[index] == 0 and expect_speed[index+1] != 0:
                if abs(round(height_gap[index+1],1)-expect_height)< 0.005: 
                    for j in range(len(dataset)):
                        dataset2[j].append(dataset[j][index])
                    start_record = True
        else:
            for j in range(len(dataset)):
                dataset2[j].append(dataset[j][index])   
            if expect_speed[index] == 0 and abs(real_speed[index]) < 0.02:
                start_record = False
                for i in range(0,30,1):
                    for j in range(len(dataset)):
                        dataset2[j].append(0)   
    return dataset2
                
# 可视化数据集
def createAXPlot(dataset , visual_indexs=[],  name_list=[],expect_height=0.1, ax = None):
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
                "xkcd:coral",
                "xkcd:blue",
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
                ]
    ax.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  # 设置横坐标最多10个刻度

    x = np.linspace(0, len(dataset[visual_indexs[0]]), len(
        dataset[visual_indexs[0]]))    
    ax.set_title(str(expect_height))
    
    # dataset = findHeightData(dataset,expect_height)
    x = np.linspace(0, len(dataset[visual_indexs[0]]), len(
        dataset[visual_indexs[0]]))    
    for index in visual_indexs:
        ax.plot(x, dataset[index], linewidth=3.0,color=colors[index])
    
    ax.legend(name_list)
    return ax

    
    
def inference(*data):
    input_num = len(data[0])

    input_data_list =  [torch.tensor(np.array(d)).to(device).view(-1, 1, input_size) for d in data[0]]
    if input_num == 2:
        outputs = model(input_data_list[0], input_data_list[1])
    elif input_num == 3:
        outputs = model(input_data_list[0], input_data_list[1], input_data_list[2])
    elif input_num == 4:
        outputs = model(input_data_list[0], input_data_list[1], input_data_list[2], input_data_list[3])
    elif input_num == 5:
        outputs = model(input_data_list[0], input_data_list[1], input_data_list[2], input_data_list[3],input_data_list[4])    
    # return float(np.round(convertTensor2Numpy(outputs), 3))
    return [float(np.round(convertTensor2Numpy(i), 3)) for i in outputs[0]]

# 速度空间下进行搜索
def searchSpeedSpace(data1, data2, data3, model):
    speed_find_list = np.arange(-0.02,0.03,0.01)
    output_list = [[] for i in speed_find_list]
    data2_last = data2[-1]
    for index,speed in enumerate(speed_find_list):
        data2[-1] = data2_last + speed
        predict = inference([data1, data2, data3])
        output_list[index].append(predict)
    return output_list

def multiOutput(data1, data2, data3, model):
    output_list = [[],[],[]]
    predict = inference([data1, data2, data3])
    for i in range(3):
        output_list[i].append(predict[i])
    return output_list


def continusExpectPredict(path,input_num,output_num):
    data_list = generateTrainData(
        path,
        path,
        1.4,
        interp_interval=0.05,
        repetition_rate=0.96,
        need_input_data = ["expect_temp","height_gap_temp","real_temp"],
        need_output_data = ["expect_label"],
        interp=True, 
        difference=False,
        save_file=False
    )
    speed_ = [[],[]]
    dataset = []

    for data_ in data_list:
        speed_list = inference([[float(i) for i in data_[j].split(" ")] for j in range(1,input_num+1)])
        # speed_list = inference([float(i) for i in data_[1].split(" ")],[float(i) for i in data_[2].split(" ")],[float(i) for i in data_[3].split(" ")])
        speed_[0].append(speed_list)
        speed_[1].append(float(data_[0]))
    dataset.append([i for i in speed_[0]])
    dataset.append([i for i in speed_[1]])
    return dataset

        
def continusSpeadSpaceSearch(path):
    data_list = generateTrainData(
        path,
        path,
        1.5,
        interp_interval=0.05,
        repetition_rate=0.96,
        need_input_data=["real_temp", "cmd_temp", "height_gap_temp"],
        need_output_data=["real_label"],
        interp=True, 
        difference=False,
        save_file=False
    )
    dataset = []
    speed_ = [[],[],[],[],[],[]]
    # for data_ in data_list:
    #     speed_list = searchSpeedSpace([float(i) for i in data_[1].split(" ")],[float(i) for i in data_[2].split(" ")],[float(i) for i in data_[3].split(" ")],model)
    #     speed_[0].append(speed_list[0])
    #     speed_[1].append(speed_list[1])
    #     speed_[2].append(speed_list[2])
    #     speed_[3].append(speed_list[3])
    #     speed_[4].append(speed_list[4])
    #     speed_[5].append(float(data_[0]))
    for data_ in data_list:
        speed_list = multiOutput([float(i) for i in data_[1].split(" ")],[float(i) for i in data_[2].split(" ")],[float(i) for i in data_[3].split(" ")],model)
        speed_[0].append(speed_list[0])
        speed_[1].append(speed_list[1])
        speed_[2].append(speed_list[2])
    dataset.append([i[0] for i in speed_[0]])
    dataset.append([i[0] for i in speed_[1]])
    dataset.append([i[0] for i in speed_[2]])
    # visual_indexs = [0,1,2,3,4,5]
    return dataset


# 读取recursive_search
def continusRecursiveSearch(file_path):
    with open(file_path, "r") as f:
        data = f.read()  # 读取第一行
        # 分割数据
        parts = data.split('[*********************************Default argument]')

        expect_speed = []
        speed_list = [[] for i in range(-30,30,5)]
        # 存储recursive_search行的列表
        recursive_search_lines = []
        len_list = []
        # 遍历每部分，提取recursive_search行
        for part in parts:
            if part:  # 确保部分不为空
                lines = part.strip().split('\n')
                for line in lines:
                    if '[recursive_search]' in line:
                        recursive_search_lines.append(line)
                tmp_speed_list = [0 for i in range(-30,30,5)]
                has_expect_append = False
                for line in recursive_search_lines:
                    line_list = line.split("|")
                    speed_index = int((float(line_list[5]) + 0.005)/0.005)
                    tmp_speed_list[speed_index] = line_list[10]
                    if not has_expect_append:
                        expect_speed.append(float(f"{float(line_list[9]):.3f}"))
                        has_expect_append = True
                for index,d in enumerate(tmp_speed_list):
                    speed_list[index].append(float(f"{float(d):.3f}"))
                # 打印提取的recursive_search行
                len_list.append(len(recursive_search_lines))
                recursive_search_lines = []
        while len(speed_list[-1]) > len(expect_speed):
            expect_speed.append(0)
        while len(speed_list[-1]) < len(expect_speed):
            expect_speed.pop(0)
        speed_list.append(expect_speed)
    return speed_list

        
if __name__ == "__main__":
    # 模型加载
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = torch.load(r"model/model.pth").to(device)
    # model.eval()

    # mode = int(input("mode:"))
    mode = 3
    # single_data = input("single data 0 or 1:")
    single_data = 1
    
    if mode == 0:
        if not single_data:
            orin_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train/"
            for root, dirs, files in os.walk(orin_path):
                global file_name
                for file_name in tqdm(files):
                    print(file_name)
                    p = orin_path+file_name
                    dataset = loadDataSet(p, last_file = False)
                    motor_info = ["motor_t", "motor_real", "motor_cmd", "motor_expect", "motor_height_gap", "motor_speed_gap","orin_cmd","predict_expect","real_gap","cmd_gap","expect_gap"]
                    visual_indexs = [1,2,3,4,5,7]
                    visualizeDataset(dataset, title_name = file_name, visual_indexs=visual_indexs, compare_indexs=[1,3], name_list=[motor_info[i] for i in visual_indexs])
        else:
            p = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train/robokit_2024-10-23_17-27-40.0.log.txt"
            dataset = loadDataSet(p, last_file = False)       
            motor_info = ["motor_t", "motor_real", "motor_cmd", "motor_expect", "motor_height_gap", "motor_speed_gap","orin_cmd","predict_expect","real_gap","cmd_gap","expect_gap"]
            visual_indexs = [1,2,3,4,5,7]
            visualizeDataset(dataset, title_name = file_name, visual_indexs=visual_indexs, compare_indexs=[1,3], name_list=[motor_info[i] for i in visual_indexs])
    # 查看在速度空间下的推理结果
    elif mode == 1:
        motor_info = ["-0.02", "-0.01", "0.0", "0.01", "0.02"]
        p = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train/robokit_2024-10-23_17-27-40.0.log.txt"
        visual_indexs = [0,2,4,5]
        dataset = continusSpeadSpaceSearch(p)
        visualizeDataset(dataset, title_name = file_name, visual_indexs=visual_indexs, compare_indexs=[0,1], name_list=[motor_info[i] for i in visual_indexs])


    elif mode == 2:
        file_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/train/robokit_2024-09-27_15-48-20.2.log"
        file_path.split()
        motor_info = [str(i/1000) for i in range(-20,20,2)]
        motor_info.append("expect")
        visual_indexs=[0,4,8,12,16,-1]
        
        speed_list = continusRecursiveSearch(file_path)
        visualizeDataset(speed_list,title_name="1",visual_indexs=visual_indexs,name_list = [motor_info[i] for i in visual_indexs])

    elif mode == 3:
        orin_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/train/"
        for root, dirs, files in os.walk(orin_path):
            for file_name in tqdm(files):
                fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=False, sharey=False)
                plt.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                file_path = orin_path+file_name

                # file_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/train/robokit_2024-11-12_17-30-19.4.log"
                # p = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train/robokit_2024-10-23_17-27-40.0.log.txt"
                try:

                    dataset = loadDataSet(file_path.replace("data_set","compared_data")+".txt", last_file = False)       
                    motor_info = ["motor_t", "motor_real", "motor_cmd", "motor_expect", "motor_height_gap", "motor_speed_gap","orin_cmd","predict_expect","real_gap","cmd_gap","expect_gap"]
                    visual_indexs = [1,2,3,7]
                    # visualizeDataset(dataset, title_name = file_name, visual_indexs=visual_indexs, compare_indexs=[1,3], name_list=[motor_info[i] for i in visual_indexs])
                    ax1 = createAXPlot(dataset,visual_indexs,[motor_info[i] for i in visual_indexs],0.1,ax1) # 绘制模型在速度空间下的搜索结果
                    # ax2 = createAXPlot(dataset,visual_indexs,[motor_info[i] for i in visual_indexs],0.2,ax2) # 绘制模型在速度空间下的搜索结果
                    # ax3 = createAXPlot(dataset,visual_indexs,[motor_info[i] for i in visual_indexs],0.3,ax3) # 绘制模型在速度空间下的搜索结果

                    name = file_path.split("/")[-1]
                    motor_info = [str(i/1000) for i in range(-20,20,2)]
                    # motor_info.append("expect")
                    motor_info.append("expect")
                    motor_info.append("real")
                    visual_indexs=[0,7,14,-1]
                    # visual_indexs=[8,-1]
                    temp = np.append(dataset[7],0)
                    dd = np.append(dataset[1],0)
                
                    recursiveSearchlist = continusRecursiveSearch(file_path) # 读取log中速度空间搜索下的结果
                    recursiveSearchlist.append(temp)
                    # recursiveSearchlist.append(temp2)
                    # ax2 = createAXPlot(recursiveSearchlist,visual_indexs,[motor_info[i] for i in visual_indexs],ax = ax2) # 绘制模型在速度空间下的搜索结果
                    # ax3 = createAXPlot(recursiveSearchlist,visual_indexs,[motor_info[i] for i in visual_indexs],ax3) # 绘制模型在速度空间下的搜索结果
                    

                    # motor_info = ["-0.02", "-0.01", "0.0", "0.01", "0.02","label"]
                    # visual_indexs = [0,2,4,5]
                    
                    # motor_info = ["0.3", "0.4", "0.5"]
                    # visual_indexs = [0,1,2]
                    # SpeadSpaceSearchList = continusSpeadSpaceSearch(file_path.replace("data_set","compared_data")+".txt") # 使用模型在速度空间下进行搜索
                    # ax3 = createAXPlot(SpeadSpaceSearchList,visual_indexs,[motor_info[i] for i in visual_indexs],ax3) # 绘制模型在速度空间下的搜索结果
                    
                
                    # motor_info = ["predict","expect"]
                    # visual_indexs = [0,1]
                    # SpeadSpaceSearchList = continusExpectPredict(file_path.replace("data_set","compared_data")+".txt",3,1) # 使用模型在速度空间下进行搜索
                    # ax3 = createAXPlot(SpeadSpaceSearchList,visual_indexs,motor_info,ax3) # 绘制模型在速度空间下的搜索结果
                    visualizeDataset2(name , fig, (ax1,ax2,ax3))
                except Exception as e:
                    print(e)
                    plt.close()  # 关闭画图窗口
    elif mode == 4:
        orin_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/train/"
        for root, dirs, files in os.walk(orin_path):
            for file_name in tqdm(files):
                try:
                    file_path = orin_path+file_name
                    dataset = loadDataSet(file_path.replace("data_set","compared_data")+".txt", last_file = False)       

                    search_heights = findAllHeight(dataset)
                    search_heights = [-0.2,-0.1,0.2,0.1]
                    row = len(search_heights) //2 
                    fig, axs = plt.subplots(row, 2, figsize=(8, 8), sharex=False, sharey=False)
                    plt.grid(True, color='gray', linestyle='--', linewidth=1, alpha=0.5)

                    name = file_path.split("/")[-1]

                    motor_info = ["motor_t", "motor_real", "motor_cmd", "motor_expect", "motor_height_gap", "motor_speed_gap","orin_cmd","predict_expect","real_gap","cmd_gap","expect_gap"]
                    visual_indexs = [1,2,3,7]
                    # search_heights = findAllHeight(dataset)
                    index = 0
                    for ax in axs:
                        createAXPlot(dataset,visual_indexs,[motor_info[i] for i in visual_indexs],search_heights[index],ax[0]) # 绘制模型在速度空间下的搜索结果
                        createAXPlot(dataset,visual_indexs,[motor_info[i] for i in visual_indexs],search_heights[index+1],ax[1]) # 绘制模型在速度空间下的搜索结果
                        index += 2
                    visualizeDataset2(name , fig, axs)
                except Exception as e:
                    pass