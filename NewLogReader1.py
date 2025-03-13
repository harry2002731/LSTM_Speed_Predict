from matplotlib import pyplot as plt
import numpy as np
import torch
from copy import deepcopy
import pandas as pd
from scipy import stats
from scipy.signal import butter, lfilter

import shutil
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import RectangleSelector
import os
# 创建实时绘制横纵轴变量
x = []
y = []
predict_y = []
pause = False
delete = False
close = False
selected = []
ax = None

def extract_lines(input_file, output_file, start_line, end_line):
    """
    从源文件中提取指定行并保存到目标文件。
    
    :param input_file: 源文件路径。
    :param output_file: 目标文件路径。
    :param start_line: 提取的起始行号（含）。
    :param end_line: 提取的结束行号（含）。
    """
    try:
        # 打开源文件并读取所有行
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 提取指定行
        extracted_lines = lines[start_line:end_line + 1]
        
        # 打开目标文件并写入提取的行
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(extracted_lines)
        
        print(f"已提取第 {start_line + 1} 行到第 {end_line + 1} 行（从 1 开始计数），并保存到 {output_file}。")
    except Exception as e:
        print(f"发生错误：{e}")

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
    elif event.key == 's':  # 检查是否按下了 's' 键
        start_index = int(selected[0][0])
        end_index = int(selected[-1][0])
        file_path = r"C:\Users\HarryWen\Desktop\test1\\robokit_2025-02-28_10-19-06.1.log.txt"
        file_name = r"C:\Users\HarryWen\Desktop\test1\\twst.txt"
        extract_lines(file_path, file_name, start_index, end_index)

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
    global delete, close,x,y,ax
    fig, ax = plt.subplots(figsize=(10, 8))
    
    selector = RectangleSelector(
    ax,
    onselect,
    useblit=True,
    button=[1],  # 左键触发
    minspanx=5,
    minspany=5,
    spancoords='pixels',
    interactive=True
    )

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
        y = dataset[1]
        for index in visual_indexs:
            ax.plot(x, dataset[index], linewidth=3.0,color=colors[index])
            current_xlim = ax.get_xlim()
            x_min = current_xlim[0]
            x_max = current_xlim[1]
            print(f"Current X min: {x_min}")
            print(f"Current X max: {x_max}")

        plt.legend(name_list)
        plt.waitforbuttonpress()
        plt.pause(0.001)
    ax.cla()  # 清除图形
    plt.close()  # 关闭画图窗口
    delete = False
    close = False

def onselect(eclick, erelease):
    global selected
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    if len(selected) > 0:
        update_plot('xkcd:grass green')
        print(selected)
    # 获取框选区域内的数据点
    selected = [
        (xi, yi) for xi, yi in zip(x, y)
        if (x1 < xi < x2) and (y1 < yi < y2)
    ]
    # 更新图表以突出选中的点
    update_plot('red')

def update_plot(color='red'):
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    # plt.cla()  # 清除当前图
    plt.scatter(*zip(*selected), s=20 ,color=color)  # 选中的点红色显示
    print(current_xlim,current_ylim)
    plt.xlim(current_xlim)

    plt.ylim(current_ylim)
    plt.draw()
       
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
        
if __name__ == "__main__":

    mode = 0
    single_data = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = torch.load(r"model/model.pth").to(device)
    model.eval()
    



 
    data1 = [[
                [0.08699999749660492,0.08699999749660492,0.08799999952316284,0.08900000154972076,0.09000000357627869,0.09000000357627869,0.09000000357627869,0.09000000357627869,0.09000000357627869,0.09000000357627869,0.09000000357627869,0.09000000357627869,0.09000000357627869,0.09000000357627869,0.09000000357627869,0.08900000154972076,0.08799999952316284,0.08799999952316284,0.08699999749660492,0.08699999749660492,0.08699999749660492,0.0860000029206276,0.0860000029206276,0.0860000029206276,0.08500000089406967,0.08500000089406967,0.08399999886751175,0.08399999886751175,0.08299999684095383,0.08299999684095383,0.08299999684095383,0.0820000022649765,0.0820000022649765,0.0820000022649765,0.08100000023841858,0.08100000023841858,0.07999999821186066,0.07900000363588333,0.07900000363588333,0.07800000160932541,0.07699999958276749,0.07599999755620956,0.07500000298023224,0.07400000095367432,0.0729999989271164,0.07199999690055847,0.07100000232458115,0.07000000029802322,0.0689999982714653,0.06800000369548798],
                [0.08699999749660492,0.0860000029206276,0.08500000089406967,0.08500000089406967,0.08399999886751175,0.08299999684095383,0.08299999684095383,0.0820000022649765,0.0820000022649765,0.0820000022649765,0.08100000023841858,0.08100000023841858,0.08100000023841858,0.07999999821186066,0.07999999821186066,0.07900000363588333,0.07900000363588333,0.07900000363588333,0.07800000160932541,0.07800000160932541,0.07800000160932541,0.07699999958276749,0.07699999958276749,0.07599999755620956,0.07599999755620956,0.07500000298023224,0.0729999989271164,0.07199999690055847,0.07199999690055847,0.07100000232458115,0.07100000232458115,0.07000000029802322,0.07000000029802322,0.07000000029802322,0.0689999982714653,0.0689999982714653,0.06800000369548798,0.06700000166893005,0.06599999964237213,0.06499999761581421,0.06499999761581421,0.06400000303983688,0.06300000101327896,0.06199999898672104,0.06199999898672104,0.06199999898672104,0.061000000685453415,0.061000000685453415,0.061000000685453415,0.061000000685453415],
                [0.11400000005960464,0.11100000143051147,0.10999999940395355,0.1080000028014183,0.10599999874830246,0.10400000214576721,0.10300000011920929,0.10100000351667404,0.10000000149011612,0.09799999743700027,0.09600000083446503,0.0949999988079071,0.09300000220537186,0.09099999815225601,0.08900000154972076,0.08699999749660492,0.08500000089406967,0.08299999684095383,0.08100000023841858,0.07900000363588333,0.07800000160932541,0.07599999755620956,0.07400000095367432,0.0729999989271164,0.07199999690055847,0.07100000232458115,0.0689999982714653,0.06700000166893005,0.06499999761581421,0.06400000303983688,0.06199999898672104,0.061000000685453415,0.05900000035762787,0.057999998331069946,0.0560000017285347,0.054999999701976776,0.05299999937415123,0.052000001072883606,0.05000000074505806,0.04800000041723251,0.04699999839067459,0.04600000008940697,0.04399999976158142,0.041999999433755875,0.04100000113248825,0.03999999910593033,0.039000000804662704,0.03700000047683716,0.035999998450279236,0.03500000014901161]]]

    input_data_list =  [torch.tensor(np.array(d)).to(device).view(-1, 1, 20) for d in data1[0]]
    outputs = model(input_data_list)
    print(outputs)
        
    # if mode == 0:
    #     if not single_data:
    #         orin_path = r"C:\Users\HarryWen\Desktop\test1\\"
    #         for root, dirs, files in os.walk(orin_path):
    #             global file_name
    #             for file_name in tqdm(files):
    #                 print(file_name)
    #                 p = orin_path+file_name
    #                 dataset = loadDataSet(p, last_file = False)
    #                 motor_info = ["motor_t", "motor_real", "motor_cmd", "motor_expect", "motor_height_gap", "motor_speed_gap","orin_cmd","predict_expect","weight","predict_weight_index"]
    #                 visual_indexs = [1,2]
    #                 visualizeDataset(dataset, title_name = file_name, visual_indexs=visual_indexs, compare_indexs=[1,3], name_list=[motor_info[i] for i in visual_indexs])
    #     else:
    #         p = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train/robokit_2024-10-23_17-27-40.0.log.txt"
    #         dataset = loadDataSet(p, last_file = False)       
    #         motor_info = ["motor_t", "motor_real", "motor_cmd", "motor_expect", "motor_height_gap", "motor_speed_gap","orin_cmd","predict_expect","real_gap","cmd_gap","expect_gap"]
    #         visual_indexs = [1,2,3,4,5,7]
    #         visualizeDataset(dataset, title_name = file_name, visual_indexs=visual_indexs, compare_indexs=[1,3], name_list=[motor_info[i] for i in visual_indexs])
    # 查看在速度空间下的推理结果
  