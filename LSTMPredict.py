from matplotlib import pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from LSTM import *
import os
import pandas as pd
import shutil
from scipy import stats

# 创建实时绘制横纵轴变量
x = []
y = []
predict_y = []
pause = False
delete = False
close = False

# def on_key_press(event):
#     global pause
#     if pause:
#         pause = False
#     else:
#         pause = True

def on_delete_press(event):
    global delete
    global close

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

def find_latest_file(directory):
    # 用来存储最新文件的变量，初始化为None
    latest_file = None
    latest_time = 0
    last_file_name = ""

    # 遍历目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 确保是文件而不是文件夹
        if os.path.isfile(file_path):
            # 获取文件的最后修改时间
            file_modified_time = os.path.getmtime(file_path)

            # 检查是否是最新的文件
            if file_modified_time > latest_time:
                latest_time = file_modified_time
                latest_file = file_path
                last_file_name = filename

    return last_file_name


def loadDataSet(file_path="", last_file=False, delete_zero = False):
    input_size = 50
    total_data_list = []
    motor_real = []
    motor_cmd = []
    motro_expect = []
    motor_t = []
    if last_file:
        last_file_name = find_latest_file(
            "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/")+".txt"
        latest_file_path = os.path.join(
            "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/", last_file_name)
        file_path = latest_file_path
        
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

    for index, i in enumerate(total_data_list):
        if delete_zero and i[3] == 0:
            continue
        motor_t.append(i[0])
        motor_real.append(i[1])
        motor_cmd.append(i[2])
        motro_expect.append(i[3])
    return motor_t,motor_real,motor_cmd,motro_expect




def continus_predict(motor_t,motor_real,motor_cmd,motor_expect,model,if_plot=False,if_difference = False):
    fig, ax = plt.subplots()

    moving_win = 0
    need_data_len = 50
    if len(motor_real) >= input_size:
        current_index_0 = []
        acc, count = 0, 0
        for index, item in enumerate(motor_cmd):
            predict_y = []
            predict_x = []
            for j in range(1):

                next_index = moving_win + j + need_data_len + 1
                next_t = motor_t[next_index]  # 下一帧数
                next_real = motor_real[next_index+20]

                cur_index = moving_win + j + need_data_len
                cur_t_not_interp = motor_t[moving_win + j:cur_index + 1]  # 当前帧
                cur_cmd_not_intep = motor_cmd[moving_win + j: cur_index + 1]
                cur_expect_not_intep = motor_expect[moving_win + j: cur_index + 1]
                cur_real_not_interp = motor_real[moving_win + j:cur_index + 1]

                cur_t = motor_t[cur_index]
                cur_t_interp = np.arange(cur_t-0.98, cur_t + 0.02, 0.02) 
                cur_cmd_t_interp = np.arange(cur_t-0.98, cur_t + 0.02, 0.02)  # 当前帧(为了取到当前时间所以，所以调整了相加的时间)

                if if_difference:
                    cur_cmd_t_interp = np.arange(cur_t-1.0, cur_t + 0.02, 0.02)  # 当前帧(为了取到当前时间所以，所以调整了相加的时间)
                
                cur_real_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_real_not_interp), 3)
                
                cur_expect_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_expect_not_intep), 3)    
                           
                cur_cmd_interp = np.round(
                    np.interp(cur_cmd_t_interp, cur_t_not_interp, cur_cmd_not_intep), 3)
                if if_difference:
                    if len(cur_cmd_interp) > len(cur_t_interp) + 1:
                        cur_cmd_interp = np.delete(cur_cmd_interp,0)
                    cur_cmd_interp = np.round(np.diff(cur_cmd_interp),3)

                cur_real = cur_real_interp[-1]

                # 不使用差值
                # cur_real_interp = motor_real[moving_win+j+1:moving_win + j + need_data_len +1]
                # cur_cmd_interp = motor_cmd[moving_win+j+1:moving_win + j + need_data_len +1]

                cur_real_interp_tensor = torch.tensor(
                    np.array(cur_real_interp)).to(device)
                cur_cmd_interp_tensor = torch.tensor(
                    np.array(cur_cmd_interp)).to(device)
                cur_expect_interp_tensor = torch.tensor(
                    np.array(cur_expect_interp)).to(device)
                images = cur_real_interp_tensor.view(-1, 1, input_size)
                images2 = cur_cmd_interp_tensor.view(-1, 1, input_size)
                images3 = cur_expect_interp_tensor.view(-1, 1, input_size)
                outputs = model(images, images2,images3)
                pred_next_real = np.round(outputs.unsqueeze(
                    1).item(), 3)  # 将目标的形状从[2]变为[2, 1]

                predict_x.append(index + 50 + j)

                if next_real != 0:
                    count += 1

                    if abs(pred_next_real - next_real) < 0.01:
                        acc += 1
                    print(acc/count, index+49, next_real, pred_next_real)
                    print(images, images2, images3)
                    predict_y.append(np.round(pred_next_real, 3))
                else:
                    predict_y.append(0.0)

                # y = np.round(test_labels.item(),3)
                if j == 0:
                    current_index_0.append(predict_y[-1])
                    # print(abs(target-y))

            moving_win += 1
            x = np.linspace(0, len(motor_real), len(motor_real))
            if if_plot:
                ax.plot(x, motor_real, linewidth=3.0)  
                ax.plot(x, motro_expect, linewidth=3.0)  

                x1 = np.linspace(input_size, input_size +
                                 len(current_index_0)-1, len(current_index_0))
                ax.plot(x1, current_index_0, color='c',
                        label='当前帧')  

                ax.scatter(predict_x, predict_y)
                if pause:
                    plt.waitforbuttonpress()
                    current_xlim = ax.get_xlim()
                    current_ylim = ax.get_ylim()
                    ax.cla()  # 清除图形
                    ax.set_xlim(current_xlim)
                    ax.set_ylim(current_ylim)
                else:
                    current_xlim = ax.get_xlim()
                    current_ylim = ax.get_ylim()
                    plt.pause(0.001)  
                    ax.cla()  # 清除图形
                    ax.set_xlim(current_xlim)
                    ax.set_ylim(current_ylim)
                plt.ioff()  # 关闭画图窗口


def speed_control(motor_t,motor_real,motor_cmd,motor_expect,model,if_plot=False,if_difference = False):
    moving_win = 0
    need_data_len = 50
    if len(motor_real) >= input_size:
        current_index_0 = []
        acc, count = 0, 0
        for index, item in enumerate(motor_cmd):
            predict_y = []
            predict_x = []
            for j in range(1):

                next_index = moving_win + j + need_data_len + 1
                next_t = motor_t[next_index]  # 下一帧数
                next_real = motor_real[next_index]

                cur_index = moving_win + j + need_data_len
                cur_t_not_interp = motor_t[moving_win + j:cur_index + 1]  # 当前帧
                cur_cmd_not_intep = motor_cmd[moving_win + j: cur_index + 1]
                cur_real_not_interp = motor_real[moving_win + j:cur_index + 1]
                cur_expect_not_interp = motor_expect[moving_win + j:cur_index + 1]

                cur_t = motor_t[cur_index]
                cur_t_interp = np.arange(cur_t-0.98, cur_t + 0.02, 0.02) 
                cur_cmd_t_interp = np.arange(cur_t-0.98, cur_t+0.02, 0.02)  # 当前帧(为了取到当前时间所以，所以调整了相加的时间)
                if if_difference:
                    cur_cmd_t_interp = np.arange(cur_t-1.02, cur_t-0.01, 0.02)  # 当前帧(为了取到当前时间所以，所以调整了相加的时间)

                cur_real_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_real_not_interp), 3)
                
                cur_cmd_interp = np.round(
                    np.interp(cur_cmd_t_interp, cur_t_not_interp, cur_cmd_not_intep), 3)
                
                cur_expect_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_expect_not_interp), 3)

                if if_difference:
                    if len(cur_cmd_interp) >= 50:
                        cur_cmd_interp = np.delete(cur_cmd_interp,0)
                    cur_cmd_interp = np.round(np.diff(cur_cmd_interp),3)
                    cur_cmd_interp = np.append(cur_cmd_interp,0.0)

                # cur_real = cur_real_interp[-1]
                # cur_cmd_interp1 = np.append(cur_cmd_interp1,0.0)
                i = -0.15
                while i < 0.15:
                    cur_cmd_interp[-1] = i
                    imgs1 = torch.tensor(np.array(cur_real_interp)).to(device)
                    imgs2 = torch.tensor(np.array(cur_cmd_interp)).to(device)
                    imgs3 = torch.tensor(np.array(cur_expect_interp)).to(device)
                    images = imgs1.view(-1, 1, input_size)
                    images2 = imgs2.view(-1, 1, input_size)
                    images3 = imgs3.view(-1, 1, input_size)
                    outputs = model(images, images2,images3)
                    test_labels = outputs.unsqueeze(1)
                    if if_difference:
                        print(cur_real_not_interp[-1]+cur_cmd_interp[-1],np.round(test_labels.item(), 3))
                    else:
                        print(cur_cmd_interp[-1],np.round(test_labels.item(), 3))
                    i+=0.01
                print("********************")
                predict_x.append(index + 50 + j)

                # if next_real != 0:
                #     count += 1
                #     p = abs(50 * (cur_real - pred_next_real) * (next_t - (cur_t+0.02)))
                #     if (cur_real < pred_next_real):
                #         p = pred_next_real + p
                #     else:
                #         p = pred_next_real - p

                #     if abs(pred_next_real - next_real) < 0.01:
                #         acc += 1
                #     # if abs(next_t - (cur_t+0.02)) > 0.04:
                #     print(acc/count, index+49, next_real, pred_next_real)
                #     predict_y.append(np.round(pred_next_real, 3))
                # else:
                #     predict_y.append(0.0)

                # # y = np.round(test_labels.item(),3)
                # if j == 0:
                #     current_index_0.append(predict_y[-1])
                #     # print(abs(target-y))

            moving_win += 1
            # x = np.linspace(0, len(motor_real), len(motor_real))
            # if if_plot:
            #     ax.plot(x, motor_real, linewidth=3.0)  
            #     ax.plot(x, motro_expect, linewidth=3.0)  

            #     x1 = np.linspace(input_size, input_size +
            #                      len(current_index_0)-1, len(current_index_0))
            #     ax.plot(x1, current_index_0, color='c',
            #             label='当前帧')  

            #     ax.scatter(predict_x, predict_y)
            #     if pause:
            #         plt.waitforbuttonpress()
            #         current_xlim = ax.get_xlim()
            #         current_ylim = ax.get_ylim()
            #         ax.cla()  # 清除图形
            #         ax.set_xlim(current_xlim)
            #         ax.set_ylim(current_ylim)
            #     else:
            #         current_xlim = ax.get_xlim()
            #         current_ylim = ax.get_ylim()
            #         plt.pause(0.001)  
            #         ax.cla()  # 清除图形
            #         ax.set_xlim(current_xlim)
            #         ax.set_ylim(current_ylim)
            #     plt.ioff()  # 关闭画图窗口



# 可视化数据集
def visualizeDataset(motor_t,motor_real,motor_cmd,motro_expect):
    global delete
    global close
    fig, ax = plt.subplots(figsize=(10, 8))
    calCorrelation(motor_real,motor_cmd)
    acc_list = calACC(motor_real,motro_expect)
    while not delete and not close:
        fig.canvas.mpl_disconnect(
        fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册
        fig.canvas.mpl_connect('key_press_event', on_delete_press)
        x = np.linspace(0, len(motor_real), len(motor_real))
        ax.plot(x, motor_real, linewidth=3.0, color='r')  
        ax.plot(x, np.array(motor_cmd), linewidth=3.0)  
        ax.plot(x, motro_expect, linewidth=3.0, color='b')  
        # ax.plot(x, np.array(motor_real) - np.array(motro_expect), linewidth=3.0, color='g')  

        for item in acc_list:
            plt.plot(item[0], 0, 'o')  # 'o' 表示用圆圈标记数据点
            plt.text(item[0], 0, f'{np.round(item[2],2)}', ha='right', va='bottom')
        plt.legend(['motor_real',"motor_cmd","motor_expect"])
        plt.waitforbuttonpress()
        plt.pause(0.001)  
    ax.cla()  # 清除图形
    plt.close()  # 关闭画图窗口
    delete = False
    close = False

def group_data(data, step=1):
    if_find = False
    find_index = 0
    if not data:
        return []
    groups = []
    for i in range(1, len(data)):
        # 检查步长
        if not if_find and i - find_index >= step:
            if_find = True
        if if_find:
        # 检查是否上升到0或下降到0
            if (data[i] == 0 and data[i-1] != 0) or (data[i-1] == 0 and data[i] != 0):
                groups.append(i)
                if_find = False
                find_index = i

    return groups

# 分段计算贴合度    
def calACC(data1,data2, thres = 0.01):
    index = group_data(data2,5)
    d1 = np.split(data1, index)
    d2 = np.split(data2, index)
    index.append(len(data2))
    acc_list = []
    for i,d in enumerate(d1):
        minus_speed = abs(d1[i] - d2[i])
        arr = (minus_speed < thres).astype(int)
        ones_ratio = np.sum(arr) / arr.size
        acc_list.append([index[i], data2[index[i]-1], ones_ratio])
        # print("贴合度:",index[i], data2[index[i]-1], ones_ratio)
    return acc_list
# 计算数据的相关性
def calCorrelation(data1, data2, delete_zero = False):
    # 将数据转换为NumPy数组
    data1 = np.array(data1)
    data2 = np.array(data2)
    if delete_zero:
        zero_indices = np.where(data1 == 0)[0]
        data1 = np.delete(data1, zero_indices)
        data2 = np.delete(data2, zero_indices)
    # 计算Pearson相关系数
    pearson_corr, _ = stats.pearsonr(data1, data2)
    # 计算Spearman相关系数
    spearman_corr, _ = stats.spearmanr(data1, data2)
 
    # 打印相关系数
    print("Pearson相关系数: ", pearson_corr)
    print("Spearman相关系数: ", spearman_corr)
    
# 计算数据的滞后性
def calPeason(data_t, data1, data2, ax, if_plot = False):
    cur_t_interp = np.arange(motor_t[0], motor_t[-1], 0.02) 
    y1 = np.array([data1])
    y2 = np.array([data2])
    
    # cur_cmd_interp = np.round(np.interp(cur_t_interp, motor_t, motor_cmd), 3)
    # cur_real_interp = np.round(np.interp(cur_t_interp, motor_t, motor_real), 3)
                    
    x = np.linspace(0, len(y1), len(y1))
    """利用pearson计算滞后性"""
    # data_cor = pd.DataFrame(np.array([cur_real_interp, cur_cmd_interp]).T, columns=['y1', 'y2'])
    # for i in range(5, 25):
    #     data_cor[str(i)] = data_cor['y2'].shift(i) 
    # data_cor.dropna(inplace=True)
    # p = data_cor.corr()
    # print("person相关系数：\n", data_cor.corr())
    
    # plt.plot(range(5, 25),data_cor.corr().iloc[0][2:].values)
    # plt.legend(['y1', 'y2'])
    # plt.title('pearson')
    # plt.xlabel('y2-lag_order')
    # plt.show()

    """利用互相关性计算滞后性"""
    a = np.correlate(y2, y1, mode="same")
    res = len(a) // 2 - a.argmax()
    # if res != 0 :
    print("y1滞后y2：", len(a) // 2 - a.argmax())  # 若为负数，说明y1提前y2
    if if_plot:
        if res > 0:
            temp = len(x[:-res])
            plt.plot(x, y1,)
            plt.plot(x[:-res],y2[res:res+temp]) # 结论y1超前y2五个单位。将y1时间向前错位即可重合
            plt.plot(x,y2[30:],linewidth=3.8) 

        elif res == 0:
            plt.plot(x, y1)
            plt.plot(x,y2[30:])
        elif res < 0:
            a = np.correlate(y1, y2, mode="same")
            res = len(a) // 2 - a.argmax()
            temp = len(x[:res])
            plt.plot(x, y1,)
            plt.plot(x[:res],y2[res:res+temp]) # 结论y1超前y2五个单位。将y1时间向前错位即可重合
            plt.plot(x,y2[30:],linewidth=3.8) 

        plt.legend(['real',"cmd","or","oc"])
        plt.pause(0.001)  
        ax.cla()  # 清除图形
    return res
    
def testSingleData(model):
    predict_motor_real = [0.05299999937415123,0.052000001072883606,0.05400000140070915,0.050999999046325684,0.04800000041723251,0.04500000178813934,0.04399999976158142,0.0430000014603138,0.04100000113248825,0.0430000014603138,0.03999999910593033,0.03799999877810478,0.03700000047683716,0.039000000804662704,0.04100000113248825,0.04100000113248825,0.03999999910593033,0.039000000804662704,0.03700000047683716,0.03500000014901161,0.03500000014901161,0.03700000047683716,0.039000000804662704,0.041999999433755875,0.04100000113248825,0.041999999433755875,0.04399999976158142,0.04500000178813934,0.04399999976158142,0.041999999433755875,0.041999999433755875,0.04399999976158142,0.04399999976158142,0.04399999976158142,0.0430000014603138,0.03999999910593033,0.03799999877810478,0.039000000804662704,0.03799999877810478,0.039000000804662704,0.03700000047683716,0.039000000804662704,0.03799999877810478,0.03999999910593033,0.041999999433755875,0.03999999910593033,0.03999999910593033,0.04100000113248825,0.04100000113248825,0.04399999976158142]
    predict_motor_cmd = [-0.000999998301267624,0.0020000003278255463,-0.0030000023543834686,-0.00299999862909317,-0.00299999862909317,-0.0010000020265579224,-0.000999998301267624,-0.0020000003278255463,0.0020000003278255463,-0.0030000023543834686,-0.0020000003278255463,-0.000999998301267624,0.0020000003278255463,0.0020000003278255463,0.0,-0.0010000020265579224,-0.000999998301267624,-0.0020000003278255463,-0.0020000003278255463,0.0,0.0020000003278255463,0.0020000003278255463,0.00299999862909317,-0.000999998301267624,0.0,0.00299999862909317,0.0010000020265579224,-0.0010000020265579224,-0.0020000003278255463,0.0,0.0020000003278255463,0.0,0.0,-0.000999998301267624,-0.0030000023543834686,-0.0020000003278255463,0.0010000020265579224,-0.0010000020265579224,0.0010000020265579224,-0.0020000003278255463,0.0020000003278255463,-0.0010000020265579224,0.0020000003278255463,0.0020000003278255463,-0.0020000003278255463,0.0,0.0010000020265579224,0.0,0.00299999862909317,-0.04382335767149925]
    predict_motor_expect = [0.0020000000949949026,0.003000000026077032,0.004000000189989805,0.004000000189989805,0.004999999888241291,0.006000000052154064,0.007000000216066837,0.00800000037997961,0.00800000037997961,0.008999999612569809,0.009999999776482582,0.010999999940395355,0.012000000104308128,0.012000000104308128,0.013000000268220901,0.014000000432133675,0.014999999664723873,0.014999999664723873,0.01600000075995922,0.017000000923871994,0.017999999225139618,0.017999999225139618,0.01899999938905239,0.019999999552965164,0.020999999716877937,0.020999999716877937,0.02199999988079071,0.023000000044703484,0.024000000208616257,0.024000000208616257,0.02500000037252903,0.026000000536441803,0.026000000536441803,0.027000000700354576,0.02800000086426735,0.02800000086426735,0.028999999165534973,0.029999999329447746,0.029999999329447746,0.03099999949336052,0.03200000151991844,0.03200000151991844,0.032999999821186066,0.032999999821186066,0.03400000184774399,0.03500000014901161,0.03500000014901161,0.035999998450279236,0.03700000047683716,0.03700000047683716]
    # predict_motor_real = [0.014999999664723873,0.014000000432133675,0.014000000432133675,0.014000000432133675,0.014999999664723873,0.017000000923871994,0.019999999552965164,0.019999999552965164,0.020999999716877937,0.019999999552965164,0.01899999938905239,0.017999999225139618,0.017000000923871994,0.017000000923871994,0.01600000075995922,0.01600000075995922,0.01600000075995922,0.01899999938905239,0.020999999716877937,0.024000000208616257,0.026000000536441803,0.027000000700354576,0.027000000700354576,0.026000000536441803,0.026000000536441803,0.02500000037252903,0.024000000208616257,0.024000000208616257,0.023000000044703484,0.023000000044703484,0.023000000044703484,0.024000000208616257,0.024000000208616257,0.02500000037252903,0.027000000700354576,0.028999999165534973,0.03099999949336052,0.03200000151991844,0.032999999821186066,0.03500000014901161,0.03700000047683716,0.03799999877810478,0.03799999877810478,0.03700000047683716,0.03500000014901161,0.032999999821186066,0.03099999949336052,0.028999999165534973,0.02800000086426735,0.02679322473704815]
    # predict_motor_cmd = [0.014000000432133675,0.014000000432133675,0.014999999664723873,0.017000000923871994,0.01899999938905239,0.019999999552965164,0.020999999716877937,0.019999999552965164,0.01899999938905239,0.017999999225139618,0.017999999225139618,0.017000000923871994,0.01600000075995922,0.01600000075995922,0.01600000075995922,0.01899999938905239,0.020999999716877937,0.024000000208616257,0.026000000536441803,0.027000000700354576,0.027000000700354576,0.026000000536441803,0.026000000536441803,0.02500000037252903,0.024000000208616257,0.024000000208616257,0.023000000044703484,0.023000000044703484,0.023000000044703484,0.024000000208616257,0.024000000208616257,0.02500000037252903,0.027000000700354576,0.028999999165534973,0.03099999949336052,0.03200000151991844,0.032999999821186066,0.03500000014901161,0.03700000047683716,0.03799999877810478,0.03799999877810478,0.03700000047683716,0.03500000014901161,0.032999999821186066,0.03099999949336052,0.028999999165534973,0.02800000086426735,0.027000000700354576,0.026000000536441803,0.05]
    # predict_motor_expect = [0.5210000276565552,0.5199999809265137,0.5189999938011169,0.5180000066757202,0.5170000195503235,0.5149999856948853,0.5139999985694885,0.5130000114440918,0.5120000243186951,0.5099999904632568,0.5090000033378601,0.5080000162124634,0.5070000290870667,0.5049999952316284,0.5040000081062317,0.5019999742507935,0.5009999871253967,0.5,0.49799999594688416,0.4970000088214874,0.4950000047683716,0.49399998784065247,0.492000013589859,0.4909999966621399,0.49000000953674316,0.48899999260902405,0.4869999885559082,0.4860000014305115,0.48399999737739563,0.4830000102519989,0.48100000619888306,0.47999998927116394,0.4790000021457672,0.47699999809265137,0.47600001096725464,0.4740000069141388,0.4729999899864197,0.47099998593330383,0.47099998593330383,0.4690000116825104,0.46799999475479126,0.4659999907016754,0.46399998664855957,0.46299999952316284,0.4620000123977661,0.46000000834465027,0.45899999141693115,0.4569999873638153,0.45500001311302185,0.45399999618530273]
    imgs1 = torch.tensor(np.array(predict_motor_real)).to(device)
    imgs2 = torch.tensor(np.array(predict_motor_cmd)).to(device)
    imgs3= torch.tensor(np.array(predict_motor_expect)).to(device)
    images = imgs1.view(-1, 1, input_size)
    images2 = imgs2.view(-1, 1, input_size)
    images3 = imgs3.view(-1, 1, input_size)
    outputs = model(images, images2)
    test_labels = outputs.unsqueeze(1)
    print(test_labels)

    # predict_motor_cmd1 = np.round(np.diff(predict_motor_cmd),3)

    # predict_motor_cmd1 = np.append(predict_motor_cmd1,0.0)
    # i = -0.15
    # while i < 0.15:
    #     predict_motor_cmd1[-1] = i
    #     imgs1 = torch.tensor(np.array(predict_motor_real)).to(device)
    #     imgs2 = torch.tensor(np.array(predict_motor_cmd1)).to(device)
    #     images = imgs1.view(-1, 1, input_size)
    #     images2 = imgs2.view(-1, 1, input_size)
    #     outputs = model(images, images2)
    #     test_labels = outputs.unsqueeze(1)
    #     print(predict_motor_cmd[-1]+i,np.round(test_labels.item(), 3))
    #     i+=0.01


    
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(r"model/model.pth").to(device)
    model.eval()

    orin_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/"
    for root, dirs, files in os.walk(orin_path):
        global file_name
        for file_name in files:
            motor_t,motor_real,motor_cmd,motro_expect = loadDataSet(orin_path+file_name,False,False)
            visualizeDataset(motor_t,motor_real,motor_cmd,motro_expect)
            
    # motor_t,motor_real,motor_cmd,motro_expect = loadDataSet("/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/robokit_2024-09-12_11-16-51.0.log.txt",last_file=False)
    # fig, ax = plt.subplots()
    # aaa = []
    # moving_win = 30
    # need_data_len = 30
    # if len(motor_real) >= input_size:
    #     current_index_0 = []
    #     acc, count = 0, 0

    #     for index, item in enumerate(motor_cmd):
    #         try:

    #             predict_y = []
    #             predict_x = []
    #             for j in range(1):
    #                 # next_index = moving_win + j + need_data_len + 1
    #                 # next_t = motor_t[next_index]  # 下一帧数
    #                 # next_real = motor_real[next_index]

    #                 cur_index = moving_win + j + need_data_len
    #                 cur_t_not_interp = motor_t[moving_win + j:cur_index + 1]  # 当前帧
    #                 cur_cmd_not_intep = motor_cmd[moving_win + j -30: cur_index + 1]
    #                 cur_real_not_interp = motor_real[moving_win + j:cur_index + 1]
    #                 cur_expect_not_interp = motro_expect[moving_win + j:cur_index + 1]
    #                 res = calPeason(cur_t_not_interp,cur_real_not_interp,cur_cmd_not_intep,cur_expect_not_interp,ax)
    #                 if res != 0:
    #                     aaa.append(res)
    #                 moving_win += 1
    #         except Exception as expection:
    #             print(Exception)
    # print(np.mean(np.array(aaa)))
    # visualizeDataset(motor_t,motor_real,motor_cmd,motro_expect)
    # calACC(motor_real,motor_cmd,motro_expect)
    testSingleData(model)

    # speed_control(motor_t,motor_real,motor_cmd,motro_expect,model,False,False)
    # calPeason(motor_t,motor_real,motor_cmd,motro_expect)
    # continus_predict(motor_t,motor_real,motor_cmd,motro_expect,model,False,False)
