from matplotlib import pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from LSTM import *
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
# def loadDataTest(if_random):
#     total_data_list = randomSortData(r"data/train.txt",30,if_random)
#     data_label = []
#     real_data = []
#     cmd_data = []
#     for data_list in total_data_list:

#         data_list = [float(item) for item in data_list]
#         label = data_list[0]
#         len_data_list = len(data_list)
#         real_list = data_list[1:int((len_data_list-1)/2)+1]
#         cmd_list = data_list[int((len_data_list-1)/2)+1:]
#         while len(real_list) >input_size:
#             real_list.pop(0)
#             cmd_list.pop(0)
#         if len(real_list) == input_size:
#             data_label.append(label)
#             real_data.append(np.array(real_list))
#             cmd_data.append(np.array(cmd_list))

#     real_scaler = MinMaxScaler(feature_range=(-1, 1))
#     cmd_scaler = MinMaxScaler(feature_range=(-1, 1))

#     return real_scaler,cmd_scaler

# 查找最后更新进文件夹的文件

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


def loadDataSet(file_path="", last_file=False):
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
        motor_t.append(i[0])
        motor_real.append(i[1])
        motor_cmd.append(i[2])
        motro_expect.append(i[3])
    return motor_t,motor_real,motor_cmd,motro_expect




def continus_predict(motor_t,motor_real,motor_cmd,motro_expect,model,if_plot=False):
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
                next_real = motor_real[next_index]

                cur_index = moving_win + j + need_data_len
                cur_t_not_interp = motor_t[moving_win + j:cur_index + 1]  # 当前帧
                cur_cmd_not_intep = motor_cmd[moving_win + j: cur_index + 1]
                cur_real_not_interp = motor_real[moving_win + j:cur_index + 1]

                cur_t = motor_t[cur_index]
                cur_t_interp = np.arange(cur_t-0.98, cur_t + 0.02, 0.02) 
                cur_cmd_t_interp = np.arange(cur_t-1.0, cur_t + 0.02, 0.02)  # 当前帧(为了取到当前时间所以，所以调整了相加的时间)

                cur_real_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_real_not_interp), 3)
                
                cur_cmd_interp = np.round(
                    np.interp(cur_cmd_t_interp, cur_t_not_interp, cur_cmd_not_intep), 3)
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
                images = cur_real_interp_tensor.view(-1, 1, input_size)
                images2 = cur_cmd_interp_tensor.view(-1, 1, input_size)
                outputs = model(images, images2)
                pred_next_real = np.round(outputs.unsqueeze(
                    1).item(), 3)  # 将目标的形状从[2]变为[2, 1]

                predict_x.append(index + 50 + j)

                if next_real != 0:
                    count += 1
                    p = abs(50 * (cur_real - pred_next_real) * (next_t - (cur_t+0.02)))
                    if (cur_real < pred_next_real):
                        p = pred_next_real + p
                    else:
                        p = pred_next_real - p

                    if abs(pred_next_real - next_real) < 0.01:
                        acc += 1
                    # if abs(next_t - (cur_t+0.02)) > 0.04:
                    print(acc/count, index+49, next_real, pred_next_real)
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
                ax.plot(x, motor_real, linewidth=3.0)  # 画出当前x列表和y列表中的值的图形
                ax.plot(x, motro_expect, linewidth=3.0)  # 画出当前x列表和y列表中的值的图形

                x1 = np.linspace(input_size, input_size +
                                 len(current_index_0)-1, len(current_index_0))
                ax.plot(x1, current_index_0, color='c',
                        label='当前帧')  # 画出当前x列表和y列表中的值的图形

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
                    plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
                    ax.cla()  # 清除图形
                    ax.set_xlim(current_xlim)
                    ax.set_ylim(current_ylim)
                plt.ioff()  # 关闭画图窗口


def speed_control(motor_t,motor_real,motor_cmd,motro_expect,model,if_plot=False):
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

                cur_t = motor_t[cur_index]
                cur_t_interp = np.arange(cur_t-0.98, cur_t + 0.02, 0.02) 
                cur_cmd_t_interp = np.arange(cur_t-1.0, cur_t, 0.02)  # 当前帧(为了取到当前时间所以，所以调整了相加的时间)

                cur_real_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_real_not_interp), 3)
                
                cur_cmd_interp = np.round(
                    np.interp(cur_cmd_t_interp, cur_t_not_interp, cur_cmd_not_intep), 3)
                if len(cur_cmd_interp) > len(cur_t_interp) + 1:
                    cur_cmd_interp = np.delete(cur_cmd_interp,0)
                cur_cmd_interp1 = np.round(np.diff(cur_cmd_interp),3)

                cur_real = cur_real_interp[-1]

                cur_cmd_interp1 = np.append(cur_cmd_interp1,0.0)
                i = -0.15
                while i < 0.15:
                    cur_cmd_interp1[-1] = i
                    imgs1 = torch.tensor(np.array(cur_real_interp)).to(device)
                    imgs2 = torch.tensor(np.array(cur_cmd_interp1)).to(device)
                    images = imgs1.view(-1, 1, input_size)
                    images2 = imgs2.view(-1, 1, input_size)
                    outputs = model(images, images2)
                    test_labels = outputs.unsqueeze(1)
                    print(cur_cmd_interp[-1]+i,np.round(test_labels.item(), 3))
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
            x = np.linspace(0, len(motor_real), len(motor_real))
            if if_plot:
                ax.plot(x, motor_real, linewidth=3.0)  # 画出当前x列表和y列表中的值的图形
                ax.plot(x, motro_expect, linewidth=3.0)  # 画出当前x列表和y列表中的值的图形

                x1 = np.linspace(input_size, input_size +
                                 len(current_index_0)-1, len(current_index_0))
                ax.plot(x1, current_index_0, color='c',
                        label='当前帧')  # 画出当前x列表和y列表中的值的图形

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
                    plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
                    ax.cla()  # 清除图形
                    ax.set_xlim(current_xlim)
                    ax.set_ylim(current_ylim)
                plt.ioff()  # 关闭画图窗口



# 可视化数据集
def visualizeDataset(motor_t,motor_real,motor_cmd,motro_expect):
    global delete
    global close
    fig, ax = plt.subplots()

    while not delete and not close:
        # print(len(motor_real))
        fig.canvas.mpl_disconnect(
        fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册
        fig.canvas.mpl_connect('key_press_event', on_delete_press)
        x = np.linspace(0, len(motor_real), len(motor_real))
        ax.plot(x, motor_real, linewidth=3.0, color='r')  # 画出当前x列表和y列表中的值的图形
        ax.plot(x, motor_cmd, linewidth=3.0)  # 画出当前x列表和y列表中的值的图形
        ax.plot(x, motro_expect, linewidth=3.0, color='b')  # 画出当前x列表和y列表中的值的图形
        plt.waitforbuttonpress()
        plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
    ax.cla()  # 清除图形
    plt.close()  # 关闭画图窗口
    delete = False
    close = False



def test(motor_t,motor_real,motor_cmd,motro_expect):
    cur_t_interp = np.arange(motor_t[0], motor_t[-1], 0.02) 

    """原始数据"""
    y1 = np.array([motor_real])
    y2 = np.array([motro_expect])
    
    cur_real_interp = np.round(np.interp(cur_t_interp, motor_t, motor_real), 3)
    cur_cmd_interp = np.round(np.interp(cur_t_interp, motor_t, motro_expect), 3)
                    
    x = np.linspace(0, len(cur_real_interp), len(cur_real_interp))

    # plt.plot(x, cur_real_interp)
    # plt.plot(x, cur_cmd_interp)
    # plt.legend(['y1', 'y2'])
    # plt.show()

    """利用pearson计算滞后性"""
    # 从图中可以看出y2滞后5阶
    data_cor = pd.DataFrame(np.array([cur_real_interp, cur_cmd_interp]).T, columns=['y1', 'y2'])
    for i in range(15, 35):
        data_cor[str(i)] = data_cor['y2'].shift(i) 
    data_cor.dropna(inplace=True)
    p = data_cor.corr()
    print("person相关系数：\n", data_cor.corr())
    plt.plot(range(15, 35),data_cor.corr().iloc[0][2:].values)
    plt.legend(['y1', 'y2'])
    plt.title('pearson')
    plt.xlabel('y2-lag_order')
    plt.show()

    """利用互相关性计算滞后性"""
    a = np.correlate(cur_real_interp, cur_cmd_interp, mode="same")
    print("y1滞后y2：", len(a) // 2 - a.argmax())  # 若为负数，说明y1超前y2
    plt.plot(x[:-15],cur_real_interp[15:]) # 结论y1超前y2五个单位。将y1时间向前错位即可重合
    plt.plot(x, cur_cmd_interp)
    plt.legend(['real', 'cmd'])

    plt.show()
    
def testSingleData(model):

    predict_motor_real = [-0.0,-0.0,-0.0,0.0,0.0,0.0,0.0,-0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0,-0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0,-0.0,0.0,-0.0,-0.0,0.0,0.0,0.0,-0.0,-0.0,-0.0,-0.0,-0.0,0.0,0.0,0.0020000000949949026,0.004000000189989805,0.00800000037997961,0.012000000104308128,0.01600000075995922,0.019999999552965164,0.023]
    predict_motor_cmd = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.003000000026077032,0.009999999776482582,0.01600000075995922,0.020999999716877937,0.02500000037252903,0.02800000086426735,0.03099999949336052,0.03400000184774399,0.03700000047683716,0.03999999910593033,0.04399999976158142,0.04699999839067459,0.052000001072883606,0.0560000017285347,0.05999999865889549,0.06499999761581421,0.0689999982714653,0.07199999690055847,0.0729999989271164,0.07100000232458115,0.07000000029802322,0.07100000232458115,0.0729999989271164,0.07400000095367432,0.07500000298023224,0.07500000298023224,0.07100000232458115,0.07000000029802322,0.06800000369548798,0.06599999964237213,0.05999999865889549,0.04800000041723251,0.035999998450279236]
    predict_motor_cmd1 = np.round(np.diff(predict_motor_cmd),3)

    predict_motor_cmd1 = np.append(predict_motor_cmd1,0.0)
    i = -0.15
    while i < 0.15:
        predict_motor_cmd1[-1] = i
        imgs1 = torch.tensor(np.array(predict_motor_real)).to(device)
        imgs2 = torch.tensor(np.array(predict_motor_cmd1)).to(device)
        images = imgs1.view(-1, 1, input_size)
        images2 = imgs2.view(-1, 1, input_size)
        outputs = model(images, images2)
        test_labels = outputs.unsqueeze(1)
        print(predict_motor_cmd[-1]+i,np.round(test_labels.item(), 3))
        i+=0.01


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(r"model/model.pth").to(device)
    model.eval()
    orin_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/"
    for root, dirs, files in os.walk(orin_path):
        global file_name
        for file_name in files:
            motor_t,motor_real,motor_cmd,motro_expect = loadDataSet(orin_path+file_name,False)
            visualizeDataset(motor_t,motor_real,motor_cmd,motro_expect)
            # plt.pause()

    # testSingleData(model)

    # speed_control(motor_t,motor_real,motor_cmd,motro_expect,model,False)
    # test(motor_t,motor_real,motor_cmd,motro_expect)
	# continus_predict(motor_t,motor_real,motor_cmd,motro_expect,model,False)
