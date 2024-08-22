from matplotlib import pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from LSTM import *
import os

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
def on_key_press(event):
    global pause
    if pause:
        pause = False
    else:
        pause = True

fig, ax = plt.subplots()
fig.canvas.mpl_disconnect(
    fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册
fig.canvas.mpl_connect('key_press_event', on_key_press)

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


def loadDataSet():
	input_size = 50
	total_data_list = []
	motor_real = []
	motor_cmd = []
	motro_expect = []
	motor_t = []
	last_file_name = find_latest_file(
		"/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/")+".txt"
	latest_file_path = os.path.join(
		"/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/", last_file_name)
	with open(latest_file_path, "r") as f:
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
                cur_t_interp = np.arange(cur_t-0.98, cur_t + 0.02, 0.02)  # 当前帧(为了取到当前时间所以，所以调整了相加的时间)
                cur_real_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_real_not_interp), 3)
                cur_cmd_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_cmd_not_intep), 3)
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


def speed_control(motor_t,motor_real,motor_cmd,motro_expect,model):
    moving_win = 0
    if len(motor_real) >= input_size:
        current_index_0 = []
        current_index_1 = []
        current_index_2 = []
        for index, item in enumerate(motor_cmd):
            predict_y = []
            predict_x = []

            predict_motor_real = motor_real[moving_win:moving_win+25]
            predict_motor_cmd = motor_cmd[moving_win:moving_win+25]
            predict_motor_t = motor_t[moving_win:moving_win+25]

            end_time = motor_t[moving_win+25-1]
            start_time = motor_t[moving_win]

            new_timestamps = np.arange(end_time-1.5, end_time, 0.05)
            predict_motor_real = np.round(
                np.interp(new_timestamps, predict_motor_t, predict_motor_real), 3)
            predict_motor_cmd = np.round(
                np.interp(new_timestamps, predict_motor_t, predict_motor_cmd), 3)

            imgs1 = torch.tensor(np.array(predict_motor_real)).to(device)
            imgs2 = torch.tensor(np.array(predict_motor_cmd)).to(device)
            images = imgs1.view(-1, 1, input_size)
            images2 = imgs2.view(-1, 1, input_size)
            outputs = model(images, images2)
            test_labels = outputs.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
            current_index_2.append(predict_motor_cmd[-1])

            cnt = 0
            while abs(motro_expect[moving_win+25] - test_labels.item()) > 0.005:
                predict_motor_cmd[-1] += (motro_expect[moving_win +
                                          25] - test_labels.item())
                imgs2 = torch.tensor(np.array(predict_motor_cmd)).to(device)
                images2 = imgs2.view(-1, 1, input_size)
                outputs = model(images, images2)
                test_labels = outputs.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
                cnt += 1
                if cnt >= 20:
                    break

            predict_x.append(index+25)
            predict_y.append(np.round(test_labels.item(), 3))

            y = np.round(test_labels.item(), 3)
            current_index_0.append(y)
            current_index_1.append(predict_motor_cmd[-1])

            moving_win += 1
            x = np.linspace(0, len(motor_real), len(motor_real))

            # ax.plot(x, motor_real, linewidth = 3.0)  # 画出当前x列表和y列表中的值的图形
            ax.plot(x, motro_expect, linewidth=3.0)  # 画出当前x列表和y列表中的值的图形

            x1 = np.linspace(input_size+10, input_size +
                             len(current_index_0)-1+10, len(current_index_0))
            ax.plot(x1, current_index_0, color='c',
                    label='当前帧')  # 画出当前x列表和y列表中的值的图形

            x2 = np.linspace(input_size+10, input_size +
                             len(current_index_1)-1+10, len(current_index_1))
            ax.plot(x2, current_index_1, color='g',
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
    print(len(motor_real))
    x = np.linspace(0, len(motor_real), len(motor_real))
    ax.plot(x, motor_real, linewidth=3.0, color='r')  # 画出当前x列表和y列表中的值的图形
    ax.plot(x, motor_cmd, linewidth=3.0)  # 画出当前x列表和y列表中的值的图形
    ax.plot(x, motro_expect, linewidth=3.0, color='b')  # 画出当前x列表和y列表中的值的图形
    plt.show()


def testSingleData(model):
    predict_motor_real = [0.050999999046325684, 0.050999999046325684, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.04899999871850014, 0.04899999871850014, 0.04699999839067459, 0.04600000008940697, 0.04600000008940697, 0.04699999839067459, 0.04800000041723251, 0.04800000041723251, 0.04600000008940697, 0.04500000178813934, 0.04600000008940697, 0.04399999976158142, 0.0430000014603138, 0.0430000014603138, 0.04399999976158142, 0.04500000178813934, 0.04600000008940697, 0.04600000008940697, 0.04500000178813934, 0.04500000178813934,
                          0.04399999976158142, 0.0430000014603138, 0.0430000014603138, 0.0430000014603138, 0.041999999433755875, 0.041999999433755875, 0.041999999433755875, 0.041999999433755875, 0.041999999433755875, 0.041999999433755875, 0.0430000014603138, 0.04399999976158142, 0.04399999976158142, 0.04399999976158142, 0.04500000178813934, 0.04600000008940697, 0.04600000008940697, 0.04600000008940697, 0.04600000008940697, 0.04699999839067459, 0.04600000008940697, 0.04399999976158142, 0.0430000014603138, 0.041999999433755875, 0.04100000113248825]
    predict_motor_cmd = [0.050999999046325684, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.04899999871850014, 0.04899999871850014, 0.04699999839067459, 0.04600000008940697, 0.04600000008940697, 0.04699999839067459, 0.04800000041723251, 0.04800000041723251, 0.04600000008940697, 0.04500000178813934, 0.04600000008940697, 0.04399999976158142, 0.0430000014603138, 0.0430000014603138, 0.04399999976158142, 0.04500000178813934, 0.04600000008940697, 0.04600000008940697, 0.04500000178813934, 0.04500000178813934, 0.04399999976158142,
                         0.0430000014603138, 0.0430000014603138, 0.0430000014603138, 0.041999999433755875, 0.041999999433755875, 0.041999999433755875, 0.041999999433755875, 0.041999999433755875, 0.041999999433755875, 0.0430000014603138, 0.04399999976158142, 0.04399999976158142, 0.04399999976158142, 0.04500000178813934, 0.04600000008940697, 0.04600000008940697, 0.04600000008940697, 0.04600000008940697, 0.04699999839067459, 0.04600000008940697, 0.04399999976158142, 0.0430000014603138, 0.041999999433755875, 0.04100000113248825, 0.041999999433755875]
    imgs1 = torch.tensor(np.array(predict_motor_real)).to(device)
    imgs2 = torch.tensor(np.array(predict_motor_cmd)).to(device)
    images = imgs1.view(-1, 1, input_size)
    images2 = imgs2.view(-1, 1, input_size)
    outputs = model(images, images2)
    test_labels = outputs.unsqueeze(1)
    print(np.round(test_labels.item(), 3))


if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = torch.load(r"model/model.pth").to(device)
	model.eval()
	motor_t,motor_real,motor_cmd,motro_expect = loadDataSet()
	# testSingleData(model)
	# visualizeDataset(motor_t,motor_real,motor_cmd,motro_expect)
	continus_predict(motor_t,motor_real,motor_cmd,motro_expect,model,True)
