from matplotlib import pyplot as plt
import numpy as np
import torch 
from copy import deepcopy
from LSTM import *


def loadDataTest(if_random):
    total_data_list = randomSortData(r"data\train.txt",35,if_random)
    data_label = []
    real_data = []
    cmd_data = []
    for data_list in total_data_list:

        data_list = [float(item) for item in data_list]
        label = data_list[0]
        len_data_list = len(data_list)
        real_list = data_list[1:int((len_data_list-1)/2)+1]
        cmd_list = data_list[int((len_data_list-1)/2)+1:]
        while len(real_list) >input_size:
            real_list.pop(0)
            cmd_list.pop(0)
        if len(real_list) == input_size:
            data_label.append(label)
            real_data.append(np.array(real_list))
            cmd_data.append(np.array(cmd_list))

    real_scaler = MinMaxScaler(feature_range=(-1, 1))
    cmd_scaler = MinMaxScaler(feature_range=(-1, 1))

    return real_scaler,cmd_scaler



# 创建实时绘制横纵轴变量
x = []
y = []
predict_y = []
# 创建绘制实时损失的动态窗口
# plt.ion()

# # 创建循环
# for i in range(100):
# 	x.append(i)	# 添加i到x轴的数据中
# 	y.append(i**2)	# 添加i的平方到y轴的数据中
# 	plt.clf()  # 清除之前画的图
# 	plt.plot(x, y * np.array([-1]))  # 画出当前x列表和y列表中的值的图形
# 	plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
# 	plt.ioff()  # 关闭画图窗口
input_size = 35

total_data_list = []
motor_real = []
motor_cmd = []
path = r"C:\Projects\Python\LSTM_Speed_Predict\data\new_dataset_car2\compared_speed\robokit_2024-07-09_11-32-22.13.log.txt"
with open(path, "r") as f:
	line = f.readline() # 读取第一行
	while line:
		data_list = line.split(" ")
		data_list.pop()
		data_list.pop(0)
		data_list = [float(item) for item in data_list]
		total_data_list.append(data_list)
		line = f.readline() # 读取下一行

moving_win = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(r"model.pth").to(device)
model.eval()
for index, i in enumerate(total_data_list):
	motor_real.append(i[0])
	motor_cmd.append(i[1])

pause = False



def on_key_press(event):
	global pause
	if pause:
		pause = False
	else:
		pause = True

fig, ax = plt.subplots()
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)#取消默认快捷键的注册
fig.canvas.mpl_connect('key_press_event', on_key_press)

real_scaler,cmd_scaler = loadDataTest(False)
if len(motor_real) >= 35:
	current_index_0 = []
	current_index_1 = []
	current_index_2 = []
	current_index_3 = []
	current_index_4 = []

	for index, item in enumerate(motor_cmd):
		predict_y = []
		predict_x = []
		for j in range(5):
			predict_motor_cmd = motor_cmd[moving_win+j:moving_win+j+35]
			if j == 0:
				predict_motor_real = motor_real[moving_win:moving_win+35]
				tmp_predict_motor_real = deepcopy(predict_motor_real)
			else:
				predict_motor_real = tmp_predict_motor_real[j:]

			# predict_motor_real = real_scaler.fit_transform([predict_motor_real])
			# predict_motor_cmd = cmd_scaler.fit_transform([predict_motor_cmd])
			imgs1 = torch.tensor(np.array(predict_motor_real)).to(device)
			imgs2 = torch.tensor(np.array(predict_motor_cmd)).to(device)
			images = imgs1.view(-1, 1, input_size)
			images2 = imgs2.view(-1, 1, input_size)
			outputs = model(images,images2)
			test_labels = outputs.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
			predict_x.append(index+35+j)
			tmp_predict_motor_real.append(np.round(test_labels.item(),3))
			predict_y.append(np.round(test_labels.item(),3))
			y = np.round(test_labels.item(),3)
			if j == 0:
				current_index_0.append(y)
			elif j == 1:
				current_index_1.append(y)
			elif j == 2:
				current_index_2.append(y)
			elif j == 3:
				current_index_3.append(y)								
			elif j == 4:
				current_index_4.append(y)				
		moving_win += 1
		x = np.linspace(0, len(motor_real), len(motor_real))

		ax.plot(x, motor_real, linewidth = 3.0)  # 画出当前x列表和y列表中的值的图形

		x1 = np.linspace(35, 35+len(current_index_0)-1, len(current_index_0))
		ax.plot(x1, current_index_0,color='c', label='当前帧')  # 画出当前x列表和y列表中的值的图形

		# plt.plot(x1, current_index_1)  # 画出当前x列表和y列表中的值的图形

		x3 = np.linspace(38, 38+len(current_index_0)-1, len(current_index_0))
		ax.plot(x3, current_index_2,color='m', label='第三帧')  # 画出当前x列表和y列表中的值的图形

		# plt.plot(x1, current_index_3)  # 画出当前x列表和y列表中的值的图形

		x5 = np.linspace(40, 40+len(current_index_0)-1, len(current_index_0))
		ax.plot(x5, current_index_4,color='y', label='第五帧')  # 画出当前x列表和y列表中的值的图形

		ax.scatter(predict_x, predict_y)
		if pause:
			plt.waitforbuttonpress()
			current_xlim = ax.get_xlim()
			current_ylim = ax.get_ylim()
			ax.cla()        #清除图形
			ax.set_xlim(current_xlim)
			ax.set_ylim(current_ylim)
		else:
			current_xlim = ax.get_xlim()
			current_ylim = ax.get_ylim()
			plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
			ax.cla()        #清除图形
			ax.set_xlim(current_xlim)
			ax.set_ylim(current_ylim)
		plt.ioff()  # 关闭画图窗口
