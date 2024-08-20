from matplotlib import pyplot as plt
import numpy as np
import torch 
from copy import deepcopy
from LSTM import *
import os

def loadDataTest(if_random):
    total_data_list = randomSortData(r"data/train.txt",30,if_random)
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

# 创建实时绘制横纵轴变量
x = []
y = []
predict_y = []

input_size = 50

total_data_list = []
motor_real = []
motor_cmd = []
motro_expect = []
motor_t = []
last_file_name = find_latest_file("/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/")+".txt"
latest_file_path = os.path.join("/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/", last_file_name)
with open(latest_file_path, "r") as f:
	line = f.readline() # 读取第一行
	while line:
		line = line.replace(' \n', '').replace(' \r', '')
		data_list = line.split(" ")
		if " " in data_list:
			data_list.remove(" ")
		if '' in data_list:
			data_list.remove('')
		data_list = [float(item) for item in data_list]
		total_data_list.append(data_list)
		line = f.readline() # 读取下一行

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(r"model/model.pth").to(device)
model.eval()
for index, i in enumerate(total_data_list):
	motor_t.append(i[0])
	motor_real.append(i[1])
	motor_cmd.append(i[2])
	motro_expect.append(i[3])

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
  
def continus_predict():
	moving_win = 0

	if len(motor_real) >= input_size:
		current_index_0 = []
		current_index_1 = []
		current_index_2 = []
		current_index_3 = []
		current_index_4 = []

		for index, item in enumerate(motor_cmd):
			predict_y = []
			predict_x = []
			for j in range(5):
				predict_motor_cmd = motor_cmd[moving_win+j:moving_win+j+ 50]
				predict_motor_t = motor_t[moving_win+j:moving_win+j+ 50]
	
				if j == 0:
					predict_motor_real = motor_real[moving_win:moving_win+ 50]
					tmp_predict_motor_real = deepcopy(predict_motor_real)
				else:
					predict_motor_real = tmp_predict_motor_real[j:]
		
				end_time = motor_t[moving_win+j+ 50-1]
				start_time = motor_t[moving_win+j]
			
				new_timestamps = np.arange(end_time-1.0, end_time, 0.02)
				predict_motor_real = np.round(np.interp(new_timestamps, predict_motor_t, predict_motor_real),3)
				predict_motor_cmd = np.round(np.interp(new_timestamps, predict_motor_t, predict_motor_cmd),3)	

				imgs1 = torch.tensor(np.array(predict_motor_real)).to(device)
				imgs2 = torch.tensor(np.array(predict_motor_cmd)).to(device)
				images = imgs1.view(-1, 1, input_size)
				images2 = imgs2.view(-1, 1, input_size)
				outputs = model(images,images2)
				test_labels = outputs.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
				predict_x.append(index+ 50+j)
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
			ax.plot(x, motro_expect, linewidth = 3.0)  # 画出当前x列表和y列表中的值的图形

			x1 = np.linspace(input_size+10, input_size+len(current_index_0)-1+10, len(current_index_0))
			ax.plot(x1, current_index_0,color='c', label='当前帧')  # 画出当前x列表和y列表中的值的图形

			# plt.plot(x1, current_index_1)  # 画出当前x列表和y列表中的值的图形

			x3 = np.linspace(input_size+3+10, input_size+3+len(current_index_0)-1+10, len(current_index_0))
			ax.plot(x3, current_index_2,color='m', label='第三帧')  # 画出当前x列表和y列表中的值的图形

			# plt.plot(x1, current_index_3)  # 画出当前x列表和y列表中的值的图形

			x5 = np.linspace(input_size+5+10, input_size+5+len(current_index_0)-1+10, len(current_index_0))
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


def speed_control():
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
			predict_motor_real = np.round(np.interp(new_timestamps, predict_motor_t, predict_motor_real),3)
			predict_motor_cmd = np.round(np.interp(new_timestamps, predict_motor_t, predict_motor_cmd),3)	

			imgs1 = torch.tensor(np.array(predict_motor_real)).to(device)
			imgs2 = torch.tensor(np.array(predict_motor_cmd)).to(device)
			images = imgs1.view(-1, 1, input_size)
			images2 = imgs2.view(-1, 1, input_size)
			outputs = model(images,images2)
			test_labels = outputs.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
			current_index_2.append(predict_motor_cmd[-1])

			cnt = 0
			while abs(motro_expect[moving_win+25] - test_labels.item()) > 0.005:
				predict_motor_cmd[-1] += (motro_expect[moving_win+25] - test_labels.item())
				imgs2 = torch.tensor(np.array(predict_motor_cmd)).to(device)
				images2 = imgs2.view(-1, 1, input_size)
				outputs = model(images,images2)
				test_labels = outputs.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
				cnt+=1
				if cnt >=20:
					break
    
			predict_x.append(index+25)
			predict_y.append(np.round(test_labels.item(),3))


			y = np.round(test_labels.item(),3)
			current_index_0.append(y)
			current_index_1.append(predict_motor_cmd[-1])

		
			moving_win += 1
			x = np.linspace(0, len(motor_real), len(motor_real))

			# ax.plot(x, motor_real, linewidth = 3.0)  # 画出当前x列表和y列表中的值的图形
			ax.plot(x, motro_expect, linewidth = 3.0)  # 画出当前x列表和y列表中的值的图形

			x1 = np.linspace(input_size+10, input_size+len(current_index_0)-1+10, len(current_index_0))
			ax.plot(x1, current_index_0,color='c', label='当前帧')  # 画出当前x列表和y列表中的值的图形

			x2 = np.linspace(input_size+10, input_size+len(current_index_1)-1+10, len(current_index_1))
			ax.plot(x2, current_index_1,color='g', label='当前帧')  # 画出当前x列表和y列表中的值的图形
      
   
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


# 可视化数据集
def test():
	print(len(motor_real))
	x = np.linspace(0, len(motor_real), len(motor_real))
	ax.plot(x, motor_real, linewidth = 3.0,color='r')  # 画出当前x列表和y列表中的值的图形
	ax.plot(x, motor_cmd, linewidth = 3.0)  # 画出当前x列表和y列表中的值的图形
	ax.plot(x, motro_expect, linewidth = 3.0,color='b')  # 画出当前x列表和y列表中的值的图形
	plt.show()

def testSingle():
	predict_motor_real = [0.02800000086426735,0.027000000700354576,0.027000000700354576,0.026000000536441803,0.026000000536441803,0.02500000037252903,0.02500000037252903,0.024000000208616257,0.024000000208616257,0.023000000044703484,0.023000000044703484,0.02199999988079071,0.02199999988079071,0.020999999716877937,0.020999999716877937,0.019999999552965164,0.019999999552965164,0.01899999938905239,0.01899999938905239,0.017999999225139618,0.017999999225139618,0.017000000923871994,0.017000000923871994,0.01600000075995922,0.01600000075995922,0.014999999664723873,0.014999999664723873,0.014000000432133675,0.014000000432133675,0.013000000268220901]
	predict_motor_cmd = [0.02800000086426735,0.027000000700354576,0.027000000700354576,0.026000000536441803,0.026000000536441803,0.026000000536441803,0.02500000037252903,0.024000000208616257,0.024000000208616257,0.023000000044703484,0.023000000044703484,0.02199999988079071,0.02199999988079071,0.020999999716877937,0.020999999716877937,0.019999999552965164,0.019999999552965164,0.01899999938905239,0.01899999938905239,0.017999999225139618,0.017999999225139618,0.017000000923871994,0.017000000923871994,0.01600000075995922,0.01600000075995922,0.014999999664723873,0.014999999664723873,0.014000000432133675,0.014000000432133675,0.013000000268220901]
	imgs1 = torch.tensor(np.array(predict_motor_real)).to(device)
	imgs2 = torch.tensor(np.array(predict_motor_cmd)).to(device)
	images = imgs1.view(-1, 1, input_size)
	images2 = imgs2.view(-1, 1, input_size)
	outputs = model(images,images2)
	test_labels = outputs.unsqueeze(1)  # 将目标的形状从[2]变为[2, 1]
	print(np.round(test_labels.item(),3))
# testSingle()
# test()
continus_predict()