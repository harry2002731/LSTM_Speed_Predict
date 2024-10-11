from matplotlib import pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from LSTM import *
import pandas as pd
import shutil
from scipy import stats
from fileProcessor import *
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




def loadDataSet(file_path="", last_file=False):
    total_data_list = []
    if last_file:
        last_file_name = find_latest_file(
            "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/data_set/")+".txt"
        latest_file_path = os.path.join(
            "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/", last_file_name)
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


def continus_predict(motor_t, motor_real, motor_cmd, motor_expect, model, if_plot=False, if_difference=False):
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
                cur_expect_not_intep = motor_expect[moving_win +
                                                    j: cur_index + 1]
                cur_real_not_interp = motor_real[moving_win + j:cur_index + 1]

                cur_t = motor_t[cur_index]
                cur_t_interp = np.arange(cur_t-0.98, cur_t + 0.02, 0.02)
                # 当前帧(为了取到当前时间所以，所以调整了相加的时间)
                cur_cmd_t_interp = np.arange(cur_t-0.98, cur_t + 0.02, 0.02)

                if if_difference:
                    # 当前帧(为了取到当前时间所以，所以调整了相加的时间)
                    cur_cmd_t_interp = np.arange(cur_t-1.0, cur_t + 0.02, 0.02)

                cur_real_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_real_not_interp), 3)

                cur_expect_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_expect_not_intep), 3)

                cur_cmd_interp = np.round(
                    np.interp(cur_cmd_t_interp, cur_t_not_interp, cur_cmd_not_intep), 3)
                if if_difference:
                    if len(cur_cmd_interp) > len(cur_t_interp) + 1:
                        cur_cmd_interp = np.delete(cur_cmd_interp, 0)
                    cur_cmd_interp = np.round(np.diff(cur_cmd_interp), 3)

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
                outputs = model(images, images2, images3)
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
                ax.plot(x, motor_expect, linewidth=3.0)

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


def speed_control(motor_t, motor_real, motor_cmd, motor_expect, model, if_plot=False, if_difference=False):
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
                cur_expect_not_interp = motor_expect[moving_win +
                                                     j:cur_index + 1]

                cur_t = motor_t[cur_index]
                cur_t_interp = np.arange(cur_t-0.98, cur_t + 0.02, 0.02)
                # 当前帧(为了取到当前时间所以，所以调整了相加的时间)
                cur_cmd_t_interp = np.arange(cur_t-0.98, cur_t+0.02, 0.02)
                if if_difference:
                    # 当前帧(为了取到当前时间所以，所以调整了相加的时间)
                    cur_cmd_t_interp = np.arange(cur_t-1.02, cur_t-0.01, 0.02)

                cur_real_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_real_not_interp), 3)

                cur_cmd_interp = np.round(
                    np.interp(cur_cmd_t_interp, cur_t_not_interp, cur_cmd_not_intep), 3)

                cur_expect_interp = np.round(
                    np.interp(cur_t_interp, cur_t_not_interp, cur_expect_not_interp), 3)

                if if_difference:
                    if len(cur_cmd_interp) >= 50:
                        cur_cmd_interp = np.delete(cur_cmd_interp, 0)
                    cur_cmd_interp = np.round(np.diff(cur_cmd_interp), 3)
                    cur_cmd_interp = np.append(cur_cmd_interp, 0.0)

                # cur_real = cur_real_interp[-1]
                # cur_cmd_interp1 = np.append(cur_cmd_interp1,0.0)
                i = -0.15
                while i < 0.15:
                    cur_cmd_interp[-1] = i
                    imgs1 = torch.tensor(np.array(cur_real_interp)).to(device)
                    imgs2 = torch.tensor(np.array(cur_cmd_interp)).to(device)
                    imgs3 = torch.tensor(
                        np.array(cur_expect_interp)).to(device)
                    images = imgs1.view(-1, 1, input_size)
                    images2 = imgs2.view(-1, 1, input_size)
                    images3 = imgs3.view(-1, 1, input_size)
                    outputs = model(images, images2, images3)
                    test_labels = outputs.unsqueeze(1)
                    if if_difference:
                        print(
                            cur_real_not_interp[-1]+cur_cmd_interp[-1], np.round(test_labels.item(), 3))
                    else:
                        print(cur_cmd_interp[-1],
                              np.round(test_labels.item(), 3))
                    i += 0.01
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
            #     ax.plot(x, motor_expect, linewidth=3.0)

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
def visualizeDataset(dataset,title_name="", visual_indexs=[], compare_indexs=[], name_list=[]):
    global delete, close
    fig, ax = plt.subplots(figsize=(10, 8))
    calCorrelation(dataset[compare_indexs[0]], dataset[compare_indexs[1]])
    acc_list = calACC(dataset[compare_indexs[0]],
                      dataset[compare_indexs[1]])  # 计算重合度
    colors = ["xkcd:blue",
                "xkcd:light red",
                "xkcd:grass green",
                "xkcd:goldenrod",
                "xkcd:forest green",
                "xkcd:sky blue",
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


def group_data(data, step=1):
    if_find = False
    find_index = 0
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


def calACC(data1, data2, thres=0.01):
    index = group_data(data2, 5)
    d1 = np.split(data1, index)
    d2 = np.split(data2, index)
    index.append(len(data2))
    acc_list = []
    for i, d in enumerate(d1):
        minus_speed = abs(d1[i] - d2[i])
        arr = (minus_speed < thres).astype(int)
        ones_ratio = np.sum(arr) / arr.size
        acc_list.append([index[i], data2[index[i]-1], ones_ratio])
        # print("贴合度:",index[i], data2[index[i]-1], ones_ratio)
    return acc_list
# 计算数据的相关性


def calCorrelation(data1, data2, delete_zero=False):
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


def calPeason(data_t, data1, data2, ax, if_plot=False):
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
            plt.plot(x[:-res], y2[res:res+temp])  # 结论y1超前y2五个单位。将y1时间向前错位即可重合
            plt.plot(x, y2[30:], linewidth=3.8)

        elif res == 0:
            plt.plot(x, y1)
            plt.plot(x, y2[30:])
        elif res < 0:
            a = np.correlate(y1, y2, mode="same")
            res = len(a) // 2 - a.argmax()
            temp = len(x[:res])
            plt.plot(x, y1,)
            plt.plot(x[:res], y2[res:res+temp])  # 结论y1超前y2五个单位。将y1时间向前错位即可重合
            plt.plot(x, y2[30:], linewidth=3.8)

        plt.legend(['real', "cmd", "or", "oc"])
        plt.pause(0.001)
        ax.cla()  # 清除图形
    return res

def testSingleData(model):

    real_vector = [0.09000000357627869,0.08900000154972076,0.08900000154972076,0.08799999952316284,0.08699999749660492,0.0860000029206276,0.08500000089406967,0.08500000089406967,0.08299999684095383,0.0820000022649765,0.08100000023841858,0.07999999821186066,0.07900000363588333,0.07800000160932541,0.07699999958276749,0.07599999755620956,0.07500000298023224,0.07400000095367432,0.0729999989271164,0.07199999690055847,0.07000000029802322,0.0689999982714653,0.06800000369548798,0.06700000166893005,0.06599999964237213,0.06499999761581421,0.06499999761581421,0.06400000303983688,0.06300000101327896,0.06199999898672104,0.061000000685453415,0.05999999865889549,0.05900000035762787,0.057999998331069946,0.05700000002980232,0.0560000017285347,0.0560000017285347,0.054999999701976776,0.05400000140070915,0.05299999937415123,0.05299999937415123,0.05299999937415123,0.052000001072883606,0.050999999046325684,0.05000000074505806,0.04899999871850014,0.04800000041723251,0.04699999839067459,0.04699999839067459,0.04600000008940697]
    cmd_vector = [0.07199999690055847,0.07100000232458115,0.07100000232458115,0.07000000029802322,0.0689999982714653,0.0689999982714653,0.06800000369548798,0.06800000369548798,0.06700000166893005,0.06700000166893005,0.06599999964237213,0.06599999964237213,0.06499999761581421,0.06499999761581421,0.06499999761581421,0.06400000303983688,0.06400000303983688,0.06300000101327896,0.06300000101327896,0.06199999898672104,0.06199999898672104,0.061000000685453415,0.061000000685453415,0.05999999865889549,0.05999999865889549,0.05900000035762787,0.05900000035762787,0.057999998331069946,0.057999998331069946,0.057999998331069946,0.05700000002980232,0.05700000002980232,0.05700000002980232,0.05700000002980232,0.05700000002980232,0.05700000002980232,0.0560000017285347,0.0560000017285347,0.0560000017285347,0.054999999701976776,0.054999999701976776,0.05400000140070915,0.05299999937415123,0.052000001072883606,0.050999999046325684,0.05000000074505806,0.04899999871850014,0.04800000041723251,0.04800000041723251,0.044]
    expect_vector = [0.08399999886751175,0.08299999684095383,0.08100000023841858,0.07999999821186066,0.07900000363588333,0.07900000363588333,0.07800000160932541,0.07699999958276749,0.07599999755620956,0.07500000298023224,0.07400000095367432,0.07199999690055847,0.07100000232458115,0.07000000029802322,0.0689999982714653,0.0689999982714653,0.06800000369548798,0.06700000166893005,0.06599999964237213,0.06499999761581421,0.06300000101327896,0.06199999898672104,0.061000000685453415,0.05999999865889549,0.05900000035762787,0.057999998331069946,0.05700000002980232,0.0560000017285347,0.054999999701976776,0.05400000140070915,0.05299999937415123,0.052000001072883606,0.050999999046325684,0.04899999871850014,0.04800000041723251,0.04699999839067459,0.04600000008940697,0.04500000178813934,0.04399999976158142,0.041999999433755875,0.04100000113248825,0.04100000113248825,0.039000000804662704,0.03799999877810478,0.035999998450279236,0.03500000014901161,0.032999999821186066,0.03200000151991844,0.029999999329447746,0.02894810028374195]
    position_vector = [0.07100000232458115,0.0689999982714653,0.06700000166893005,0.06599999964237213,0.06400000303983688,0.06199999898672104,0.05999999865889549,0.05900000035762787,0.05700000002980232,0.0560000017285347,0.05400000140070915,0.05299999937415123,0.050999999046325684,0.05000000074505806,0.04800000041723251,0.04699999839067459,0.04600000008940697,0.04399999976158142,0.0430000014603138,0.041999999433755875,0.04100000113248825,0.039000000804662704,0.03799999877810478,0.03700000047683716,0.035999998450279236,0.03500000014901161,0.032999999821186066,0.03200000151991844,0.029999999329447746,0.028999999165534973,0.02800000086426735,0.027000000700354576,0.026000000536441803,0.02500000037252903,0.024000000208616257,0.023000000044703484,0.02199999988079071,0.020999999716877937,0.019999999552965164,0.01899999938905239,0.017000000923871994,0.01600000075995922,0.014999999664723873,0.014000000432133675,0.013000000268220901,0.013000000268220901,0.012000000104308128,0.010999999940395355,0.009999999776482582,0.008999999612569809]
    speed_gap_vector = [-0.006000004708766937,-0.006000004708766937,-0.008000001311302185,-0.008000001311302185,-0.007999993860721588,-0.006999999284744263,-0.006999999284744263,-0.008000001311302185,-0.006999999284744263,-0.006999999284744263,-0.006999999284744263,-0.008000001311302185,-0.008000001311302185,-0.008000001311302185,-0.008000001311302185,-0.006999999284744263,-0.006999999284744263,-0.006999999284744263,-0.006999999284744263,-0.006999999284744263,-0.006999999284744263,-0.006999999284744263,-0.007000003010034561,-0.007000003010034561,-0.006999999284744263,-0.006999999284744263,-0.007999997586011887,-0.008000001311302185,-0.008000001311302185,-0.007999997586011887,-0.008000001311302185,-0.007999997586011887,-0.008000001311302185,-0.008999999612569809,-0.008999999612569809,-0.009000003337860107,-0.010000001639127731,-0.009999997913837433,-0.010000001639127731,-0.010999999940395355,-0.011999998241662979,-0.011999998241662979,-0.013000000268220901,-0.013000000268220901,-0.014000002294778824,-0.013999998569488525,-0.015000000596046448,-0.01499999687075615,-0.016999999061226845,-0.017051899805665016]



    # predict_motor_real = [0.04500000178813934,0.041999999433755875,0.03999999910593033,0.03799999877810478,0.035999998450279236,0.03799999877810478,0.03999999910593033,0.0430000014603138,0.03999999910593033,0.03799999877810478,0.035999998450279236,0.039000000804662704,0.04100000113248825,0.04399999976158142,0.04600000008940697,0.04800000041723251,0.05000000074505806,0.05299999937415123,0.05299999937415123,0.05000000074505806,0.04699999839067459,0.04399999976158142,0.041999999433755875,0.03999999910593033,0.03700000047683716,0.03799999877810478,0.03999999910593033,0.041999999433755875,0.04399999976158142,0.04699999839067459,0.04600000008940697,0.04399999976158142,0.041999999433755875,0.03999999910593033,0.03700000047683716,0.03799999877810478,0.03999999910593033,0.041999999433755875,0.04399999976158142,0.04699999839067459,0.05000000074505806,0.052000001072883606,0.04899999871850014,0.04600000008940697,0.04399999976158142,0.04399999976158142,0.03999999910593033,0.03700000047683716,0.03500000014901161,0.03799999877810478]
    # predict_motor_cmd = [-0.001999996602535248,-0.0030000023543834686,-0.0020000003278255463,-0.0020000003278255463,-0.0020000003278255463,0.0020000003278255463,0.0020000003278255463,0.0030000023543834686,-0.0030000023543834686,-0.0020000003278255463,-0.0020000003278255463,0.0020000003278255463,0.0030000023543834686,0.00299999862909317,0.0020000003278255463,0.0020000003278255463,0.0020000003278255463,0.00299999862909317,0.0,-0.00299999862909317,-0.0030000023543834686,-0.00299999862909317,-0.0020000003278255463,-0.0020000003278255463,-0.00299999862909317,0.000999998301267624,0.0020000003278255463,0.0020000003278255463,0.0020000003278255463,0.00299999862909317,-0.000999998301267624,-0.0020000003278255463,-0.0020000003278255463,-0.0020000003278255463,-0.00299999862909317,0.000999998301267624,0.0020000003278255463,0.0020000003278255463,0.0020000003278255463,0.00299999862909317,0.0030000023543834686,0.0020000003278255463,-0.0020000003278255463,-0.0040000006556510925,-0.0020000003278255463,-0.0020000003278255463,-0.0020000003278255463,-0.00299999862909317,-0.0020000003278255463,0.02]

    imgs1 = torch.tensor(np.array(real_vector)).to(device)
    imgs2 = torch.tensor(np.array(cmd_vector)).to(device)
    imgs3 = torch.tensor(np.array(position_vector)/1.4).to(device)
    print(imgs3)
    images = imgs1.view(-1, 1, input_size)
    images2 = imgs2.view(-1, 1, input_size)
    images3 = imgs3.view(-1, 1, input_size)
    outputs = model(images, images2,images3)
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
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = torch.load(r"model/model.pth").to(device)
    # model.eval()
    # testSingleData(model)
    
    # # 查看所有数据
    orin_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/"
    for root, dirs, files in os.walk(orin_path):
        global file_name
        for file_name in files:
            print(file_name)
            motor_info = ["motor_t", "motor_real", "motor_cmd", "motor_expect", "motor_height_gap", "motor_speed_gap","orin_cmd","predict_expect","real_gap","cmd_gap","expect_gap"]
            # dataset = loadDataSet(orin_path+file_name, True)
            # dataset = loadDataSet("/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/robokit_2024-10-10_15-32-11.2.log.txt", False) #无载重
            # dataset = loadDataSet("/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/robokit_2024-09-27_15-48-20.2.log.txt", False) #有载重
            # dataset = loadDataSet("/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/robokit_2024-09-29_16-38-14.2.log.txt", False) #原始
            
            dataset = loadDataSet("/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/robokit_2024-10-10_15-32-11.2.log.txt", False)
            # real_ = np.diff(np.array(dataset[1]))
            # real_ = np.insert(real_,0,0)
            # dataset = np.insert(dataset,8,real_,axis = 0)
            
            # cmd_ = np.diff(np.array(dataset[2]))
            # cmd_ = np.insert(cmd_,0,0)
            # dataset = np.insert(dataset,9,cmd_,axis = 0)
            
            # expect_ = np.diff(np.array(dataset[3]))
            # expect_ = np.insert(expect_,0,0)
            # dataset = np.insert(dataset,10,expect_,axis = 0)

            real_ = dataset[1] - dataset[3]
            dataset = np.insert(dataset,8,real_,axis = 0)
            
            visual_indexs = [1,2,3,7]
            visualizeDataset(dataset, title_name = file_name, visual_indexs=visual_indexs, compare_indexs=[1,3], name_list=[motor_info[i] for i in visual_indexs])
            print("****************************************************")





    # motor_t, motor_real, motor_cmd, motor_expect = loadDataSet(
    #     "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/robokit_2024-09-02_17-46-38.1.log.txt", last_file=False)
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
    #                 cur_expect_not_interp = motor_expect[moving_win + j:cur_index + 1]
    #                 res = calPeason(cur_t_not_interp,cur_real_not_interp,cur_cmd_not_intep,cur_expect_not_interp,ax)
    #                 if res != 0:
    #                     aaa.append(res)
    #                 moving_win += 1
    #         except Exception as expection:
    #             print(Exception)
    # print(np.mean(np.array(aaa)))
    # visualizeDataset(motor_t, motor_real, motor_cmd, motor_expect)
    # calACC(motor_real,motor_cmd,motor_expect)
    # testSingleData(model)

    # speed_control(motor_t,motor_real,motor_cmd,motor_expect,model,False,False)
    # calPeason(motor_t,motor_real,motor_cmd,motor_expect)
    # continus_predict(motor_t,motor_real,motor_cmd,motor_expect,model,False,False)
