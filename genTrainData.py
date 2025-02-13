import random
import numpy as np
from BasicFunc import *
# 时间戳
class TimeInfo:
    def __init__(self) -> None:
        self.date = ""  # 日期
        self.time = ""  # 时间
        self.frame_num = ""  # 数据id

# 泵控电机真实值
class MotorReal:
    def __init__(self) -> None:
        self.time_info = TimeInfo()
        self.odom_t = ""
        self.loc = ""
        self.speed = ""
        self.current = ""
        self.voltage = ""
        self.stop = ""
        self.error = ""
        self.stop = ""
        self.degree = ""
        self.encoder = ""
        self.error_code = ""
        self.ori_location = ""

    def decode(self, data_str):
        str_ = ""
        index = 0
        record = False
        for i in data_str:
            if i == "[":
                record = True
                str_ = ""
                continue
            elif i == "]":
                record = False
                if index == 0:
                    time_info = str_.split(" ")
                    self.time_info.date = time_info[0]
                    self.time_info.time = float(time_info[1])
                elif index == 5:
                    motor_real = str_.split("|")
                    self.odom_t = motor_real[0]
                    self.loc = motor_real[1]
                    self.speed = float(motor_real[2])
                    self.current = motor_real[3]
                    self.voltage = motor_real[4]
                    self.stop = motor_real[5]
                    self.error = motor_real[6]
                    self.stop = motor_real[7]
                    self.degree = motor_real[8]
                    self.encoder = motor_real[9]
                    self.error_code = motor_real[10]
                    self.ori_location = motor_real[11]
                index += 1
            if record:
                str_ += i

# 泵控电机控制指令
class MotorCmd:
    def __init__(self) -> None:
        self.t = ""
        self.motor = ""
        self.motor1 = ""
        self.motor2 = ""
        self.time_info = TimeInfo()

    def decode(self, data_str):
        str_ = ""
        index = 0
        record = False
        for i in data_str:
            if i == "[":
                record = True
                str_ = ""
                continue
            elif i == "]":
                record = False
                if index == 0:
                    time_info = str_.split(" ")
                    self.time_info.date = time_info[0]
                    self.time_info.time = float(time_info[1])
                elif index == 5:
                    motor_real = str_.split("|")
                    self.t = motor_real[0]
                    self.motor = float(motor_real[1])
                    self.motor1 = float(motor_real[3])
                    self.motor2 = float(motor_real[5])
                index += 1
            if record:
                str_ += i

# 泵控电机期望指令
class MotorExpect:
    def __init__(self) -> None:
        self.name = ""
        self.speed = ""
        self.position = ""
        self.value = ""
        self.goal = ""
        self.reachGoal = ""
        self.time_info = TimeInfo()

    def decode(self, data_str):
        str_ = ""
        index = 0
        record = False
        for i in data_str:
            if i == "[":
                record = True
                str_ = ""
                continue
            elif i == "]":
                record = False
                if index == 0:
                    time_info = str_.split(" ")
                    self.time_info.date = time_info[0]
                    self.time_info.time = float(time_info[1])
                elif index == 6:
                    tmp = str_.split("|")
                    self.name = tmp[0]
                    self.speed = float(tmp[1])
                    self.position = float(tmp[2])
                    self.value = float(tmp[3])
                    self.goal = float(tmp[4])
                    self.reachGoal = float(tmp[5])
                index += 1
            if record:
                str_ += i

# 泵控电机位置
class ForkPosition:
    def __init__(self) -> None:
        self.time_info = TimeInfo()
        self.odom_t = ""
        self.loc = ""
        self.speed = ""
        self.current = ""
        self.voltage = ""
        self.stop = ""
        self.error = ""
        self.stop = ""
        self.degree = ""
        self.encoder = ""
        self.error_code = ""
        self.ori_location = ""

    def decode(self, data_str):
        str_ = ""
        index = 0
        record = False
        for i in data_str:
            if i == "[":
                record = True
                str_ = ""
                continue
            elif i == "]":
                record = False
                if index == 0:
                    time_info = str_.split(" ")
                    self.time_info.date = time_info[0]
                    self.time_info.time = float(time_info[1])
                elif index == 6:
                    fork_position = str_.split("|")
                    self.motor_name = fork_position[0]
                    self.speed = float(fork_position[1])
                    self.cur_pos = float(fork_position[2])
                    self.tar_pos = float(fork_position[3])
                index += 1
            if record:
                str_ += i
    def getNeedStr(self):
        return str(self.time_info.time)+" "+str(self.crandom_stateur_pos)+" "+str(self.tar_pos)+"\n"


# 泵控电机期望指令
class MotorTotal:
    def __init__(self,weight,temp_real=None) -> None:
        self.a = ""
        self.b = ""
        self.real = ""
        self.cmd = ""
        self.expect = ""
        self.accuracy = ""
        self.search_count = ""
        self.time_info = TimeInfo()
        self.data_len = 0
        self.weight = weight
        self.temp_real = temp_real
    
    def decode(self, data_str):
        str_ = ""
        index = 0
        record = False
        for i in data_str:
            if i == "[":
                record = True
                str_ = ""
                continue
            elif i == "]":
                record = False
                if index == 0:
                    time_info = str_.split(" ")
                    self.time_info.date = time_info[0]
                    self.time_info.time = float(time_info[1])
                elif index == 5:
                    tmp = str_.split("|")
                    # self.name = tmp[0]
                    # self.speed = float(tmp[1])
                    # self.position = float(tmp[2])
                    self.real = float(tmp[3])
                    # self.real = self.temp_real
                    self.cmd = float(tmp[4])
                    self.expect = float(tmp[5])
                    self.height_gap = float(tmp[6])
                    self.data_len = len(tmp)
                    if  self.data_len >8:
                        self.orin_cmd = float(tmp[7])
                        self.predict_expect = float(tmp[8])
                        if self.data_len > 9:
                            self.predict_weight_index = float(tmp[9])
                    self.speed_gap = float(self.expect - self.real)
                index += 1
            if record:
                str_ += i
    def getNeedStr(self):
        if self.data_len > 8:
            data_string = " ".join([
                str(self.time_info.time),
                str(self.real),
                str(self.cmd),
                str(self.expect),
                str(self.height_gap),
                str(self.speed_gap),
                str(self.orin_cmd),
                str(self.predict_expect),
                str(self.weight)
            ]) + "\n"
        elif self.data_len > 9:
            data_string = " ".join([
                str(self.time_info.time),
                str(self.real),
                str(self.cmd),
                str(self.expect),
                str(self.height_gap),
                str(self.speed_gap),
                str(self.orin_cmd),
                str(self.predict_expect),
                str(self.weight),
                str(self.predict_weight_index)
            ]) + "\n"
        else:
            data_string = str(self.time_info.time)+" "+str(self.real)+" "+str(self.cmd)+" "+str(self.expect)+" "+str(self.height_gap)+" "+str(self.speed_gap)+"\n"

        return data_string

# def decodeDataList(path, data_type,weight = ""):
#     data = []
#     with open(path, "r") as file:
#         lines = file.readlines()
#         temp_real = -1
#         for line in lines:
#             if "MotorInfo2" in line:
#                 temp_real = round(float(line.split("|")[2]),3)
#                 continue
#             if temp_real != -1:
#                 if weight != "":
#                     m = data_type(weight,temp_real)
#                     temp_real = -1
#                 else:
#                     m = data_type()
#             else:
#                 continue
#             m.decode(line)
#             data.append(m)
#     return data
def decodeDataList(path, data_type,weight = ""):
    data = []
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if weight != "":
                m = data_type(weight)
            else:
                m = data_type()

            m.decode(line)
            data.append(m)
    return data

class MotorDataProcess:
    def __init__(self, data) -> None:
        self.repetition_rate = 0
        self.data = data
        self.timestamps, self.speeds = np.array(data).T
        self.use = True

    def judgeTimeStamp(self, time_gap):
        start_time = self.timestamps.min()
        end_time = self.timestamps.max()
        if end_time - start_time < time_gap:
            return False
        else:
            return True

    def calRepRate(self):
        self.repetition_rate = (len(self.data) - len(set(self.data))) / len(self.data)
        return self.repetition_rate

    def filter(self):
        if self.calRepRate(self.data) > 0.99:
            self.use = False
        if len(self.data) < 30:
            self.use = False
        return self.use

    def interp(self, time_span, time_gap):
        start_time = self.timestamps.min()
        end_time = self.timestamps.max()
        new_timestamps = np.arange(end_time - time_span, end_time, time_gap)

        # 使用线性插值计算新时间点的速度
        new_speeds = np.interp(new_timestamps, self.timestamps, self.speeds)
        new_speeds_rounded = np.round(new_speeds, 3)
        new_timestamps_rounded = np.round(new_timestamps, 3)
        temp = np.stack([new_timestamps_rounded, new_speeds_rounded])
        self.data = np.transpose(temp, (1, 0))


# 滤除文件内的[]和，
def formatFile(path):
    with open(path, "r") as file:
        lines = file.readlines()
        with open(r"data\train.txt", "a") as write_file:
            write_file.truncate(0)  # 清除文件内容
            for line in lines:
                line = line.replace("[", "").replace("]", "").replace(",", "")
                write_file.write(str(line))


# 数据时间戳匹配
def matchOrinData(motor_reals, motor_cmds, motor_expects, path, debug=False):
    threshold = 0.1
    real_data = np.array([d.time_info.time for d in motor_reals])
    cmd_data = np.array([d.time_info.time for d in motor_cmds])
    expect_data = np.array([d.time_info.time for d in motor_expects])

    real_data_speed = np.array([d.speed for d in motor_reals])
    cmd_data_speed = np.array([d.motor2 for d in motor_cmds])
    expect_data_speed = np.array([d.speed for d in motor_expects])

    len1, len2, len3 = len(real_data), len(cmd_data), len(expect_data)
    with open(path, "a+") as file:
        file.truncate(0)
        cmd_temp = []
        real_temp = []
        expect_temp = []
        matches = []

        idx1, idx2, idx3 = 0, 0, 0

        # 同时遍历三个数组
        while idx1 < len1 and idx2 < len2 and idx3 < len3 - 2:
            timestamp1 = real_data[idx1]
            # 使用二分查找在array2和array3中查找接近的时间戳
            left2 = np.searchsorted(cmd_data, timestamp1 - threshold, side="left")
            right2 = np.searchsorted(cmd_data, timestamp1 + threshold, side="right")
            left3 = np.searchsorted(expect_data, timestamp1 - threshold, side="left")
            right3 = np.searchsorted(expect_data, timestamp1 + threshold, side="right")

            # 检查索引是否有效
            if left2 < right2 and left3 < right3:
                file.write(
                    str(timestamp1)
                    + " "
                    + str(round(real_data_speed[idx1], 3))
                    + " "
                    + str(round(cmd_data_speed[left2], 3))
                    + " "
                    + str(round(expect_data_speed[left3], 3))
                    + " "
                )
                file.write("\n")

                if debug:
                    matches.append((timestamp1, cmd_data[left2], expect_data[left3]))
                    real_temp.append([timestamp1, round(real_data_speed[idx1], 3)])
                    cmd_temp.append([cmd_data[left2], round(cmd_data_speed[left2], 3)])
                    expect_temp.append(
                        [expect_data[left3], round(expect_data_speed[left3], 3)]
                    )
            # 更新索引
            idx1 += 1
            idx2 = max(idx2, left2)
            idx3 = max(idx3, left3)


# 数据时间戳匹配
def matchForkData(motor_expects , postion_cur, postion_tar, path, debug=False):
    threshold = 0.1
    expect_data = np.array([d.time_info.time for d in motor_expects])
    cur_data = np.array([d.time_info.time for d in postion_cur])
    tar_data = np.array([d.time_info.time for d in postion_tar])

    expect_data_speed = np.array([d.expect for d in motor_expects])
    real_data_speed = np.array([d.real for d in motor_expects])
    cmd_data_speed = np.array([d.cmd for d in motor_expects])
    
    cur_data_speed = np.array([d.cur_pos for d in postion_cur])
    tar_data_speed = np.array([d.tar_pos for d in postion_tar])

    len1, len2, len3 = len(expect_data) , len(cur_data), len(tar_data)
    with open(path, "a+") as file:
        file.truncate(0)
        idx1, idx2, idx3 = 0, 0, 0

        # 同时遍历三个数组
        while idx1 < len1 and idx2 < len2 and idx3 < len3 - 2:
            timestamp1 = expect_data[idx1]
            # 使用二分查找在array2和array3中查找接近的时间戳
            left2 = np.searchsorted(cur_data, timestamp1 - threshold, side="left")
            right2 = np.searchsorted(cur_data, timestamp1 + threshold, side="right")
            left3 = np.searchsorted(tar_data, timestamp1 - threshold, side="left")
            right3 = np.searchsorted(tar_data, timestamp1 + threshold, side="right")

            # 检查索引是否有效
            if left2 < right2 and left3 < right3:
                file.write(
                    str(timestamp1)
                    + " "
                    + str(round(real_data_speed[idx1], 3))
                    + " "
                    + str(round(cmd_data_speed[idx1], 3))
                    + " "
                    + str(round(tar_data_speed[left3] - cur_data_speed[left2], 3))
                    + " "
                )
                file.write("\n")
            # 更新索引
            idx1 += 1
            idx2 = max(idx2, left2)
            idx3 = max(idx3, left3)

def readFile(path):
    with open(path, "r") as file:
        lines = file.readlines()
        data_list = []
        for line in lines:
            line = line.replace("\n", "").replace(" \r", "")
            line = line.split(" ")
            while " " in line:
                line.remove(" ")
            while "" in line:
                line.remove("")
            data_list.append([float(item) for item in line])
    return data_list

def writeFile(path,data_list):
    delete_start_zero = True # 用于删除数据开始时候的0数据
    with open(path, "a") as file:
        file.truncate(0)
        for index, line in enumerate(data_list):
            # if delete_start_zero :
            #     if data_list[index].real == 0 and data_list[index + 10].real != 0:
            #         delete_start_zero = False
            # elif not delete_start_zero:
            file.write(line.getNeedStr())

def calRepetitionRate(d):
    return (len(d) - len(set(d))) / len(d)
    
def generateTrainData(
    write_path,
    read_path,
    time_gap,
    interp_interval=0.08,
    repetition_rate=0.95,
    need_input_data = [],
    need_output_data = [],
    multiStepOutput = [False, [-5,-3,-1] ],
    interp=False, # 是否时间间隔插值，确保时间步的统一
    difference = False, # 是否前后差值计算
    save_file = True # 是否保存训练文件
):
    aaa = []
    with open(write_path, "a") as write_file:
        file_name = write_path.split("/")[-1]
        if save_file:
            write_file.truncate(0)

        data_list = readFile(read_path)

        first_t = data_list[0][0]
        for index, data in enumerate(data_list):
            if data[0] - first_t >= time_gap:
                start_index = index + 1
                break

        len_list = []
        for i in range(start_index, len(data_list)):
            file_name = write_path.split("/")[-1]
            t_temp, real_temp, cmd_temp, expect_temp, height_gap_temp,speed_gap_temp, predict_expect, weight_temp = [], [], [], [],[],[],[],[]

            for item in data_list[0 : i + 1]:
                # if sixty_to_decimal(data_list[i][0]) - sixty_to_decimal(item[0]) > time_gap:
                #     print(data_list[i][0],item[0], sixty_to_decimal(data_list[i][0]) - sixty_to_decimal(item[0]))
                if 0.0 <= sixty_to_decimal(data_list[i][0]) - sixty_to_decimal(item[0]) <= time_gap:
                    t_temp.append(sixty_to_decimal(item[0]))  # 
                    # t_temp.append(item[0])  # 
                    real_temp.append(item[1])
                    cmd_temp.append(item[2])
                    expect_temp.append(item[3])
                    height_gap_temp.append(item[4]/1.4) # 归一化
                    speed_gap_temp.append(item[5])
                    predict_expect.append(item[6])
                    weight_temp.append(item[-1])

            if (len(real_temp) > 0  and (calRepetitionRate(real_temp) < repetition_rate or calRepetitionRate(cmd_temp) < repetition_rate)):
                if timeGapDistribution(t_temp) < 0.08:
                # if True:
                    start_time = np.array(t_temp).min()
                    end_time = np.array(t_temp).max()
                    if end_time - start_time > time_gap * 0.8 and len(real_temp) >= 5:
                        end_time *= 1000
                        time_gap *= 1000
                        interp_interval *= 1000
                        
                        # 数据时间间隔插值
                        if interp:
                            new_timestamps = np.arange(
                                end_time - time_gap + interp_interval,
                                end_time + interp_interval,
                                interp_interval,
                            )  # 取不到end_time

                            
                            expect_temp = np.round(
                                np.interp(new_timestamps/1000, t_temp, expect_temp), 3
                            )
                            height_gap_temp = np.round(
                                np.interp(new_timestamps/1000, t_temp, height_gap_temp), 3
                            )
                            speed_gap_temp = np.round(
                                np.interp(new_timestamps/1000, t_temp, speed_gap_temp), 3
                            )
                            predict_expect = np.round(
                                np.interp(new_timestamps/1000, t_temp, predict_expect), 3
                            )
                            if not difference:
                                cmd_temp = np.round(
                                    np.interp(new_timestamps/1000, t_temp, cmd_temp), 3
                                )
                                cmd_label = formatFloat(cmd_temp[-1])
                            else:

                                new_cmd_timestamps = np.arange(
                                    end_time - time_gap - interp_interval,
                                    end_time,
                                    interp_interval,
                                ) 
                                cmd_temp = np.round(np.interp(new_cmd_timestamps/1000, t_temp, cmd_temp), 3)
                                cmd_label = formatFloat(cmd_temp[-1])

                                cmd_temp = np.round(np.diff(cmd_temp),3)
                                # print(cmd_temp,cmd_temp.shape)
                                # print(real_temp,real_temp.shape)
                                # print("************************************")
                        end_time /= 1000
                        time_gap /= 1000
                        interp_interval /= 1000
                        
                        if not multiStepOutput[0]:
                            real_label = formatFloat(real_temp[-1])
                            expect_label = formatFloat(expect_temp[-1])
                            height_gap_label = formatFloat(height_gap_temp[-1])
                            speed_gap_label = formatFloat(speed_gap_temp[-1])
                            weght_label = weight_temp[-1]
                        else:
                            real_label = [(formatFloat(element)) for element in real_temp[multiStepOutput[-1]]]
                            expect_label = [(formatFloat(element)) for element in expect_temp[multiStepOutput[-1]]]
                            height_gap_label = [(formatFloat(element)) for element in height_gap_temp[multiStepOutput[-1]]]
                            speed_gap_label = [(formatFloat(element)) for element in speed_gap_temp[multiStepOutput[-1]]]
                            cmd_label = [(formatFloat(element)) for element in cmd_temp[multiStepOutput[-1]]]
                            weght_label = [weight_temp[-1] for element in cmd_temp[multiStepOutput[-1]]]

                        # end_index = int((time_gap - 1.0)/interp_interval) + 1
                        end_index = int((time_gap - 1.0)/interp_interval)
                        
                        # 判断原始模型预测结果是否准确，若不准确则加入到训练数据中
                        # if (abs(predict_expect[-end_index]) - abs(predict_expect[-1]) > 0.008):

                        len_list.append(len(real_temp))
                        real_temp = " ".join(str(formatFloat(element)) for element in real_temp[0:-end_index])
                        cmd_temp = " ".join(str(formatFloat(element))  for element in cmd_temp[0:-end_index])
                        expect_temp = " ".join(str(formatFloat(element))  for element in expect_temp[0:-end_index])
                        height_gap_temp = " ".join(str(formatFloat(element))  for element in height_gap_temp[0:-end_index])
                        speed_gap_temp = " ".join(str(formatFloat(element))  for element in speed_gap_temp[0:-end_index])
                        # height_gap_temp = " ".join(str(formatFloat(element))  for element in height_gap_temp[end_index:])
                        # speed_gap_temp = " ".join(str(formatFloat(element))  for element in speed_gap_temp[end_index:])

                        data_dict = {"real_temp":real_temp,"cmd_temp":cmd_temp,"expect_temp":expect_temp,"height_gap_temp":height_gap_temp,"speed_gap_temp":speed_gap_temp}
                        label_dict = {"real_label":real_label,"cmd_label":cmd_label,"expect_label":expect_label,"height_gap_label":height_gap_label,"speed_gap_label":speed_gap_label,"weight_label":weght_label}
                        tmp_data = []
                        for label in need_output_data:
                            if  multiStepOutput[0]:
                                for _ in label_dict[label]:
                                    tmp_data.append(_)
                            else:
                                tmp_data.append(label_dict[label])
                            
                        for data in need_input_data:
                            tmp_data.append(data_dict[data])
                        aaa.append(tmp_data)
                        if save_file:
                            all_repetition = (len(real_temp) - len(set(real_temp))) / len(real_temp) < repetition_rate # 重复率
                            zero_repetition = len(list(filter(lambda x: x != 0, real_temp)))/ len(real_temp) < repetition_rate # 非0值率
                            # if (not (all_repetition) and not(zero_repetition)):
                            #     continue
                                # file_name += ".txt"
                            # write_file.write(str(file_name) + " " )
                            # write_file.write(f'{str(file_name)}|s|{new_timestamps[0]/1000:1.3f}|e|{new_timestamps[-(end_index+1)]/1000:1.3f}|l|{new_timestamps[-1]/1000:1.3f} ')
                            write_file.write(f'{str(file_name)}|s|{decimal_to_sixty(new_timestamps[0]/1000):1.3f}|e|{decimal_to_sixty(new_timestamps[-(end_index+1)]/1000):1.3f}|l|{decimal_to_sixty(new_timestamps[-1]/1000):1.3f} ')
                            for item_ in tmp_data:
                                write_file.write(str(item_) + " | ")
                            write_file.write("\n")

        # print(len_list)
        return aaa


def sixty_to_decimal(sixty_num):
    decimal_tens = (sixty_num // 10) % 10
    decimal_ones = sixty_num % 10
    other = sixty_num // 100
    return other *60 + decimal_tens *10 + decimal_ones

def decimal_to_sixty(decimal_num):
    return decimal_num//60*100 +decimal_num%60

 
def generateDRLTestData(
    write_path,
    read_path,
    interp_interval=0.08,
    interp=False, # 是否时间间隔插值，确保时间步的统一
    save_file = True # 是否保存文件
):
    aaa = []
    with open(write_path, "a") as write_file:
        file_name = write_path.split("/")[-1]
        if save_file:
            write_file.truncate(0)

        data_list = readFile(read_path)

        len_list = []
        file_name = write_path.split("/")[-1]
        t_temp, real_temp, cmd_temp, expect_temp, height_gap_temp,speed_gap_temp, predict_expect, weight_temp = [], [], [], [],[],[],[],[]
        
        record = False
        for index, item in enumerate(data_list[0 : -2]):
            if data_list[index+2][2] != 0.0 and not record:
                record =True
                continue
            elif data_list[index+2][2] == 0.0 and record:
                record =False
                break
            if record:
                t_temp.append(sixty_to_decimal(item[0]))
                real_temp.append(item[1])
                cmd_temp.append(item[2])
                expect_temp.append(item[3])
                height_gap_temp.append(item[4]/1.4) # 归一化
                speed_gap_temp.append(item[5])
                predict_expect.append(item[6])
                weight_temp.append(item[-1])

        if (len(real_temp) > 0):
            start_time = np.array(t_temp).min()
            end_time = np.array(t_temp).max()
            start_time *= 1000
            end_time *= 1000
            interp_interval *= 1000
            
            # 数据时间间隔插值
            if interp:
                new_timestamps = np.arange(
                    start_time + interp_interval,
                    end_time + interp_interval,
                    interp_interval,
                )  # 取不到end_time

                    
                real_temp = np.round(
                    np.interp(new_timestamps/1000, t_temp, real_temp), 3
                )

                cmd_temp = np.round(
                    np.interp(new_timestamps/1000, t_temp, cmd_temp), 3
                )
                height_temp = np.round(
                    np.interp(new_timestamps/1000, t_temp, height_gap_temp), 3
                )
                end_time /= 1000
                interp_interval /= 1000

            print(new_timestamps.shape,real_temp.shape)
            len_list.append(len(real_temp))
            # real_temp = " ".join(str(formatFloat(element)) for element in real_temp)
            # cmd_temp = " ".join(str(formatFloat(element))  for element in cmd_temp)
            
                
            if save_file:
                for index,t in enumerate(new_timestamps):
                    write_file.write(f'{str(formatFloat(decimal_to_sixty(t/1000)))} | {str(real_temp[index])} | {str(cmd_temp[index])} | {str(height_gap_temp[2])}')
                    write_file.write("\n")

        # print(len_list)
        return aaa

def mergeData(write_path, read_path):
    # 读取源文件的全部内容
    with open(read_path, "r", encoding="utf-8") as source_file:
        content = source_file.read()
    # 将读取的内容写入目标文件
    with open(write_path, "a", encoding="utf-8") as target_file:
        target_file.write(content)

        # print("finsh")


