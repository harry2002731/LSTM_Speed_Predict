import numpy as np

class TimeInfo():
    def __init__(self) -> None:
        self.date = ""  # 日期
        self.time = ""  # 时间
        self.frame_num = ""  # 数据id

# 泵控电机真实值
class MotorReal():
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
class MotorCmd():
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
class MotorExpect():
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


def decodeMotorSpeeed(path, data_type):
    data = []
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            m = data_type()
            m.decode(line)
            data.append(m)
    return data


class MotorDataProcess():
    def __init__(self, data) -> None:
        self.repetition_rate = 0
        self.data = data
        self.timestamps, self.speeds = np.array(data).T
        self.use = True

    def judgeTimeStamp(self, time_gap):
        start_time = self.timestamps.min()
        end_time = self.timestamps.max()
        if end_time-start_time < time_gap:
            return False
        else:
            return True

    def calRepRate(self):
        self.repetition_rate = (
            len(self.data) - len(set(self.data)))/len(self.data)
        return self.repetition_rate

    def filter(self):
        if self.calRepRate(self.data) > 0.999:
            self.use = False
        if self.calRepRate(self.data) > 0.999:
            self.use = False
        if len(self.data) < 30:
            self.use = False
        return self.use

    def interp(self, time_span, time_gap):
        start_time = self.timestamps.min()
        end_time = self.timestamps.max()
        new_timestamps = np.arange(end_time-time_span, end_time, time_gap)

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
                line = line.replace('[', '').replace(']', '').replace(',', '')
                write_file.write(str(line))

# 数据时间戳匹配


def matchData(motor_reals, motor_cmds, motor_expects, path, debug=False):
    threshold = 0.05
    real_data = np.array([d.time_info.time for d in motor_reals])
    cmd_data = np.array([d.time_info.time for d in motor_cmds])
    expect_data = np.array([d.time_info.time for d in motor_expects])

    real_data_speed = np.array([d.speed for d in motor_reals])
    cmd_data_speed = np.array([d.motor2 for d in motor_cmds])
    expect_data_speed = np.array([d.speed for d in motor_expects])

    len1, len2, len3 = len(real_data), len(cmd_data), len(expect_data)
    with open(path, "a+") as file:
        cmd_temp = []
        real_temp = []
        expect_temp = []
        matches = []

        idx1, idx2, idx3 = 0, 0, 0

        # 同时遍历三个数组
        while idx1 < len1 and idx2 < len2 and idx3 < len3:
            timestamp1 = real_data[idx1]
        # 使用二分查找在array2和array3中查找接近的时间戳
            left2 = np.searchsorted(
                cmd_data, timestamp1 - threshold, side='left')
            right2 = np.searchsorted(
                cmd_data, timestamp1 + threshold, side='right')
            left3 = np.searchsorted(
                expect_data, timestamp1 - threshold, side='left')
            right3 = np.searchsorted(
                expect_data, timestamp1 + threshold, side='right')

            # 检查索引是否有效
            if left2 < right2 and left3 < right3:
                if (debug):
                    matches.append(
                        (timestamp1, cmd_data[left2], expect_data[left3]))
                    real_temp.append([timestamp1, round(real_data_speed[idx1], 3)])
                    cmd_temp.append(
                        [cmd_data[left2], round(cmd_data_speed[left2], 3)])
                    expect_temp.append(
                        [expect_data[left3], round(expect_data_speed[left3], 3)])

                file.write(str(timestamp1)+" "+str(round(real_data_speed[idx1], 3))+" "+str(
                    round(cmd_data_speed[left2], 3))+" " + str(round(expect_data_speed[left3], 3))+" ")
                file.write("\n")

            # 更新索引
            idx1 += 1
            idx2 = max(idx2, left2)
            idx3 = max(idx3, left3)

def generateTrainData(write_path, read_path):
    with open(write_path, "a") as write_file:
        write_file.truncate(0)
        with open(read_path, "r") as file:
            lines = file.readlines()
            data_list = []
            for line in lines:
                line = line.split(" ")
                line.pop()
                data_list.append([float(item) for item in line])
            first_t = data_list[0][0]
            for index, data in enumerate(data_list):
                if data[0]-first_t >= 3.0:
                    start_index = index+1
                    break
            len_list = []
            for i in range(start_index, len(data_list)):
                label = data_list[i][1]
                real_temp = []
                cmd_temp = []
                t_temp = []
                for item in data_list[0:i]:
                    if 0.0 <= data_list[i][0] - item[0] <= 3.0:
                        real_temp.append(item[1])
                        cmd_temp.append(item[2])
                        t_temp.append(item[0])
                if len(real_temp) > 0:
                    start_time = np.array(t_temp).min()
                    end_time = np.array(t_temp).max()
                    if end_time-start_time > 2.8:
                        if len(real_temp) > 30 and (len(real_temp) - len(set(real_temp)))/len(real_temp) < 0.9:
                            new_timestamps = np.arange(
                                end_time-3.0, end_time, 0.08)
                            # 使用线性插值计算新时间点的速度
                            real_temp = np.interp(
                                new_timestamps, t_temp, real_temp)
                            cmd_temp = np.interp(
                                new_timestamps, t_temp, cmd_temp)
                            len_list.append(len(real_temp))
                            real_temp = ', '.join(str(element)
                                                  for element in real_temp)
                            cmd_temp = ', '.join(str(element)
                                                 for element in cmd_temp)
                            tmp_data = [label, real_temp, cmd_temp]
                            for item in tmp_data:
                                write_file.write(str(item)+" ")
                            write_file.write("\n")
        print(len(len_list), len_list)


def mergeData(write_path, read_path):
    # 读取源文件的全部内容
    with open(read_path, 'r', encoding='utf-8') as source_file:
        content = source_file.read()

    # 将读取的内容写入目标文件
    with open(write_path, 'a', encoding='utf-8') as target_file:
        target_file.write(content)

        print("finsh")


def generateTrainData(write_path, read_path):
    with open(write_path, "a") as write_file:
        write_file.truncate(0)
        with open(read_path, "r") as file:
            lines = file.readlines()
            data_list = []
            for line in lines:
                line = line.split(" ")
                line.pop()
                data_list.append([float(item) for item in line])
            first_t = data_list[0][0]
            for index, data in enumerate(data_list):
                if data[0]-first_t >= 3.0:
                    start_index = index+1
                    break
            len_list = []
            for i in range(start_index, len(data_list)):
                label = data_list[i][1]
                real_temp = []
                cmd_temp = []
                t_temp = []
                for item in data_list[0:i]:
                    if 0.0 <= data_list[i][0] - item[0] <= 3.0:
                        real_temp.append(item[1])
                        cmd_temp.append(item[2])
                        t_temp.append(item[0])
                if len(real_temp) > 0:
                    start_time = np.array(t_temp).min()
                    end_time = np.array(t_temp).max()
                    if end_time-start_time > 2.8:
                        if len(real_temp) > 30 and (len(real_temp) - len(set(real_temp)))/len(real_temp) < 0.9:
                            new_timestamps = np.arange(
                                end_time-3.0, end_time, 0.08)
                            # 使用线性插值计算新时间点的速度
                            real_temp = np.interp(
                                new_timestamps, t_temp, real_temp)
                            cmd_temp = np.interp(
                                new_timestamps, t_temp, cmd_temp)
                            len_list.append(len(real_temp))
                            real_temp = ', '.join(str(element)
                                                  for element in real_temp)
                            cmd_temp = ', '.join(str(element)
                                                 for element in cmd_temp)
                            tmp_data = [label, real_temp, cmd_temp]
                            for item in tmp_data:
                                write_file.write(str(item)+" ")
                            write_file.write("\n")
        print(len(len_list), len_list)


def mergeData(write_path, read_path):
    # 读取源文件的全部内容
    with open(read_path, 'r', encoding='utf-8') as source_file:
        content = source_file.read()

    # 将读取的内容写入目标文件
    with open(write_path, 'a', encoding='utf-8') as target_file:
        target_file.write(content)


# if __name__ == "__main__":
#     directory_path = r'data\new_dataset_car2_600kg\data_set'
#     with open(r"data\orin_train.txt", "a") as file:
#         file.truncate(0)
#     for root, dirs, files in os.walk(directory_path):
#         for file_name in files:
#             print(file_name)
#             motor_reals = decodeMotorReal(r"data\new_dataset_car2_600kg\real_speed\\"+file_name + ".txt")
#             motor_cmds = decodeMotorCMD(r"data\new_dataset_car2_600kg\cmd_speed\\"+file_name + ".txt")
#             motor_reals,motor_cmds = alignData(motor_reals,motor_cmds)
#             matchData(motor_reals,motor_cmds,r"data\orin_train.txt")
#     formatFile(r"data\orin_train.txt")
