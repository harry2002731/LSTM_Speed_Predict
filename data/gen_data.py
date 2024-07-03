import numpy as np
import os

class TimeInfo():
    def __init__(self) -> None:
        self.date = "" # 日期
        self.time = "" # 时间
        self.frame_num = "" # 数据id

#泵控电机真实值
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
    def decode(self,data_str):
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
                index +=1
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

    def decode(self,data_str):
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
                index +=1
            if record:
                str_ += i

def decodeMotorCMD(path):
    data = []
    with open(path,"r") as file:
        lines = file.readlines() 
        for line in lines: 
            m = MotorCmd()
            m.decode(line)
            data.append(m)
    return data
    

def decodeMotorReal(path):
    data = []
    with open(path,"r") as file:
        lines = file.readlines() 
        for line in lines: 
            m = MotorReal()
            m.decode(line)
            data.append(m)
    return data

class DataProcess():
    def __init__(self,data) -> None:
        self.repetition_rate = 0
        self.data = data
        self.use = True
    def calRepRate(self,data):
        self.repetition_rate = (len(self.data) - len(set(self.data)))/len(self.data)
        return self.repetition_rate
    def filter(self):
        if self.calRepRate(self.data)>0.5:
            self.use = False
            # pass
        if len(self.data) < 92:
            self.use = False  
        return self.use
    
    def processer(self):
        pass


def matchData(motor_reals,motor_cmds):
    temp_real_t = motor_reals[0].time_info.time
    temp_cmd_t = motor_cmds[0].time_info.time
    index = -1

    if temp_real_t > temp_cmd_t:
        for i,motor_cmd in enumerate(motor_cmds):
            if temp_real_t - motor_cmd.time_info.time <= 3.0:
                index = i
                motor_cmds = motor_cmds[index:-1]
                break
    else:
        for i,motor_real in enumerate(motor_reals):
            if motor_real.time_info.time - temp_cmd_t >= 3.0:
                index = i
                motor_reals = motor_reals[index:-1]
                break
    assert index != -1,"没有匹配的数据"
    j = 0
    len_list = []
    with open(r"C:\Projects\Python\speed_control\train.txt", "a") as file:
        # file.truncate(0)
        for motor_real in motor_reals:
            temp = []
            temp.append(motor_real.speed)
            for motor_cmd in motor_cmds:
                if  0.0 < motor_real.time_info.time - motor_cmd.time_info.time <= 3.0:
                    temp.append(motor_cmd.motor2)
            dp = DataProcess(temp)
            if dp.filter():
                len_list.append(len(temp))
                j+=1
                for item in temp:
                    file.write(str(item)+" ")
                file.write("\n")
        print(np.mean(len_list),j)

if __name__ == "__main__":
    directory_path = r'data\origin_dataset\data_set'
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            motor_reals = decodeMotorReal(r"data\origin_dataset\real_speed\\"+file_name + ".txt")
            motor_cmds = decodeMotorCMD(r"data\origin_dataset\cmd_speed\\"+file_name + ".txt")
            matchData(motor_reals,motor_cmds)

