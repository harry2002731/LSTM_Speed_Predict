import numpy as np
import os

seach_scope = 25
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
        if self.calRepRate(self.data)>0.9:
            self.use = False
        if len(self.data) < 25:
            self.use = False  
        return self.use
    
    def processer(self):
        pass

# 对齐起始的两组数据的时间戳
def alignData(motor_reals,motor_cmds):
    temp_real_t = motor_reals[0].time_info.time
    temp_cmd_t = motor_cmds[0].time_info.time
    index = -1

    if temp_real_t > temp_cmd_t:
        for index,motor_cmd in enumerate(motor_cmds):
            if temp_real_t - motor_cmd.time_info.time <= 3.0:
                motor_cmds = motor_cmds[index:-1]
                break
    else:
        for index,motor_real in enumerate(motor_reals):
            if motor_real.time_info.time - temp_cmd_t >= 3.0:
                motor_reals = motor_reals[index:-1]
                break
    assert index != -1,"没有匹配的数据"
    return motor_reals, motor_cmds 

# # 数据时间戳匹配
# def matchData(motor_reals,motor_cmds):
#     len_list = []
#     with open(r"data\train.txt", "a") as file:
#         file.truncate(0)
#         for index, motor_real in enumerate(motor_reals):
#             cmd_temp = []
#             real_temp = []
#             for motor_cmd in motor_cmds:
#                 if  0.0 < motor_real.time_info.time - motor_cmd.time_info.time <= 3.0:
                    
#                     if index >= seach_scope and index < len(motor_reals) - seach_scope:
#                         start_index = index - seach_scope
#                         end_index = index + seach_scope
#                     else:
#                         continue
        
#                     a = np.array([i.time_info.time for i in motor_reals[start_index:end_index]])
#                     b = np.array(np.full(a.shape,motor_cmd.time_info.time))
#                     minus_result = abs(a-b)
#                     tmp_ = a[minus_result.argmin()]

#                     if len(real_temp)>0:
#                         if min(minus_result) < 0.03 and tmp_ != real_temp[-1][0]:

#                             real_temp.append([tmp_,round(minus_result[minus_result.argmin()], 3)])
#                             cmd_temp.append([motor_cmd.time_info.time,round(motor_cmd.motor2, 3)])
#                     else:
#                         if min(minus_result) < 0.03:
#                             # real_temp.append([tmp_,minus_result[minus_result.argmin()],motor_cmd.time_info.time,motor_cmd.motor2])
#                             real_temp.append([tmp_,round(minus_result[minus_result.argmin()], 3)])
#                             cmd_temp.append([motor_cmd.time_info.time,round(motor_cmd.motor2, 3)])

#             final_temp = [motor_real.speed,real_temp,cmd_temp]
#             if len(real_temp) >10:
#                 # len_list.append(len(real_temp))
#                 real_array = np.array(real_temp)[:,-1]
#                 cmd_array = np.array(cmd_temp)[:,-1]
#                 dp = DataProcess(real_array)
#                 dp2 = DataProcess(cmd_array)
#                 tmp_temp = [motor_real.speed,real_array.tolist(),cmd_array.tolist()]
#                 if dp.filter() and dp2.filter(): 
#                     len_list.append(len(real_temp))
#                     for item in  tmp_temp:
#                         file.write(str(item)+" ")
#                     file.write("\n")


# 数据时间戳匹配
def matchData(motor_reals,motor_cmds):
    with open(r"data\train.txt", "a") as file:
        # file.truncate(0)
        len_list = []
        for index, motor_real in enumerate(motor_reals):
            cmd_temp = []
            real_temp = []
            for motor_cmd in motor_cmds:
                if  0.0 < motor_real.time_info.time - motor_cmd.time_info.time <= 3.0:
                    motor_cmd_time = motor_cmd.time_info.time
                    if  seach_scope <= index and index < len(motor_reals) - seach_scope:
                        start_index = index - seach_scope
                        end_index = index + seach_scope
                    else:
                        continue
        
                    real_time_array = np.array([tmp.time_info.time for tmp in motor_reals[start_index:end_index]])
                    real_speed_array = np.array([tmp.speed for tmp in motor_reals[start_index:end_index]])
                    motor_cmd_array = np.full(real_time_array.shape,motor_cmd_time)
                    time_gap = abs(real_time_array-motor_cmd_array)
                    min_gap_real = real_time_array[time_gap.argmin()]

                    if min(time_gap) < 0.03:
                        if len(real_temp)>0:
                            if min_gap_real != real_temp[-1][0]:
                                real_temp.append([min_gap_real, round(real_speed_array[time_gap.argmin()], 3)])
                                cmd_temp.append([motor_cmd_time, round(motor_cmd.motor2, 3)])
                        else:
                            real_temp.append([min_gap_real, round(real_speed_array[time_gap.argmin()], 3)])
                            cmd_temp.append([motor_cmd_time, round(motor_cmd.motor2, 3)])

            final_temp = [motor_real.speed,real_temp,cmd_temp]
            if len(real_temp) >10:
                real_array = np.array(real_temp)[:,-1]
                cmd_array = np.array(cmd_temp)[:,-1]
                dp = DataProcess(real_array)
                dp2 = DataProcess(cmd_array)
                tmp_temp = [motor_real.speed,real_array.tolist(),cmd_array.tolist()]
                if dp.filter() and dp2.filter(): 
                    len_list.append(len(real_temp))
                    for item in  tmp_temp:
                        # print(final_temp)
                        file.write(str(item)+" ")
                    file.write("\n")
# 
        print(len(len_list),len_list)

if __name__ == "__main__":
    directory_path = r'data\origin_dataset\data_set'
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            print(file_name)
            motor_reals = decodeMotorReal(r"data\origin_dataset\real_speed\\"+file_name + ".txt")
            motor_cmds = decodeMotorCMD(r"data\origin_dataset\cmd_speed\\"+file_name + ".txt")
            motor_reals,motor_cmds = alignData(motor_reals,motor_cmds)
            matchData(motor_reals,motor_cmds)

