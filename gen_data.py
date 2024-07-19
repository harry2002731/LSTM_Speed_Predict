import numpy as np
import os

seach_scope = 50
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

class MotorDataProcess():
    def __init__(self,data) -> None:
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
        self.repetition_rate = (len(self.data) - len(set(self.data)))/len(self.data)
        return self.repetition_rate
    
    def filter(self):
        if self.calRepRate(self.data)>0.999:
            self.use = False
        if self.calRepRate(self.data)>0.999:
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
        self.data = np.transpose(temp, (1,0))

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
# def matchData(motor_reals,motor_cmds,path):
#     with open(path, "a") as file:
#         # file.truncate(0)
#         len_list = []
#         for index, motor_real in enumerate(motor_reals):
#             cmd_temp = []
#             real_temp = []
#             for motor_cmd in motor_cmds:
#                 # 时间搜索范围
#                 if  0.0 < motor_real.time_info.time - motor_cmd.time_info.time <= 3.0:
#                     motor_cmd_time = motor_cmd.time_info.time
#                     # 判断索引是否大于seach_scope搜索范围
#                     if  seach_scope <= index and index < len(motor_reals) - seach_scope:
#                         start_index = index - seach_scope
#                         end_index = index + seach_scope-25
#                     else:
#                         continue
# # 数据时间戳匹配
# def matchData(motor_reals,motor_cmds,path):
#     with open(path, "a") as file:
#         # file.truncate(0)
#         len_list = []
#         for index, motor_real in enumerate(motor_reals):
#             cmd_temp = []
#             real_temp = []
#             for motor_cmd in motor_cmds:
#                 # 时间搜索范围
#                 if  0.0 < motor_real.time_info.time - motor_cmd.time_info.time <= 3.0:
#                     motor_cmd_time = motor_cmd.time_info.time
#                     # 判断索引是否大于seach_scope搜索范围
#                     if  seach_scope <= index and index < len(motor_reals) - seach_scope:
#                         start_index = index - seach_scope
#                         end_index = index + seach_scope-25
#                     else:
#                         continue

#                     # 判断搜索范围内（start_index:end_index）是否有与motor_cmd时间戳匹配（time_gap小于0.03）的motor_real
#                     real_time_array = np.array([tmp.time_info.time for tmp in motor_reals[start_index:end_index]])
#                     real_speed_array = np.array([tmp.speed for tmp in motor_reals[start_index:end_index]])
#                     motor_cmd_array = np.full(real_time_array.shape,motor_cmd_time)
#                     time_gap = abs(real_time_array-motor_cmd_array)
#                     min_gap_real = real_time_array[time_gap.argmin()]
#                     # 判断搜索范围内（start_index:end_index）是否有与motor_cmd时间戳匹配（time_gap小于0.03）的motor_real
#                     real_time_array = np.array([tmp.time_info.time for tmp in motor_reals[start_index:end_index]])
#                     real_speed_array = np.array([tmp.speed for tmp in motor_reals[start_index:end_index]])
#                     motor_cmd_array = np.full(real_time_array.shape,motor_cmd_time)
#                     time_gap = abs(real_time_array-motor_cmd_array)
#                     min_gap_real = real_time_array[time_gap.argmin()]

#                     if min(time_gap) < 0.03:
#                         if len(real_temp)>0:
#                             if min_gap_real != real_temp[-1][0]:
#                                 real_temp.append([min_gap_real, round(real_speed_array[time_gap.argmin()], 3)])
#                                 cmd_temp.append([motor_cmd_time, round(motor_cmd.motor2, 3)])
#                         else:
#                             real_temp.append([min_gap_real, round(real_speed_array[time_gap.argmin()], 3)])
#                             cmd_temp.append([motor_cmd_time, round(motor_cmd.motor2, 3)])
#                     if min(time_gap) < 0.03:
#                         if len(real_temp)>0:
#                             if min_gap_real != real_temp[-1][0]:
#                                 real_temp.append([min_gap_real, round(real_speed_array[time_gap.argmin()], 3)])
#                                 cmd_temp.append([motor_cmd_time, round(motor_cmd.motor2, 3)])
#                         else:
#                             real_temp.append([min_gap_real, round(real_speed_array[time_gap.argmin()], 3)])
#                             cmd_temp.append([motor_cmd_time, round(motor_cmd.motor2, 3)])
                            
#             # final_temp = [motor_real.speed,real_temp,cmd_temp]
#             if len(real_temp) >10:
#                 if real_temp[-1][0]-motor_real.time_info.time == 0:
#                     real_temp.pop()
#                     cmd_temp.pop()
#             # final_temp = [motor_real.speed,real_temp,cmd_temp]
#             if len(real_temp) >10:
#                 if real_temp[-1][0]-motor_real.time_info.time == 0:
#                     real_temp.pop()
#                     cmd_temp.pop()

#                 dp1 = MotorDataProcess(real_temp)
#                 dp2 = MotorDataProcess(cmd_temp)
#                 dp1 = MotorDataProcess(real_temp)
#                 dp2 = MotorDataProcess(cmd_temp)

#                 if dp1.judgeTimeStamp(time_gap=2.8) and dp2.judgeTimeStamp(time_gap=2.8):
#                     # dp1.interp(time_span=3.0, time_gap=0.1)
#                     # dp2.interp(time_span=3.0, time_gap=0.1)
#                 if dp1.judgeTimeStamp(time_gap=2.8) and dp2.judgeTimeStamp(time_gap=2.8):
#                     # dp1.interp(time_span=3.0, time_gap=0.1)
#                     # dp2.interp(time_span=3.0, time_gap=0.1)

#                     real_array = np.array(dp1.data)[:,-1]
#                     cmd_array = np.array(dp2.data)[:,-1]
#                     tmp_temp = [motor_real.speed,real_array.tolist(),cmd_array.tolist()]
#                     if dp1.filter() and dp2.filter(): 
#                         len_list.append(len(real_array))
#                         for item in tmp_temp:
#                             file.write(str(item)+" ")
#                         file.write("\n")
#         print(len(len_list),len_list)
#                     real_array = np.array(dp1.data)[:,-1]
#                     cmd_array = np.array(dp2.data)[:,-1]
#                     tmp_temp = [motor_real.speed,real_array.tolist(),cmd_array.tolist()]
#                     if dp1.filter() and dp2.filter(): 
#                         len_list.append(len(real_array))
#                         for item in tmp_temp:
#                             file.write(str(item)+" ")
#                         file.write("\n")
#         print(len(len_list),len_list)

# 滤除文件内的[]和，
def formatFile(path):
    with open(path, "r") as file:
        lines = file.readlines() 
        with open(r"data\train.txt", "a") as write_file:
            write_file.truncate(0)  # 清除文件内容
            for line in lines: 
                line =  line.replace('[', '').replace(']', '').replace(',', '')
                write_file.write(str(line))







# 数据时间戳匹配
def matchData(motor_reals,motor_cmds,path):
def matchData(motor_reals,motor_cmds,path):
    with open(path, "a") as file:
        cmd_temp = []
        real_temp = []
        file.truncate(0)
        file.truncate(0)
        for index, motor_real in enumerate(motor_reals):

            for motor_cmd in motor_cmds:
                # 时间搜索范围
                if  0.0 < motor_real.time_info.time - motor_cmd.time_info.time <= 0.1:
                        real_temp.append([motor_real.time_info.time , round(motor_real.speed, 3)])
                        cmd_temp.append([motor_cmd.time_info.time, round(motor_cmd.motor2, 3)])
                        file.write(str(motor_real.time_info.time)+" "+str(round(motor_real.speed, 3))+" "+str(round(motor_cmd.motor2, 3))+" ")
                        file.write("\n")
                        break
        print("finsh") 

def generateTrainData(write_path, read_path):
    with open(write_path,"a") as write_file:
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
                if data[0]-first_t>=3.0:
                    start_index = index+1
                    break
            len_list = []
            for i in range(start_index,len(data_list)):
                label = data_list[i][1]
                real_temp = []
                cmd_temp = []
                t_temp = [] 
                for item in data_list[0:i]:
                    if 0.0<=data_list[i][0] - item[0] <= 3.0:
                        real_temp.append(item[1])
                        cmd_temp.append(item[2])
                        t_temp.append(item[0])
                if len(real_temp)>0:
                    start_time = np.array(t_temp).min()
                    end_time = np.array(t_temp).max()                
                    if end_time-start_time > 2.8:
                        if len(real_temp)>30 and (len(real_temp) - len(set(real_temp)))/len(real_temp) < 0.9:
                            new_timestamps = np.arange(end_time-3.0, end_time, 0.08)
                            # 使用线性插值计算新时间点的速度
                            real_temp = np.interp(new_timestamps, t_temp, real_temp)
                            cmd_temp = np.interp(new_timestamps, t_temp, cmd_temp)
                            len_list.append(len(real_temp))
                            real_temp =  ', '.join(str(element) for element in real_temp)
                            cmd_temp =  ', '.join(str(element) for element in cmd_temp)
                            tmp_data = [label,real_temp,cmd_temp]
                            for item in tmp_data:
                                write_file.write(str(item)+" ")
                            write_file.write("\n")
        print(len(len_list),len_list)

def mergeData(write_path, read_path):
    # 读取源文件的全部内容
    with open(read_path, 'r', encoding='utf-8') as source_file:
        content = source_file.read()

    # 将读取的内容写入目标文件
    with open(write_path, 'a', encoding='utf-8') as target_file:
        target_file.write(content)


        print("finsh") 

def generateTrainData(write_path, read_path):
    with open(write_path,"a") as write_file:
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
                if data[0]-first_t>=3.0:
                    start_index = index+1
                    break
            len_list = []
            for i in range(start_index,len(data_list)):
                label = data_list[i][1]
                real_temp = []
                cmd_temp = []
                t_temp = [] 
                for item in data_list[0:i]:
                    if 0.0<=data_list[i][0] - item[0] <= 3.0:
                        real_temp.append(item[1])
                        cmd_temp.append(item[2])
                        t_temp.append(item[0])
                if len(real_temp)>0:
                    start_time = np.array(t_temp).min()
                    end_time = np.array(t_temp).max()                
                    if end_time-start_time > 2.8:
                        if len(real_temp)>30 and (len(real_temp) - len(set(real_temp)))/len(real_temp) < 0.9:
                            new_timestamps = np.arange(end_time-3.0, end_time, 0.08)
                            # 使用线性插值计算新时间点的速度
                            real_temp = np.interp(new_timestamps, t_temp, real_temp)
                            cmd_temp = np.interp(new_timestamps, t_temp, cmd_temp)
                            len_list.append(len(real_temp))
                            real_temp =  ', '.join(str(element) for element in real_temp)
                            cmd_temp =  ', '.join(str(element) for element in cmd_temp)
                            tmp_data = [label,real_temp,cmd_temp]
                            for item in tmp_data:
                                write_file.write(str(item)+" ")
                            write_file.write("\n")
        print(len(len_list),len_list)

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

