import os
from gen_data import *
import platform


# 实际速度关键词
def real_match_line(line):
    keyword = "MotorInfo2"
    return keyword in line

# 控制速度关键词
def cmd_match_line(line):
    keyword1 = "MotorCmd"
    keyword2 = "GDMotorCmd"
    return keyword1 in line and keyword2 not in line

# 期望速度关键词
def expect_match_line(line):
    keyword1 = "ForkMotors"
    return keyword1 in line

# 自定义关键词（包含实际、下发和期望）
def roboshop_match_line(line):
    keyword1 = "Default argument"
    return keyword1 in line

# 叉车位置关键词
def fork_match_line(line):
    keyword = "ForkMotors"
    return keyword in line
    

def extractLog(dir, tar_dir, command):
    # 遍历目录下的所有文件和子目录
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file not in [name.split(".txt")[0] for name in os.listdir(tar_dir)]:
                contents = []
                file_path = os.path.join(root, file)
                try:
                    # 打开并读取文件内容
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.readlines()
                        matching_lines = [line for line in content if command(line)]
                        contents.append(matching_lines)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                path = tar_dir + f"{file}.txt"
                with open(path, "a") as f:
                    f.truncate(0)
                    for item in contents[0]:
                        item.replace("\n", "")
                        f.write(str(item))
data_set_type_enum = ["orin","modifyied","fork"]

if __name__ == "__main__":
    data_type =  "fork"
    compare_data = False
    sys = platform.system()
    # 指定要遍历的文件夹路径
    root_directory_path = r"data/roboshop_data/"
    if sys == "Windows":
        orin_path = root_directory_path + r"data_set\\"
        real_speed_p = root_directory_path + r"extracted_data\\real_speed\\"
        cmd_speed_p = root_directory_path + r"extracted_data\\cmd_speed\\"
        expect_speed_p = root_directory_path + r"extracted_data\\expect_speed\\"
        total_speed_p = root_directory_path + r"extracted_data\\total_speed\\"

        motor_compared_path = root_directory_path + r"compared_data\\"
        motor_train_path = root_directory_path + r"train_data\\"
    elif sys == "Linux":
        orin_path = root_directory_path + r"data_set/"
        real_speed_p = root_directory_path + r"extracted_data/real_speed/"
        cmd_speed_p = root_directory_path + r"extracted_data/cmd_speed/"
        expect_speed_p = root_directory_path + r"extracted_data/expect_speed/"
        total_speed_p = root_directory_path + r"extracted_data/total_speed/"
        fork_position_p = root_directory_path + r"extracted_data/fork_speed/"
        

        motor_compared_path = root_directory_path + r"compared_data/"
        motor_train_path = root_directory_path + r"train_data/"

    # 提取符合要求的字段
    # extractLog(orin_path,cmd_speed_p, cmd_match_line)
    # extractLog(orin_path,real_speed_p, real_match_line)
    # extractLog(orin_path,expect_speed_p, expect_match_line)
    extractLog(orin_path, total_speed_p, roboshop_match_line)
    extractLog(orin_path, fork_position_p, fork_match_line)

    with open(root_directory_path + "/train.txt", "a") as file:
        file.truncate(0)
    with open("./data/train.txt", "a") as file:
        file.truncate(0)

    # 匹配数据并存放到compared中
    if compare_data:
        for root, dirs, files in os.walk(orin_path):
            for file_name in files:
                if data_type in data_set_type_enum and data_type ==  "orin":
                    extractLog(orin_path,cmd_speed_p, cmd_match_line)
                    extractLog(orin_path,real_speed_p, real_match_line)
                    extractLog(orin_path,expect_speed_p, expect_match_line)
                    real_speeds = decodeMotorSpeed(real_speed_p + file_name + ".txt", MotorReal)
                    cmd_speeds = decodeMotorSpeed(cmd_speed_p + file_name + ".txt",MotorCmd)
                    expect_speeds = decodeMotorSpeed(expect_speed_p + file_name + ".txt",MotorExpect)  
                    matchData(real_speeds,cmd_speeds,expect_speeds, motor_compared_path + file_name + ".txt", True)

                elif data_type in data_set_type_enum and data_type ==  "modifyied":
                    total_data = decodeMotorSpeed(total_speed_p + file_name + ".txt", MotorTotal)
                    writeFile(motor_compared_path + file_name + ".txt",total_data)
                    
                elif data_type in data_set_type_enum and data_type ==  "fork":
                    total_data = decodeMotorSpeed(total_speed_p + file_name + ".txt", MotorTotal)
                    fork_data = decodeMotorSpeed(fork_position_p + file_name + ".txt", ForkPosition)
                    matchData2(total_data,fork_data,fork_data, motor_compared_path + file_name + ".txt", True)

                
    # 从compared的数据中生成训练数据到train文件夹中
    for root, dirs, files in os.walk(motor_compared_path):
        for file_name in files:                
            generateTrainData(
                motor_train_path + file_name,
                motor_compared_path + file_name,
                1.3,
                interp_interval=0.02,
                repetition_rate=0.8,
                interp=True,
                difference=False
            )
            mergeData(
                root_directory_path + "/train.txt",
                motor_train_path + file_name,
            )
        mergeData("./data/train.txt", root_directory_path + "/train.txt")
    
    
    
    # formatFile(r"data\new_car2_train.txt")
# 调整时间间隔
# 修改网络输出 多时间 0.1 0.2 0.3 ，然后由滞后性判断使用哪个值
