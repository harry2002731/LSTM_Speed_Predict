import os
from gen_data import *

def real_match_line(line):
    keyword = "MotorInfo2"
    return keyword in line

def cmd_match_line(line):
    keyword1 = "MotorCmd"
    keyword2 = "GDMotorCmd"
    return keyword1 in line and keyword2 not in line

def expect_match_line(line):
    keyword1 = "ForkMotors"
    return keyword1 in line

def extractLog(dir,tar_dir, command):
    # 遍历目录下的所有文件和子目录
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file not in [name.split(".txt")[0] for name in os.listdir(tar_dir)]:
                contents = []
                file_path = os.path.join(root, file)
                try:
                    # 打开并读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.readlines()
                        matching_lines = [line for line in content if command(line)]
                        contents.append(matching_lines)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                path = tar_dir+ f"{file}.txt"
                with open(path, "a") as f:
                    f.truncate(0)
                    for item in contents[0]:
                        item.replace("\n", "")
                        f.write(str(item))

if __name__ == "__main__":


    # 指定要遍历的文件夹路径
    root_directory_path =   r'data\\new_dataset_car1\\'

    orin_path = root_directory_path + r'data_set\\'
    real_speed_p = root_directory_path+r'extracted_data\\real_speed\\'
    cmd_speed_p = root_directory_path+r"extracted_data\\cmd_speed\\"
    expect_speed_p = root_directory_path+r"extracted_data\\expect_speed\\"

    motor_compared_path = root_directory_path+r"compared_data\\"
    motor_train_path = root_directory_path+r"train_data\\"

    # 提取符合要求的字段
    extractLog(orin_path,cmd_speed_p, cmd_match_line)
    extractLog(orin_path,real_speed_p, real_match_line)
    extractLog(orin_path,expect_speed_p, expect_match_line)

    with open(r"data\new_car1_train.txt", "a") as file:
        file.truncate(0)
    for root, dirs, files in os.walk(orin_path):
        for file_name in files:
            real_speeds = decodeMotorSpeeed(real_speed_p + file_name + ".txt", MotorReal)
            cmd_speeds = decodeMotorSpeeed(cmd_speed_p + file_name + ".txt",MotorCmd)
            expect_speeds = decodeMotorSpeeed(expect_speed_p + file_name + ".txt",MotorExpect)
            
            matchData(real_speeds,cmd_speeds,expect_speeds, motor_compared_path + file_name + ".txt", True)
            generateTrainData(motor_train_path + file_name + ".txt",motor_compared_path + file_name + ".txt", 1.5, 0.05,True)
            mergeData(r"data\new_car1_train.txt", motor_train_path + file_name + ".txt")
            # break
    # formatFile(r"data\new_car2_train.txt")
