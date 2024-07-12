import os
from gen_data import *

def real_match_line(line):
    keyword = "MotorInfo2"
    return keyword in line

def cmd_match_line(line):
    keyword2 = "MotorCmd"
    keyword3 = "GDMotorCmd"
    return keyword2 in line and keyword3 not in line

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
    root_directory_path =   r'data\\new_dataset_car2\\'

    orin_path = root_directory_path + r'data_set\\'
    motor_real_path = root_directory_path+r'real_speed\\'
    motor_cmd_path = root_directory_path+r"cmd_speed\\"
    # 调用函数并打印结果
    # extractRealLog(directory_path,motor_real_directory_path)
    extractLog(orin_path,motor_cmd_path, cmd_match_line)
    extractLog(orin_path,motor_real_path, real_match_line)

    with open(r"data\new_car2_train.txt", "a") as file:
        file.truncate(0)
    for root, dirs, files in os.walk(orin_path):
        for file_name in files:
            motor_reals = decodeMotorReal(motor_real_path + file_name + ".txt")
            motor_cmds = decodeMotorCMD(motor_cmd_path + file_name + ".txt")
            motor_reals,motor_cmds = alignData(motor_reals,motor_cmds)

            matchData(motor_reals,motor_cmds,r"data\new_car2_train.txt")
            # matchData2(motor_reals,motor_cmds,r"data\111.txt")
            # break
    formatFile(r"data\new_car2_train.txt")
