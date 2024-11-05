import os
import time
from genTrainData import *
import platform
from fileProcessor import *


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

# 根据关键词进行字段匹配，提取出需要的数据


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
                        matching_lines = [
                            line for line in content if command(line)]
                        contents.append(matching_lines)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                path = tar_dir + f"{file}.txt"
                with open(path, "a") as f:
                    f.truncate(0)
                    for item in contents[0]:
                        item.replace("\n", "")
                        f.write(str(item))


if __name__ == "__main__":
    # roboshop测试时自动从本地文件夹复制到目标路径
    data_set_type_enum = ["orin", "modifyied", "fork"]
    data_type = "modifyied"
    compare_data = True
    generate_train_data = True
    copy_file = False

    compare_data = True
    generate_train_data = False
    copy_file = False
    mode = "train"

    if copy_file:
        latest_file_name = ""
        need_file_name = find_latest_file("data/roboshop_data/data_set/")

        data_from_path = "/home/ubuntu/Desktop/"                      
        # data_from_path = "/usr/local/etc/.SeerRobotics/rbk/diagnosis/log/"
        need_file_name = find_latest_file(data_from_path)

        if need_file_name != latest_file_name:
            mergeData("data/roboshop_data/data_set/" +
                      need_file_name, data_from_path+need_file_name)
            mergeData("data/roboshop_data/data_set/train/" +
                      need_file_name, data_from_path+need_file_name)

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
        orin_train_path = root_directory_path + r"data_set/train"
        orin_test_path = root_directory_path + r"data_set/test"

        real_speed_p = root_directory_path + r"extracted_data/real_speed/"
        cmd_speed_p = root_directory_path + r"extracted_data/cmd_speed/"
        expect_speed_p = root_directory_path + r"extracted_data/expect_speed/"
        total_speed_p = root_directory_path + r"extracted_data/total_speed/"
        fork_position_p = root_directory_path + r"extracted_data/fork_speed/"

        motor_train_compared_path = root_directory_path + r"compared_data/train/"
        motor_test_compared_path = root_directory_path + r"compared_data/test/"
        motor_train_path = root_directory_path + r"train_data/"

    # 提取符合要求的字段
    # extractLog(orin_path,cmd_speed_p, cmd_match_line)
    # extractLog(orin_path,real_speed_p, real_match_line)
    # extractLog(orin_path,expect_speed_p, expect_match_line)

    if mode == "train":
        orin_path = orin_train_path
        txt_name = "train"
        motor_compared_path = motor_train_compared_path
    elif mode == "test":
        orin_path = orin_test_path
        txt_name = "test"
        motor_compared_path = motor_test_compared_path

    extractLog(orin_path, total_speed_p, roboshop_match_line)
    extractLog(orin_path, fork_position_p, fork_match_line)

    # 匹配数据并存放到compared中
    if compare_data:
        for root, dirs, files in os.walk(orin_path):
            for file_name in files:
                # file_name = "robokit_2024-10-23_14-51-30.0.log"
                if data_type in data_set_type_enum and data_type == "orin":
                    extractLog(orin_path, cmd_speed_p, cmd_match_line)
                    extractLog(orin_path, real_speed_p, real_match_line)
                    extractLog(orin_path, expect_speed_p, expect_match_line)
                    real_speeds = decodeDataList(
                        real_speed_p + file_name + ".txt", MotorReal)
                    cmd_speeds = decodeDataList(
                        cmd_speed_p + file_name + ".txt", MotorCmd)
                    expect_speeds = decodeDataList(
                        expect_speed_p + file_name + ".txt", MotorExpect)
                    matchOrinData(real_speeds, cmd_speeds, expect_speeds,
                                  motor_compared_path + file_name + ".txt", True)

                elif data_type in data_set_type_enum and data_type == "modifyied":
                    total_data = decodeDataList(
                        total_speed_p + file_name + ".txt", MotorTotal)
                    writeFile(motor_compared_path +
                              file_name + ".txt", total_data)

                elif data_type in data_set_type_enum and data_type == "fork":
                    total_data = decodeDataList(
                        total_speed_p + file_name + ".txt", MotorTotal)
                    fork_data = decodeDataList(
                        fork_position_p + file_name + ".txt", ForkPosition)
                    matchForkData(total_data, fork_data, fork_data,
                                  motor_compared_path + file_name + ".txt", True)

    if generate_train_data:
        with open(root_directory_path + "/"+txt_name+".txt", "a") as file:
            file.truncate(0)
        with open("./data/"+txt_name+".txt", "a") as file:
            file.truncate(0)
        # 从compared的数据中生成训练数据到train文件夹中
        data_list = ["real_temp", "cmd_temp", "expect_temp",
                     "height_gap_temp", "speed_gap_temp"]
        label_list = ["real_label", "cmd_label",
                      "expect_label", "height_label", "speed_gap_label"]
        for root, dirs, files in os.walk(motor_compared_path):
            for file_name in files:
                start_time = time.time()
                generateTrainData(
                    motor_train_path + file_name,
                    motor_compared_path + file_name,
                    1.4,
                    interp_interval=0.05,
                    repetition_rate=0.96,
                    need_input_data=["real_temp",
                                     "cmd_temp", "height_gap_temp"],
                    need_output_data=["real_label"],


                    # need_input_data = ["cmd_temp","speed_gap_temp","height_gap_temp"],
                    # need_output_data = ["cmd_label"],

                    # need_input_data = ["expect_temp","height_gap_temp","real_temp"],
                    # need_output_data = ["expect_label"],

                    # need_input_data = ["expect_temp","height_gap_temp"],
                    # need_output_data = ["expect_label"],
                    interp=True,
                   difference=False,
                   multiStepOutput = [False, 4],
                    save_file=True
                )
                mergeData(
                    root_directory_path + "/"+txt_name+".txt",
                    motor_train_path + file_name,
                )
                print(file_name+" "+str(time.time() - start_time))
            mergeData("./data/"+txt_name+".txt",
                      root_directory_path + "/"+txt_name+".txt")


# formatFile(r"data\new_car2_train.txt")
# 调整时间间隔
# 修改网络输出 多时间 0.1 0.2 0.3 ，然后由滞后性判断使用哪个值
