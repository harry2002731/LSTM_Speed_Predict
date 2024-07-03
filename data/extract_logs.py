
keyword = "MotorInfo2"
keyword2 = "MotorCmd"
keyword3 = "GDMotorCmd"

import os

def read_files_in_directory(directory):
    # 用于存储所有文件内容的列表

    # 遍历目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            contents = []
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            try:
                # 打开并读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.readlines()
                    matching_lines = [line for line in content if keyword in line]
                    contents.append(matching_lines)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
            path = r"C:\Projects\Python\speed_control\real\\"+ f"{file}.txt"
            with open(path, "a") as f:
                f.truncate(0)
                for item in contents[0]:
                    item.replace("\n", "")
                    f.write(str(item))
                    # f.write("\n")

def extract(directory):
    # 遍历目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            contents = []
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            try:
                # 打开并读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.readlines()
                    matching_lines = [line for line in content if (keyword2 in line) and (keyword3 not in line)]
                    contents.append(matching_lines)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
            path = r"C:\Projects\Python\speed_control\cmd\\"+ f"{file}.txt"
            with open(path, "a") as f:
                f.truncate(0)
                for item in contents[0]:
                    item.replace("\n", "")
                    f.write(str(item))

# 指定要遍历的文件夹路径
directory_path = r'C:\Projects\Python\speed_control\data_set'

# 调用函数并打印结果
# contents_list = read_files_in_directory(directory_path)
contents_list = read_files_in_directoryCMD(directory_path)

# print(contents_list)