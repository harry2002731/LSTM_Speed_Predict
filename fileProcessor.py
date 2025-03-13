import os

def mergeData(write_path, read_path):
    # 读取源文件的全部内容
    with open(read_path, "r", encoding="utf-8") as source_file:
        content = source_file.read()
    # 将读取的内容写入目标文件
    with open(write_path, "a", encoding="utf-8") as target_file:
        target_file.write(content)

        # print("finsh")

def find_latest_file(directory):
    latest_file = None
    latest_time = 0
    last_file_name = ""

    # 遍历目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 确保是文件而不是文件夹
        if os.path.isfile(file_path):
            # 获取文件的最后修改时间
            file_modified_time = os.path.getmtime(file_path)

            # 检查是否是最新的文件
            if file_modified_time > latest_time:
                latest_time = file_modified_time
                latest_file = file_path
                last_file_name = filename

    return last_file_name