import os

def process_txt_files(folder_path):
    """
    遍历指定文件夹下的所有 .txt 文件，将双数行的内容拼接到单数行的后面，
    并将修改后的内容写回到原来的文件中。
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # 确保只处理 .txt 文件
            file_path = os.path.join(folder_path, filename)
            print(f"正在处理文件：{file_path}")

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 处理内容：将双数行拼接到单数行后面
            processed_lines = []
            for i in range(1, len(lines), 2):
                if i + 1 < len(lines):  # 如果存在双数行
                    # 去掉行尾的换行符并拼接
                    combined_line = lines[i].strip() + " " + lines[i + 1].strip()+ '\n'
                    processed_lines.append(combined_line)
                else:  # 如果没有双数行，直接保留单数行
                    pass
                    # processed_lines.append(lines[i])

            # # 将处理后的内容写回到原文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(processed_lines)

            print(f"文件处理完成：{file_path}")

# 指定文件夹路径
folder_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train"
process_txt_files(folder_path)