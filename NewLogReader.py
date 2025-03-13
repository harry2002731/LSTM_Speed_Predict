import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout,QSizePolicy, QVBoxLayout,QGridLayout,QSpacerItem, QWidget, QSlider, QPushButton, QTextEdit, QCheckBox, QFileDialog, QButtonGroup, QLabel
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import os
from scipy.spatial.distance import euclidean
from PyQt5.QtCore import pyqtSignal
import copy
similarity_threshold = 0.2  # 相似度阈值
# motor_info = ["motor_t", "motor_real", "motor_cmd", "motor_expect", "motor_height_gap", "motor_speed_gap","orin_cmd","predict_expect","weight","predict_weight_index"]
motor_info = ["motor_t", "motor_real", "motor_cmd", "motor_expect", "motor_height_gap", "motor_speed_gap","orin_cmd","predict_expect","weight","predict_weight_index","0.0T","0.3T","0.6T","0.9T","1.2T","1.5T","error"]

def get_py_files(folder_path):
    """获取指定文件夹下的所有 .py 文件的名称"""
    py_files = []
    # 遍历文件夹中的所有文件和子文件夹
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)) and file.endswith('.txt'):
            py_files.append(file)
    return py_files

def extract_lines(input_file, output_file, start_line, end_line):
    try:
        # 打开源文件并读取所有行
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 提取指定行
        extracted_lines = lines[start_line:end_line + 1]
        
        # 打开目标文件并写入提取的行
        with open(output_file, 'a', encoding='utf-8') as f:
            f.writelines(extracted_lines)
        
        print(f"已提取第 {start_line + 1} 行到第 {end_line + 1} 行（从 1 开始计数），并保存到 {output_file}。")
    except Exception as e:
        print(f"发生错误：{e}")


def sort_and_deduplicate_timestamps(input_file, output_file):
    # 读取文件内容
    with open(input_file, 'r') as f:
        lines = f.readlines()
    # 解析时间戳并存储原始行
    timestamp_info = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue  # 跳过空行
        try:
            timestamp_str = line.split(" ")[0]  # 提取时间戳字符串
            timestamp_info.append({
                'timestamp': float(timestamp_str),
                'timestamp_str': timestamp_str,
                'raw_line': line,
                'line_num': line_num  # 保留原始行号
            })
        except ValueError as e:
            print(f"警告：第 {line_num} 行格式错误或时间戳无效：{line}")
    
    # 按时间戳排序
    sorted_timestamps = sorted(timestamp_info, key=lambda x: x['timestamp'])
    # 去重并保留原始行内容
    deduplicated_lines = []
    prev_timestamp = None
    for ts_info in sorted_timestamps:
        current_timestamp = ts_info['timestamp']
        if current_timestamp != prev_timestamp:
            deduplicated_lines.append(ts_info)
            prev_timestamp = current_timestamp
    
    # 将结果写入输出文件
    with open(output_file, 'w') as f:
        for info in deduplicated_lines:
            f.write(f"{info['raw_line']}\n")
    
    print(f"处理完成！已将结果写入到 {output_file}。共有 {len(deduplicated_lines)} 条有效记录。")
    

# 定义绘图类，包含交互式矩形框选择功能
class InteractiveCanvas(FigureCanvas):
    merged_ranges_updated = pyqtSignal(list)

    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        # RectangleSelector
        self.rs = RectangleSelector(
            self.ax,
            self.onselect,
            useblit=False,
            button=[1],  # 使用鼠标左键
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,  # 启用交互模式
            handle_props=dict(marker='s', markersize=8, color='blue')  # 显示角点
        )
        self.selected_areas = []  # 保存框选区域的坐标
        self.selected = None  # 保存框选的点
        self.merged_ranges = []
        self.vline = None  # 垂直线
        self.hline = None  # 水平线
        self.cid = self.mpl_connect('motion_notify_event', self.on_mouse_move)  # 鼠标移动事件

    def on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            # 清除之前的十字线
            if self.vline:
                self.vline.remove()
            if self.hline:
                self.hline.remove()
            # 绘制新的十字线
            self.vline = self.ax.axvline(x, color='gray', linestyle='--', linewidth=1)
            self.hline = self.ax.axhline(y, color='gray', linestyle='--', linewidth=1)
            self.draw()

    def initRectangleSelector(self):
        self.rs = RectangleSelector(
            self.ax,
            self.onselect,
            useblit=False,
            button=[1],  # 使用鼠标左键
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,  # 启用交互模式
            handle_props=dict(marker='s', markersize=8, color='blue')  # 显示角点
        )

    def onselect(self, eclick, erelease):
        # 框选区域确定
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # 清除之前的矩形框
        self.clear_selection()
        
        # 绘制框选区域
        self.ax.axvspan(x1, x2, alpha=0.3, color='red')
        self.draw()
        
        # 假设 x 和 y 是全局数据
        global x, y
        selected = [
            (xi, yi) for xi, yi in zip(x, y)
            if (x1 < xi < x2) and (y1 < yi < y2)
        ]
        
        # 更新选中的点
        self.selected = selected
        
        # 调用 update_plot 绘制选中的点
        self.update_plot(color='red')

        # 提取框选区域内的线段
        start_idx = np.searchsorted(x, x1)
        end_idx = np.searchsorted(x, x2)
        selected_x = x[start_idx:end_idx]
        selected_y = y[start_idx:end_idx]
        
        similar_select = []  # 收集相似线段的x范围

        # 找到与框选线段相似的其他线段
        for i in range(len(x) - len(selected_x)):
            segment_x = x[i:i + len(selected_x)]
            segment_y = y[i:i + len(selected_x)]
            if euclidean(selected_y, segment_y) < similarity_threshold:
                self.ax.plot(segment_x, segment_y, color='red', linewidth=2)  # 标记相似线段为红色
                similar_select.append((segment_x[0], segment_x[-1]))
        
        # 合并有交集的相似线段区间
        self.merged_ranges = []
        for current in similar_select:
            if not self.merged_ranges:
                self.merged_ranges.append(current)
            else:
                last = self.merged_ranges[-1]
                if current[0] <= last[1]:  # 有交集或相邻
                    self.merged_ranges[-1] = (last[0], max(last[1], current[1]))
                else:
                    self.merged_ranges.append(current)
        self.merged_ranges_updated.emit(self.merged_ranges)
        self.copy_merged_ranges = copy.deepcopy(self.merged_ranges)

        # 打印合并后的相似线段x范围
        if self.merged_ranges:
            print("Merged similar segments at x ranges:")
            for start, end in self.merged_ranges:
                print(f" - {start} to {end}")
        else:
            print("No similar segments found.")

    def update_plot(self, color='red'):
        # 清除之前绘制的选中点
        lines_to_remove = []
        for line in self.ax.lines[::-1]:  # 从后往前遍历，避免索引问题
            if line.get_color() == color:
                lines_to_remove.append(line)
        
        # 移除这些线条
        for line in lines_to_remove:
            line.remove()
        
        # 如果有选中的点，绘制它们
        if self.selected:
            selected_x, selected_y = zip(*self.selected)
            self.ax.plot(selected_x, selected_y, 'o', color=color, markersize=5)

        # 更新图表
        self.draw()
    def update_plot2(self,index, color='red'):
        # 清除之前绘制的选中点
        lines_to_remove = []
        for line in self.ax.lines[::-1]:  # 从后往前遍历，避免索引问题
            if line.get_color() == color and self.merged_ranges[index][0] < line.get_xdata()[1] < self.merged_ranges[index][1]:
                lines_to_remove.append(line)
        self.copy_merged_ranges.pop(self.copy_merged_ranges.index(self.merged_ranges[index]))
        
        # 移除这些线条
        for line in lines_to_remove:
            line.remove()

        # 更新图表
        self.draw()
        
    def d(self,index):
        selected_x = x[int(self.merged_ranges[index][0]):int(self.merged_ranges[index][1])]
        selected_y = y[int(self.merged_ranges[index][0]):int(self.merged_ranges[index][1])]
        self.ax.plot(selected_x,selected_y, color='red', linewidth=2)  # 标记相似线段为红色
        self.draw()
        self.copy_merged_ranges.append(self.merged_ranges[index])
        

    def clear_selection(self):
        # 清除所有矩形框
        for patch in self.ax.patches[::-1]:  # 从后往前遍历，避免索引问题
            patch.remove()
        self.draw()
        
        
# 主窗口类
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口属性
        self.setWindowTitle("Real-Time Plot with PyQt5")
        self.setGeometry(100, 100, 1200, 800)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()  # 主布局横向排列

        # 左侧图表区域
        self.canvas = InteractiveCanvas()
        self.canvas.setParent(main_widget)
        self.toolbar = NavigationToolbar(self.canvas, main_widget)  # 添加工具栏

        left_layout = QVBoxLayout()  # 左侧图表区域垂直布局
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(self.toolbar)

        # 右侧控件区域
        self.control_widget = QWidget()
        self.control_widget.setMinimumWidth(300)  # 设置右侧控件最小宽度
        self.control_widget.setMaximumWidth(300)  # 设置右侧控件最大宽度
        right_layout = QVBoxLayout()  # 右侧控件垂直布局
        self.control_widget.setLayout(right_layout)
        main_layout.addLayout(left_layout)  # 添加左侧图表布局
        main_layout.addWidget(self.control_widget)  # 添加右侧控件区域

        main_widget.setLayout(main_layout)
        self.orin_path = "/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train/"
        files = get_py_files(self.orin_path)
        self.index = 0
        self.files = files
        self.file_name = self.files[self.index]
        self.checkBoxList = []
        self.show_dict = {str(i): False for i in range(len(motor_info))}
        # 添加右侧控件
        self.add_controls(right_layout)
        self.load_data()


        # 连接 merged_ranges_updated 信号到槽函数
        self.canvas.merged_ranges_updated.connect(self.update_merged_ranges_checkboxes)
        
    def update_merged_ranges_checkboxes(self, merged_ranges):
        # 清除之前的 CheckBox
        while self.merged_ranges_checkbox_layout.count() > 0:
            item = self.merged_ranges_checkbox_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # 动态添加新的 CheckBox
        for i, (start, end) in enumerate(merged_ranges):
            checkbox = QCheckBox(f"Range {i + 1}: {int(start)} to {int(end)}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, name= i : self.updateStatus2(name,state))
            self.merged_ranges_checkbox_layout.addWidget(checkbox)

        # 更新右侧控件布局
        self.control_widget.layout().update()
        
    def updateStatus2(self,index,state):
        if state:
            self.canvas.d(index=index)
        else:
            self.canvas.update_plot2(index)
        
        
    def addCheckBox(self,index,name,layout):
        checkbox = QCheckBox(name)
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(lambda state, name=index: self.updateStatus(name,state))
        layout.addWidget(checkbox)
        
        return checkbox
    def updateStatus(self,key,state,init = False):
        key = str(key)
        if key in self.show_dict:
            if init and self.show_dict[key]:
                pass
            else:
                self.show_dict[key] = state
                print(f"Key '{key}' value toggled to: {self.show_dict[key]}")
        else:
            print(f"Key '{key}' not found in the dictionary.")
        temp = []
        for i in self.show_dict.items():
            if i[1]:
                temp.append(int(i[0]))
        file_path = self.orin_path + self.files[self.index]
        total_data_list = []
        with open(file_path, "r") as f:
            line = f.readline()  # 读取第一行
            while line:
                line = line.replace(' \n', '').replace(' \r', '')
                data_list = line.split(" ")
                if " " in data_list:
                    data_list.remove(" ")
                if '' in data_list:
                    data_list.remove('')
                data_list = [float(item) for item in data_list]
                total_data_list.append(data_list)
                line = f.readline()  # 读取下一行

        total_data_array = np.array(total_data_list).T
        
        # for i in range(7):
        #     count = sum(1 for item in total_data_array[9] if item == i/10)
        #     print(f"元素 {i} 出现了 {count} 次")  
        # print(f"*******************************")  
        self.visualizeDataset(total_data_array, title_name = "file_name", visual_indexs=temp, compare_indexs=[1,3], name_list=[motor_info[i] for i in temp])

        
        
    def add_controls(self, layout):
        grid_layout = QGridLayout()
        for i in range(len(motor_info)):
            cb = self.addCheckBox(i,motor_info[i],layout=grid_layout)
            self.checkBoxList.append(cb)
        layout.addLayout(grid_layout)  # 将按钮布局添加到右侧控件布局中

        # 按钮放在同一行
        button_layout1 = QHBoxLayout()  # 创建一个水平布局
        btn_load = QPushButton("Load Data")
        btn_folder = QPushButton("Load Folder")

        btn_load.clicked.connect(self.load_data)
        btn_clear = QPushButton("Clear Select")
        btn_clear.clicked.connect(self.canvas.clear_selection)
        btn_save = QPushButton("Save Select")
        btn_save.clicked.connect(self.save_selection)
        button_layout1.addWidget(btn_load)
        button_layout1.addWidget(btn_folder)

        layout.addLayout(button_layout1)  # 将按钮布局添加到右侧控件布局中

        # 加载文件夹按钮
        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(btn_clear)
        button_layout2.addWidget(btn_save)
        btn_folder.clicked.connect(self.load_folder)
        layout.addLayout(button_layout2)

        # 滑动条（横着的）
        slider = QSlider(Qt.Horizontal)  # 横向滑动条
        slider.setRange(0, 10)
        slider.setValue(2)
        slider.setTickInterval(1)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.valueChanged.connect(self.update_plot)
        layout.addWidget(slider)
        self.merged_ranges_checkbox_layout = QVBoxLayout()
        self.control_widget.layout().addLayout(self.merged_ranges_checkbox_layout)
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addSpacerItem(spacer)

    def update_plot(self, value):
        # 更新图表
        global similarity_threshold
        similarity_threshold = value/10

    def load_data(self):
        self.index += 1
        self.updateStatus(1,True)
        self.updateStatus(2,True)
        for i,cb in enumerate(self.checkBoxList):
            cb.setChecked(self.show_dict[str(i)])
        

    def visualizeDataset(self, dataset, title_name="", visual_indexs=[], compare_indexs=[], name_list=[], nrow=0):
        global x, y
        colors = [
            "xkcd:blue",
            "xkcd:grass green",
            "xkcd:goldenrod",
            "xkcd:forest green",
            "xkcd:sky blue",
            "xkcd:light red",
            "xkcd:bright pink",
            "xkcd:lavender",
            "xkcd:ocean blue",
            "xkcd:mud",
            "xkcd:eggplant",
            "xkcd:cyan",
            "xkcd:slate blue",
            "xkcd:peach",
            "xkcd:coral",
            "xkcd:blue",
            "xkcd:grass green",
            "xkcd:goldenrod",
            "xkcd:forest green",
            "xkcd:sky blue",
            "xkcd:light red",
            "xkcd:bright pink"
        ]

        if len(visual_indexs) > 0:
            x = np.linspace(0, len(dataset[visual_indexs[0]]), len(dataset[visual_indexs[0]]))
            y = dataset[1]
            current_ylim = self.canvas.ax.get_ylim()
            self.canvas.clear_selection()
            self.canvas.ax.clear()
            self.canvas.initRectangleSelector()
            max_y = 0
            min_y = 0

            # 存储每条曲线的标签和颜色
            lines = []
            labels = []

            for index in visual_indexs:
                color = colors[index % len(colors)]  # 确保索引不会超出范围
                if index == 9:
                    line, = self.canvas.ax.plot(x, dataset[index]/10, marker='o', linestyle='', markersize=3,color=color, label=f"{motor_info[index]}")
                elif index == 8:
                    line, = self.canvas.ax.plot(x, dataset[index]/10, linewidth=3.0, color=color, label=f"{motor_info[index]}")
                elif index >= 10:
                    line, = self.canvas.ax.plot(x, dataset[index]/(1000), linewidth=3.0, color=color, label=f"{motor_info[index]}")
                else:
                    line, = self.canvas.ax.plot(x, dataset[index], linewidth=3.0, color=color, label=f"{motor_info[index]}")
                lines.append(line)
                labels.append(line.get_label())
                max_y = max(dataset[index]) if max(dataset[index]) > max_y else max_y
                min_y = min(dataset[index]) if min(dataset[index]) < min_y else min_y
            # self.canvas.ax.set_ylim((min_y, max_y))
            self.canvas.ax.legend(lines, labels)  # 使用明确的线条和标签
            self.canvas.draw()

    def load_folder(self):
        # 加载文件夹
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "/home")
        if folder_path:
            # 默认选择所有 .txt 文件
            files = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.txt')]
            self.text_edit.setText(f"Loaded folder: {folder_path}\nFiles: {files}")
    def save_selection(self):
        in_file_path = r"/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/train/" + self.files[self.index]
        out_file_path = r"/home/ubuntu/Desktop/project/LSTM_Speed_Predict/data/roboshop_data/compared_data/error/" + self.files[self.index]
        for start, end in self.canvas.copy_merged_ranges:
            print(f" - {start} to {end}")
            extract_lines(in_file_path, out_file_path, int(start), int(end))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())