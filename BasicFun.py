from scipy.signal import butter, lfilter
from scipy import stats
import numpy as np


fs = 1000  # 采样频率（Hz）
cutoff = 50  # 截止频率（Hz）
# 组合数据
def groupData(data, step=1):
    if_find = False
    find_index = 0
    groups = []
    for i in range(1, len(data)):
        # 检查步长
        if not if_find and i - find_index >= step:
            if_find = True
        if if_find:
            # 检查是否上升到0或下降到0
            if (data[i] == 0 and data[i-1] != 0) or (data[i-1] == 0 and data[i] != 0):
                groups.append(i)
                if_find = False
                find_index = i
    return groups


def calACC(data1, data2, thres=0.01):
    index = groupData(data2, 5)
    d1 = np.split(data1, index)
    d2 = np.split(data2, index)
    index.append(len(data2))
    acc_list = []
    for i, d in enumerate(d1):
        minus_speed = abs(d1[i] - d2[i])
        arr = (minus_speed < thres).astype(int)
        ones_ratio = np.sum(arr) / arr.size
        acc_list.append([index[i], data2[index[i]-1], ones_ratio])
        # print("贴合度:",index[i], data2[index[i]-1], ones_ratio)
    return acc_list

# 计算数据的相关性
def calCorrelation(data1, data2, delete_zero=False):
    # 将数据转换为NumPy数组
    data1 = np.array(data1)
    data2 = np.array(data2)
    if delete_zero:
        zero_indices = np.where(data1 == 0)[0]
        data1 = np.delete(data1, zero_indices)
        data2 = np.delete(data2, zero_indices)
    # 计算Pearson相关系数
    pearson_corr, _ = stats.pearsonr(data1, data2)
    # 计算Spearman相关系数
    spearman_corr, _ = stats.spearmanr(data1, data2)

    # 打印相关系数
    print("Pearson相关系数: ", pearson_corr)
    print("Spearman相关系数: ", spearman_corr)

def timeGapDistribution(timestamps):
    if len(timestamps) < 2:
        # 如果数组中少于两个时间戳，无法判断是否均匀分布
        return True
   # 计算时间戳之间的间隔
    intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

    # 计算平均间隔
    average_interval = sum(intervals) / len(intervals)

    # 检查每个间隔与平均间隔的差异是否在可接受的范围内
    interval_deviation = sum(
        abs(interval - average_interval) for interval in intervals
    ) / len(intervals)
    return interval_deviation    

def convertTensor2Numpy(tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )

#*********************数据处理***********************
# 设计滤波器
def butterLowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# 应用滤波器
def butterLowpassFilter(data, cutoff, fs, order=5):
    b, a = butterLowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 保留三位显示小数
def formatFloat(num):
    formatted_num = "{:.3f}".format(num)
    parts = formatted_num.split('.')
    if len(parts[1]) < 3:
        return formatted_num + "0" * (3 - len(parts[1]))
    else:
        return formatted_num
    
    
#*********************文件操作***********************
