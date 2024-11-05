# from tsaug import TimeWarp, Quantize, Pool, Crop, Drift, AddNoise, Convolve, Dropout
# import numpy as np
# import matplotlib.pyplot as plt

# # 假设我们有以下简单的时间序列数据
# t = np.linspace(0, 2 * np.pi, 100)
# data = np.sin(t)

# # 使用不同的增强器对数据进行增强
# augmenter = TimeWarp(n_speed_change=3, max_speed_ratio=1.5)
# augmented_data = augmenter.augment(data)

# cropper = Crop(size=50)
# cropped_data = cropper.augment(data)

# quantizer = Quantize(n_levels=10)
# quantized_data = quantizer.augment(data)

# drifter = Drift(max_drift=0.5, n_drift_points=10)
# drifted_data = drifter.augment(data)

# noiser = AddNoise(scale=0.1)
# noisy_data = noiser.augment(data)

# convolver = Convolve(size=7)
# convolved_data = convolver.augment(data)

# dropout = Dropout(p=0.2)
# dropped_data = dropout.augment(data)

# pooler = Pool(size=10, kind='max')
# pooled_data = pooler.augment(data)

# # 绘制原始数据和增强后的数据进行对比
# plt.figure()
# plt.plot(data, label='source')
# plt.plot(augmented_data, label='aug_timewarp')
# plt.plot(pooled_data, label='aug_pool')
# plt.plot(quantized_data, label='aug_quantized')
# plt.plot(drifted_data, label='aug_drifted')
# plt.plot(noisy_data, label='aug_noisy')
# plt.plot(convolved_data, label='aug_conv')
# plt.plot(dropped_data, label='aug_drop')
# plt.legend()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# 系统参数
N = 5  # 预测区间
k_steps = 100

# 状态矩阵A和输入矩阵B
A = np.array([[1, 0.1], [-1, 2]])
B = np.array([[0.2, 1], [0.5, 2]])
n = A.shape[0]
p = B.shape[1]

# Q、F、R矩阵
Q = np.diag([100, 1])
F = np.diag([100, 1])
R = np.diag([1, 0.1])

# 定义step的数量
k_steps = 100

# 开辟所有状态x的存储空间并初始状态
X_k = np.zeros((n, k_steps+1))
X_k[:,0] = np.array([20,-20])
# 开辟所有控制输入u的存储空间
U_k = np.zeros((p, k_steps))

# 计算QP中代价函数相关的矩阵
def get_QPMatrix(A, B, Q, R, F, N):
    M = np.vstack([np.eye(n), np.zeros((N*n, n))])
    C = np.zeros(((N+1)*n, N*p))
    temp = np.eye(n)
    for i in range(1,N+1):
        rows = i * n + np.arange(n)
        C[rows,:] = np.hstack([temp @ B, C[rows-n, :-p]])
        temp = A @ temp
        M[rows,:] = temp

    Q_ = np.kron(np.eye(N), Q)
    rows_Q, cols_Q = Q_.shape
    rows_F, cols_F = F.shape
    Q_bar = np.zeros((rows_Q+rows_F, cols_Q+cols_F))
    Q_bar[:rows_Q, :cols_Q] = Q_
    Q_bar[rows_Q:, cols_Q:] = F
    R_bar = np.kron(np.eye(N), R)

    # G = M.T @ Q_bar @ M
    E = C.T @ Q_bar @ M
    H = C.T @ Q_bar @ C + R_bar
    return E, H


# 定义MPC优化问题
def mpc_prediction(x_k, E, H, N, p):
    # 定义优化变量
    U = ca.SX.sym('U', N * p)
    # 定义目标函数
    objective = 0.5 * ca.mtimes([U.T, H, U]) + ca.mtimes([U.T, E, x_k])
    qp = {'x': U, 'f': objective}
    opts = {'print_time': False, 'ipopt': {'print_level': 0}}
    solver = ca.nlpsol('solver', 'ipopt', qp, opts)

    # 求解问题
    sol = solver()
    # 提取最优解
    U_k = sol['x'].full().flatten()
    u_k = U_k[:p]  # 取第一个结果

    return u_k

if __name__ == "__main__":
    # Get QP Matrix
    E, H = get_QPMatrix(A, B, Q, R, F, N)
    # Simulation
    for i in range(k_steps):
        x_k = X_k[:,i]
        u_k = mpc_prediction(x_k, E, H, N, p)
        x_k = A @ x_k + B @ u_k
        X_k[:,i+1] = x_k
        U_k[:,i] = u_k

    # 绘制结果
    plt.subplot(2, 1, 1)
    for i in range(X_k.shape[0]):
        plt.plot(X_k[i, :], label=f"x{i+1}")
    plt.legend()
    plt.title("State Variables")
    plt.xlabel("Time Step")
    plt.ylabel("State Value")
    
    # 第二个子图: 控制输入
    plt.subplot(2, 1, 2)
    for i in range(U_k.shape[0]):
        plt.plot(U_k[i, :], label=f"u{i+1}")
    plt.legend()
    plt.title("Control Inputs")
    plt.xlabel("Time Step")
    plt.ylabel("Control Value")
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()
