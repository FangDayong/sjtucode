import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# 震动波的传播速度 (m/s)
speed_of_sound = 340

# 监测设备的三维坐标 (经度, 纬度, 高度) 和音爆到达时间 (单位: 秒)
devices = np.array([
    [110.241, 27.204, 824, 100.767],
    [110.780, 27.456, 727, 112.220],
    [110.712, 27.785, 742, 188.020],
    [110.251, 27.825, 850, 258.985],
    [110.467, 27.921, 678, 266.871],
    [110.047, 27.121, 575, 163.024]
])

# 经纬度转换为米，纬度间每度 111263 米，经度间每度 97304 米
lat_conversion = 111263
lon_conversion = 97304

# 转换经纬度为米并计算距离
device_positions = np.array([
    [lon_conversion * (device[0] - devices[0, 0]), 
     lat_conversion * (device[1] - devices[0, 1]), 
     device[2]]
    for device in devices
])

# 音爆到达时间
arrival_times = devices[:, 3]

# 定义残骸位置和音爆时间的未知量
initial_guess = np.array([0, 0, 150, 4.00])

# 定义损失函数，用于最小化距离的差异
def residuals(params):
    x, y, z, t0 = params
    
    # 计算音爆到每个设备的理论时间
    theoretical_times = [
        t0 + np.linalg.norm([x - pos[0], y - pos[1], z - pos[2]]) / speed_of_sound
        for pos in device_positions
    ]
    
    # 返回实际到达时间与理论到达时间之间的差
    return np.array(theoretical_times) - arrival_times

# 使用 least_squares 进行带有边界约束的最小二乘法求解
lower_bounds = [-np.inf, -np.inf, -np.inf, 0]  # 对 t0 设置下限为 0
upper_bounds = [np.inf, np.inf, np.inf, np.inf]  # 不限制其他参数的上限

# 使用 least_squares 进行带有边界约束的最小二乘法求解
result = least_squares(residuals, initial_guess, bounds=(lower_bounds, upper_bounds), method='trf')

# 提取优化结果
x, y, z, t0 = result.x

# 计算每个设备的距离误差
distance_errors = np.linalg.norm(device_positions - np.array([x, y, z]), axis=1)

# 绘制距离误差图
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(distance_errors) + 1), distance_errors)
plt.xlabel('设备编号')
plt.ylabel('距离误差 (米)')
plt.title('每个设备的距离误差')

# 展示图表
plt.show()
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# 震动波的传播速度 (m/s)
speed_of_sound = 340

# 监测设备的三维坐标 (经度, 纬度, 高度) 和音爆到达时间 (单位: 秒)
devices = np.array([
    [110.241, 27.204, 824, 100.767],
    [110.780, 27.456, 727, 112.220],
    [110.712, 27.785, 742, 188.020],
    [110.251, 27.825, 850, 258.985],
    [110.467, 27.921, 678, 266.871],
    [110.047, 27.121, 575, 163.024]
])

# 经纬度转换为米，纬度间每度 111263 米，经度间每度 97304 米
lat_conversion = 111263
lon_conversion = 97304

# 转换经纬度为米并计算距离
device_positions = np.array([
    [lon_conversion * (device[0] - devices[0, 0]), 
     lat_conversion * (device[1] - devices[0, 1]), 
     device[2]]
    for device in devices
])

# 音爆到达时间
arrival_times = devices[:, 3]

# 定义残骸位置和音爆时间的未知量
initial_guess = np.array([0, 0, 150, 4.00])

# 定义损失函数，用于最小化距离的差异
def residuals(params):
    x, y, z, t0 = params
    
    # 计算音爆到每个设备的理论时间
    theoretical_times = [
        t0 + np.linalg.norm([x - pos[0], y - pos[1], z - pos[2]]) / speed_of_sound
        for pos in device_positions
    ]
    
    # 返回实际到达时间与理论到达时间之间的差
    return np.array(theoretical_times) - arrival_times

# 使用 least_squares 进行带有边界约束的最小二乘法求解
lower_bounds = [-np.inf, -np.inf, -np.inf, 0]  # 对 t0 设置下限为 0
upper_bounds = [np.inf, np.inf, np.inf, np.inf]  # 不限制其他参数的上限

# 使用 least_squares 进行带有边界约束的最小二乘法求解
result = least_squares(residuals, initial_guess, bounds=(lower_bounds, upper_bounds), method='trf')

# 提取优化结果
x, y, z, t0 = result.x

# 计算每个设备的理论到达时间
predicted_times = [
    t0 + np.linalg.norm([x - pos[0], y - pos[1], z - pos[2]]) / speed_of_sound
    for pos in device_positions
]

# 计算每个设备的时间误差
time_errors = np.array(predicted_times) - arrival_times

# 绘制时间误差图
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(time_errors) + 1), time_errors)
plt.xlabel('设备编号')
plt.ylabel('时间误差 (秒)')
plt.title('每个设备的时间误差')

# 展示图表
plt.show()
