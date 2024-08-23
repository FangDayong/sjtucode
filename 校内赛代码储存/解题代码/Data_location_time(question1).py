import numpy as np
from scipy.optimize import least_squares

# 震动波的传播速度 (m/s)
speed_of_sound = 340

# 监测设备的三维坐标 (经度, 纬度, 高度) 和音爆到达时间 (单位: 秒)
devices = np.array([
    [110.241, 27.204, 824, 100.767],
    [110.780, 27.456, 727, 112.220],
    [110.712, 27.785, 742, 188.020],
    [110.251, 27.825, 850, 258.985],
    [110.524, 27.617, 786, 118.443],
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
# 初始猜测：假设残骸发生音爆的初始位置在设备的第一个位置附近，音爆时间接近第一个设备的到达时间减去传播时间。
initial_guess = np.array([0, 0, 150, 4.00])

# 定义损失函数，用于最小化距离的差异
def residuals(params):
    # 提取残骸位置和时间
    x, y, z, t0 = params
    
    # 计算音爆到每个设备的理论时间
    theoretical_times = [
        t0 + np.linalg.norm([x - pos[0], y - pos[1], z - pos[2]]) / speed_of_sound
        for pos in device_positions
    ]
    
    # 返回实际到达时间与理论到达时间之间的差
    return np.array(theoretical_times) - arrival_times

# 使用信赖域反射算法（trf）进行最小二乘法求解非线性方程组，限制 t0 > 0
lower_bounds = [-np.inf, -np.inf, -np.inf, 0]  # 对 t0 设置下限为 0
upper_bounds = [np.inf, np.inf, np.inf, np.inf]  # 不限制其他参数的上限

# 使用 least_squares 进行带有边界约束的最小二乘法求解
result = least_squares(residuals, initial_guess, bounds=(lower_bounds, upper_bounds), method='trf')

# 输出结果
print(f"残骸的音爆发生位置: (x, y, z) = {result.x[:3]} 米")
print(f"音爆发生的时间: t0 = {result.x[3]} 秒")

# 提取优化结果
x, y, z, t0 = result.x

# 转换回经纬度，基于设备1的经纬度位置作为基准
reference_lon, reference_lat = devices[0, 0], devices[0, 1]

# 经度转换回度数
longitude = x / lon_conversion + reference_lon

# 纬度转换回度数
latitude = y / lat_conversion + reference_lat

# 输出残骸的音爆位置和时间
print(f"残骸的音爆发生位置 (经度, 纬度, 高度): ({longitude}, {latitude}, {z} 米)")
print(f"音爆发生的时间 t0 = {t0} 秒")
