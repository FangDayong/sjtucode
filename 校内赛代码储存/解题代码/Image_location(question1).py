import numpy as np
from scipy.optimize import least_squares
import matplotlib as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体为 SimHei，适用于 xlabel, ylabel, title 等
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 残骸位置（x, y, z）和设备位置数据
wreck_position = [x, y, z]
device_positions_3d = np.array([
    [lon_conversion * (device[0] - devices[0, 0]), 
     lat_conversion * (device[1] - devices[0, 1]), 
     device[2]]
    for device in devices
])

# 创建3D图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 设备位置数据
device_x = device_positions_3d[:, 0]
device_y = device_positions_3d[:, 1]
device_z = device_positions_3d[:, 2]

# 绘制设备位置，使用颜色映射根据高度显示颜色深浅
scatter = ax.scatter(device_x, device_y, device_z, c=device_z, cmap='coolwarm', s=100, label='设备位置')

# 绘制残骸位置
ax.scatter(wreck_position[0], wreck_position[1], wreck_position[2], color='red', s=200, marker='*', label='残骸位置')

# 添加颜色条
cbar = fig.colorbar(scatter, ax=ax, label='高度 (米)')

# 设置轴标签
ax.set_xlabel('经度 (米)')
ax.set_ylabel('纬度 (米)')
ax.set_zlabel('高度 (米)')
ax.set_title('残骸及设备位置的三维立体示意图')

# 添加图例
ax.legend()

# 显示图形
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# 设备的经纬度和高度
device_lat_lon = np.array([
    [110.241, 27.204],  # 设备1的经纬度
    [110.780, 27.456],
    [110.712, 27.785],
    [110.251, 27.825],
    [110.524, 27.617],
    [110.467, 27.921],
    [110.047, 27.121]
])

# 提取经度和纬度
device_lon = device_lat_lon[:, 0]
device_lat = device_lat_lon[:, 1]
device_z = np.array([824, 727, 742, 850, 786, 678, 575])  # 高度 (米)

# 残骸位置（绝对经纬度）
wreck_lat_lon = [110.6103, 27.1222]  # 示例残骸经纬度

# 创建二维平面图，使用经纬度为轴
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制设备位置，使用颜色映射表示高度
scatter = ax.scatter(device_lon, device_lat, c=device_z, cmap='coolwarm', s=100, label='设备位置')

# 绘制残骸位置
ax.scatter(wreck_lat_lon[0], wreck_lat_lon[1], color='red', s=200, marker='*', label='残骸位置')

# 添加颜色条，用于表示高度
cbar = fig.colorbar(scatter, ax=ax, label='高度 (米)')

# 设置轴标签和标题
ax.set_xlabel('经度')
ax.set_ylabel('纬度')
ax.set_title('残骸及设备位置的经纬度平面图')

# 添加图例
ax.legend()

# 显示图形
plt.show()
