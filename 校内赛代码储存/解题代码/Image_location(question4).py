import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体为 SimHei，适用于 xlabel, ylabel, title 等
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设备位置和爆炸点位置数据
device_locations = np.array([
    [110.241, 27.204, 824],
    [110.783, 27.456, 727],
    [110.762, 27.785, 742],
    [110.251, 28.025, 850],
    [110.524, 27.617, 786],
    [110.467, 28.081, 678],
    [110.047, 27.521, 575]
])

explosion_positions = np.array([
    [110.34377069801857, 27.29387713795242, 0.0],
    [110.61284315346103, 27.171879525699502, 849.1392796009543],
    [110.500, 27.800, 700.0],
    [110.800, 27.600, 780.0]
])

# 三维散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(device_locations[:, 0], device_locations[:, 1], device_locations[:, 2], c=device_locations[:, 2], cmap='viridis', label='设备位置')
sc2 = ax.scatter(explosion_positions[:, 0], explosion_positions[:, 1], explosion_positions[:, 2], c=explosion_positions[:, 2], cmap='plasma', marker='x', label='爆炸点')

ax.set_xlabel('经度')
ax.set_ylabel('纬度')
ax.set_zlabel('高度 (米)')
ax.set_title('设备位置与预测爆炸点的三维分布')
plt.colorbar(sc, label='设备高度 (米)')
plt.colorbar(sc2, label='爆炸点高度 (米)')
plt.legend()
plt.show()

# 二维散点图（经度-纬度）
plt.figure()
plt.scatter(device_locations[:, 0], device_locations[:, 1], c=device_locations[:, 2], cmap='viridis', label='设备位置')
plt.scatter(explosion_positions[:, 0], explosion_positions[:, 1], c=explosion_positions[:, 2], cmap='plasma', marker='x', label='爆炸点')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.title('设备位置与预测爆炸点的二维分布')
plt.colorbar(label='高度 (米)')
plt.legend()
plt.show()
