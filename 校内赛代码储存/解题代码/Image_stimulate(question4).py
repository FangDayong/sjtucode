import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体为 SimHei，适用于 xlabel, ylabel, title 等
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# Simulated device positions (longitude, latitude, height in degrees and meters)
device_positions = np.array([
    [110.34981605, 27.90985284, 481.82123524],
    [110.58028572, 27.8303345, 482.53202943],
    [110.49279758, 27.86242177, 536.90900933],
    [110.43946339, 27.65617535, 636.14039423],
    [110.26240746, 27.94097296, 594.37525839],
    [110.26239781, 27.89973279, 531.05311309],
    [110.22323344, 27.71370173, 675.33380263]
])

# Simulated sound arrival times (t1, t2, t3, t4 in seconds)
sound_arrival_times = np.array([
    [80.87942777, 65.34415916, 86.95490613, 64.40414831],
    [113.6324238, 88.71948867, 78.35953699, 49.03246073],
    [96.16137713, 71.17480207, 72.54299168, 34.13314217],
    [70.26053981, 61.49611219, 32.74341815, 103.10683488],
    [84.80535102, 79.75093274, 106.33642441, 90.05872399],
    [71.33644582, 67.41809103, 94.95022021, 89.51307107],
    [19.49712496, 54.49388892, 72.78145737, 122.65452571]
])

# Plotting device positions
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(device_positions[:, 0], device_positions[:, 1], device_positions[:, 2], c='r', marker='o')

for i, (x, y, z) in enumerate(device_positions):
    ax.text(x, y, z, f'设备 {i+1}', color='blue')

ax.set_xlabel('经度（度）')
ax.set_ylabel('纬度（度）')
ax.set_zlabel('高（米）')

# Plotting sound arrival times
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(sound_arrival_times[:, i], marker='o', label=f'设备 {i+1} 音爆震波抵达时间数据')

plt.xlabel('设备编号')
plt.ylabel('音爆到达时间（秒）')
plt.title('模拟音爆到达时间数据')
plt.legend()

# Show plots
plt.show()
