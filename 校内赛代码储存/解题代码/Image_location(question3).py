import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体为 SimHei，适用于 xlabel, ylabel, title 等
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# Device locations (longitude, latitude, height)
device_locations = np.array([
    [110.241, 27.204, 824],
    [110.783, 27.456, 727],
    [110.762, 27.785, 742],
    [110.251, 28.025, 850],
    [110.524, 27.617, 786],
    [110.467, 28.081, 678],
    [110.047, 27.521, 575]
])

# Best explosion positions (longitude, latitude, height)
explosion_positions = np.array([
    [110.43229428, 27.69959271, 773.16170389],
    [110.45972478, 27.66572359, 562.5007344],
    [110.44161337, 27.71039823, 457.62733576],
    [110.95189292, 27.59129019, 789.20322411]
])

# Create figure with two subplots: one for 2D and one for 3D
fig = plt.figure(figsize=(16, 8))

# First subplot: 2D plot
ax1 = fig.add_subplot(121)
sc1 = ax1.scatter(device_locations[:, 0], device_locations[:, 1], c=device_locations[:, 2], cmap='viridis', s=100, label='Devices', marker='o')
sc2 = ax1.scatter(explosion_positions[:, 0], explosion_positions[:, 1], c=explosion_positions[:, 2], cmap='viridis', s=100, label='Explosions', marker='x')

# Add colorbar to indicate height
cbar1 = plt.colorbar(sc1, ax=ax1, label='高 (m)')

# Add labels for devices in 2D
for i, (lon, lat) in enumerate(device_locations[:, :2]):
    ax1.text(lon + 0.01, lat + 0.01, f'Device {i+1}', color='blue')

# Add labels for explosions in 2D
for i, (lon, lat) in enumerate(explosion_positions[:, :2]):
    ax1.text(lon + 0.01, lat + 0.01, f'Explosion {i+1}', color='red')

# Add labels and title for 2D plot
ax1.set_xlabel('经度')
ax1.set_ylabel('纬度')
ax1.grid(True)

# Second subplot: 3D plot
ax2 = fig.add_subplot(122, projection='3d')
sc3 = ax2.scatter(device_locations[:, 0], device_locations[:, 1], device_locations[:, 2], c=device_locations[:, 2], cmap='viridis', s=100, label='Devices', marker='o')
sc4 = ax2.scatter(explosion_positions[:, 0], explosion_positions[:, 1], explosion_positions[:, 2], c=explosion_positions[:, 2], cmap='viridis', s=100, label='Explosions', marker='x')

# Add colorbar to indicate height
cbar2 = plt.colorbar(sc3, ax=ax2, label='高 (m)')

# Add labels for devices in 3D
for i, (lon, lat, height) in enumerate(device_locations):
    ax2.text(lon + 0.01, lat + 0.01, height + 10, f'Device {i+1}', color='blue')

# Add labels for explosions in 3D
for i, (lon, lat, height) in enumerate(explosion_positions):
    ax2.text(lon + 0.01, lat + 0.01, height + 10, f'Explosion {i+1}', color='red')

# Add labels and title for 3D plot
ax2.set_xlabel('经度')
ax2.set_ylabel('纬度')
ax2.set_zlabel('高 (m)')

# Show the plot
plt.tight_layout()
plt.show()
