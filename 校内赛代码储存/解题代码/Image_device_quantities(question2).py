import matplotlib.pyplot as plt
import numpy as np

# Simulated results from previous runs (mock data for visualization)
device_counts = [3, 4, 5, 6, 7]
mean_positions = [
    [36006.8973, 4652.62842, -1904.67058, 18.1983083],
    [34838.2789, 1231.38036, -99.3100364, 9.91790758],
    [36226.4166, -935.97407, 795.888772, 7.55066673],
    [35752.4742, -3207.84827, 837.91464, 1.51260323],
    [36181.8261, -3573.81922, 849.158427, 8.58119456e-21]
]
std_positions = [
    [15339.0089405, 13781.5942725, 7179.36929903, 33.95952961],
    [6457.69589713, 9043.536792, 5222.21576424, 17.51652891],
    [2662.66439826, 8364.57197914, 165.79887435, 20.08686059],
    [1860.63031073, 2606.46649834, 85.26425715, 3.48008336],
    [1.10461591e-04, 6.00110775e-04, 4.25061126e-02, 2.00703960e-21]
]

# Extract x, y, z (latitude, longitude, height) and time means and std deviations
x_mean = [mean[0] for mean in mean_positions]
y_mean = [mean[1] for mean in mean_positions]
z_mean = [mean[2] for mean in mean_positions]
time_mean = [mean[3] for mean in mean_positions]

x_std = [std[0] for std in std_positions]
y_std = [std[1] for std in std_positions]
z_std = [std[2] for std in std_positions]
time_std = [std[3] for std in std_positions]

# Create subplots for x, y, z, and time
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))

# Plot x (longitude) mean and std
ax1.errorbar(device_counts, x_mean, yerr=x_std, fmt='-o', capsize=5, label='经度 平均值 ± 标准差 ')
ax1.set_xlabel('设备数量')
ax1.set_ylabel('经度 (m)')
ax1.legend()

# Plot y (latitude) mean and std
ax2.errorbar(device_counts, y_mean, yerr=y_std, fmt='-o', capsize=5, label='纬度 平均值 ± 标准差 ')
ax2.set_xlabel('设备数量')
ax2.set_ylabel('纬度 (m)')
ax2.legend()

# Plot z (height) mean and std
ax3.errorbar(device_counts, z_mean, yerr=z_std, fmt='-o', capsize=5, label='高度 平均值 ± 标准差 ')
ax3.set_xlabel('设备数量')
ax3.set_ylabel('高度 (m)')
ax3.legend()

# Plot time mean and std
ax4.errorbar(device_counts, time_mean, yerr=time_std, fmt='-o', capsize=5, label='时间 平均值 ± 标准差 ', color='orange')
ax4.set_xlabel('设备数量')
ax4.set_ylabel('时间 (s)')
ax4.legend()

# Show plots
plt.tight_layout()
plt.show()
