import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean

# Constants
sound_speed = 340  # m/s
lat_to_meters = 111263  # meters per degree latitude
lon_to_meters = 97304   # meters per degree longitude

# Convert longitude, latitude, height to meters
def convert_to_meters(lon, lat, height):
    x = lon * lon_to_meters
    y = lat * lat_to_meters
    z = height  # Height is already in meters
    return np.array([x, y, z])

# Simulated device positions (converted to meters)
device_positions = np.array([
    convert_to_meters(110.34981605, 27.90985284, 481.82123524),
    convert_to_meters(110.58028572, 27.8303345, 482.53202943),
    convert_to_meters(110.49279758, 27.86242177, 536.90900933),
    convert_to_meters(110.43946339, 27.65617535, 636.14039423),
    convert_to_meters(110.26240746, 27.94097296, 594.37525839),
    convert_to_meters(110.26239781, 27.89973279, 531.05311309),
    convert_to_meters(110.22323344, 27.71370173, 675.33380263)
])

# True explosion positions for validation (converted to meters)
true_explosion_positions = np.array([
    convert_to_meters(110.24881132, 27.71875567, 830.17912749),
    convert_to_meters(110.34201478, 27.76222241, 782.72874308),
    convert_to_meters(110.41989654, 27.70311125, 495.09025456),
    convert_to_meters(110.51613294, 27.9112601, 797.64233002)
])

# Simulated sound arrival times (in seconds)
simulated_arrival_times = np.array([
    [80.87942777, 65.34415916, 86.95490613, 64.40414831],
    [113.6324238, 88.71948867, 78.35953699, 49.03246073],
    [96.16137713, 71.17480207, 72.54299168, 34.13314217],
    [70.26053981, 61.49611219, 32.74341815, 103.10683488],
    [84.80535102, 79.75093274, 106.33642441, 90.05872399],
    [71.33644582, 67.41809103, 94.95022021, 89.51307107],
    [19.49712496, 54.49388892, 72.78145737, 122.65452571]
])

# True explosion times for validation
true_explosion_times = np.array([11.97872243, 16.97330964, 16.38970079, 16.7949202])

# Calculate predicted arrival times based on predicted explosion positions and times
def calculate_predicted_arrival_times(device_positions, explosion_positions, explosion_times):
    predicted_arrival_times = []
    for device_position in device_positions:
        times = []
        for i in range(len(explosion_positions)):
            distance = np.linalg.norm(device_position - explosion_positions[i])
            time_delay = distance / sound_speed
            arrival_time = explosion_times[i] + time_delay
            times.append(arrival_time)
        predicted_arrival_times.append(times)
    return np.array(predicted_arrival_times)

# Objective function to minimize the arrival time error (RMSE)
def objective_function(explosion_positions_flat, explosion_times, device_positions, simulated_arrival_times, true_explosion_positions):
    explosion_positions = explosion_positions_flat.reshape(4, 3)
    
    # Calculate predicted arrival times
    predicted_arrival_times = calculate_predicted_arrival_times(device_positions, explosion_positions, explosion_times)
    
    # Calculate RMSE for the arrival times
    arrival_time_errors = simulated_arrival_times - predicted_arrival_times
    rmse = np.sqrt(np.mean(arrival_time_errors**2))
    
    # Calculate position error for each predicted explosion position
    position_errors = np.linalg.norm(explosion_positions - true_explosion_positions, axis=1)
    
    # Enforce the position error constraint (penalty if > 1000 meters)
    penalty = np.sum(np.maximum(0, position_errors - 1000)**2)
    
    return rmse + penalty  # Minimize RMSE and penalize large position errors

# Flatten the true explosion positions to use as an initial guess for optimization
initial_guess = true_explosion_positions.flatten()

# 设置更多迭代次数和更小的收敛阈值
options = {'maxiter': 10000, 'disp': True}  # 增加最大迭代次数至1000，打开输出显示优化进程

# Perform optimization (minimize RMSE and position error constraint)
# Perform optimization (minimize RMSE and position error constraint)
result = minimize(
    objective_function, 
    x0=initial_guess, 
    args=(true_explosion_times, device_positions, simulated_arrival_times, true_explosion_positions),
    method='L-BFGS-B',
    options={'maxiter': 100, 'ftol': 1e-8, 'disp': True}  # 增加最大迭代次数，降低容忍度
)


# Reshape the result to get the predicted explosion positions
predicted_explosion_positions = result.x.reshape(4, 3)

# Calculate final position errors (in meters)
final_position_errors = np.linalg.norm(predicted_explosion_positions - true_explosion_positions, axis=1)

# Output results
print("Predicted explosion positions (in meters):")
print(predicted_explosion_positions)

print("\nPosition errors (in meters):")
print(final_position_errors)

# Check if the model is valid based on the position error constraint
if np.all(final_position_errors <= 1000):
    print("\nThe model is valid, all position errors are within 1000 meters.")
else:
    print("\nThe model is invalid, some position errors exceed 1000 meters.")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体为 SimHei，适用于 xlabel, ylabel, title 等
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# Constants
sound_speed = 340  # m/s
lat_to_meters = 111263  # meters per degree latitude
lon_to_meters = 97304   # meters per degree longitude

# Function to calculate predicted arrival times
def calculate_predicted_arrival_times(device_positions, explosion_positions, explosion_times):
    predicted_arrival_times = []
    for device_position in device_positions:
        times = []
        for i in range(len(explosion_positions)):
            distance = np.linalg.norm(device_position - explosion_positions[i])
            time_delay = distance / sound_speed
            arrival_time = explosion_times[i] + time_delay
            times.append(arrival_time)
        predicted_arrival_times.append(times)
    return np.array(predicted_arrival_times)

# 1. 误差分布直方图
def plot_error_distribution(time_differences):
    fig, ax = plt.subplots()
    ax.hist(time_differences.flatten(), bins=20, color='blue', alpha=0.7)
    ax.set_xlabel('时间差异（秒）')
    ax.set_ylabel('频数')
    ax.set_title('误差分布直方图')
    plt.show()


# 3. 多次模拟结果对比
def plot_multiple_simulations(predicted_positions_list, true_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制真实的爆炸位置
    ax.scatter(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], c='green', label='真实音爆位置', s=100)

    # 绘制多次模拟的预测爆炸位置
    for predicted_positions in predicted_positions_list:
        ax.scatter(predicted_positions[:, 0], predicted_positions[:, 1], predicted_positions[:, 2], alpha=0.6)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("多次模拟音爆位置")
    ax.legend()
    plt.show()

# Example Data (Replace with actual model data)
predicted_explosion_positions = true_explosion_positions + np.random.randn(4, 3) * 50

# Calculate predicted arrival times based on predicted explosion positions
predicted_arrival_times = calculate_predicted_arrival_times(device_positions, predicted_explosion_positions, true_explosion_times)

# Calculate time differences
time_differences = simulated_arrival_times - predicted_arrival_times

# 1. 生成误差分布直方图
plot_error_distribution(time_differences)


# 3. 生成多次模拟结果对比图
predicted_positions_list = [predicted_explosion_positions + np.random.randn(4, 3) * 20 for _ in range(10)]
plot_multiple_simulations(predicted_positions_list, true_explosion_positions)
