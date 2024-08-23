import numpy as np
import random
from multiprocessing import Pool
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体为 SimHei，适用于 xlabel, ylabel, title 等
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# Constants
sound_speed = 340  # m/s
lat_conversion = 111263  # meters per degree of latitude
lon_conversion = 97304   # meters per degree of longitude

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

# Sound arrival times at each device
sound_times = np.array([
    [100.767, 164.229, 214.850, 270.065],
    [92.453, 112.220, 169.362, 196.583],
    [75.560, 110.696, 156.936, 188.020],
    [94.653, 141.409, 196.517, 258.985],
    [78.600, 86.216, 118.443, 126.669],
    [67.274, 166.270, 175.482, 266.871],
    [103.738, 163.024, 206.789, 210.306]
])

# Convert latitude and longitude to Cartesian coordinates for distance calculation
def lat_lon_to_cartesian(lon, lat, h):
    x = lon * lon_conversion
    y = lat * lat_conversion
    return np.array([x, y, h])

# Calculate the time delay from each device to a potential explosion point using absolute coordinates
def time_delay(device, explosion_point, explosion_time):
    device_cartesian = lat_lon_to_cartesian(*device)
    explosion_cartesian = lat_lon_to_cartesian(*explosion_point)
    distance = np.linalg.norm(device_cartesian - explosion_cartesian)
    time_diff = distance / sound_speed
    return explosion_time + time_diff

# Fitness function: evaluate the total error of the sound arrival times and add penalty for time constraints
def fitness(solution, selected_devices, selected_times):
    explosion_times = solution[:4]
    explosion_positions = solution[4:].reshape((4, 3))
    
    total_error = 0

    # Calculate RMSE for the sound arrival times
    for i in range(len(selected_devices)):  # For each selected device
        for j in range(4):  # For each explosion
            predicted_time = time_delay(selected_devices[i], explosion_positions[j], explosion_times[j])
            total_error += (predicted_time - selected_times[i, j]) ** 2  # Squared error

    return np.sqrt(total_error)  # RMSE

# Parallel fitness evaluation for the population
def parallel_fitness(population, selected_devices, selected_times):
    with Pool() as pool:
        results = pool.starmap(fitness, [(ind, selected_devices, selected_times) for ind in population])
    return results

# Initial population generation
def generate_initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        explosion_times = np.random.uniform(1, 200, 4)  # Random initial times, starting from 1 to ensure positive values
        explosion_positions = np.random.uniform([110.0, 27.0, 500], [111.0, 28.0, 900], (4, 3))  # Random positions in lat/lon/altitude
        solution = np.concatenate([explosion_times, explosion_positions.flatten()])
        population.append(solution)
    return population

# Selection, crossover, and mutation operations with elite preservation
def evolve_population(population, selected_devices, selected_times, elite_size=1, generation_num=1, max_generations=500):
    # Sort population by fitness using parallel computation
    fitnesses = parallel_fitness(population, selected_devices, selected_times)
    population = [x for _, x in sorted(zip(fitnesses, population))]
    
    # Elite preservation: retain the best solutions
    new_population = population[:elite_size]
    
    # Adaptive mutation rate
    mutation_rate = max(0.1, 1.0 - (generation_num / max_generations))  # Decaying mutation rate
    
    # Crossover and mutation
    while len(new_population) < len(population):
        parents = random.sample(population[:len(population)//2], 2)  # Select from best half
        cross_point = random.randint(1, len(parents[0]) - 1)
        child = np.concatenate([parents[0][:cross_point], parents[1][cross_point:]])
        
        # Mutation
        if random.random() < mutation_rate:
            mutation_index = random.randint(0, len(child) - 1)
            child[mutation_index] += np.random.uniform(-1, 1)
        
        new_population.append(child)
    
    return new_population

# Simulation to find the minimum number of devices needed
def simulate_device_selection(num_trials=20):
    num_devices = len(device_locations)
    
    for num_selected_devices in range(3, num_devices + 1):
        results = []
        
        for _ in range(num_trials):
            # Randomly select num_selected_devices devices
            device_indices = np.random.choice(range(num_devices), num_selected_devices, replace=False)
            selected_devices = device_locations[device_indices]
            selected_times = sound_times[device_indices]
            
            # Run genetic algorithm with PSO
            population = generate_initial_population(50)
            best_solution = min(population, key=lambda x: fitness(x, selected_devices, selected_times))
            
            # Extract results
            explosion_times = best_solution[:4]
            explosion_positions = best_solution[4:].reshape((4, 3))
            result = np.concatenate([explosion_positions.flatten(), explosion_times])
            results.append(result)
        
        # Convert results to numpy array
        results = np.array(results)
        
        # Calculate mean and standard deviation
        mean_result = np.mean(results, axis=0)
        std_result = np.std(results, axis=0)
        
        # Output the results in the required format
        print(f"设备数量: {num_selected_devices}, 平均结果: {mean_result}, 标准差: {std_result}")

# Run the simulation to determine the required number of devices
simulate_device_selection()
