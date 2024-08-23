import numpy as np
import random

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

# Sound arrival times at each device (with added random noise)
def add_random_error(sound_times):
    return sound_times + np.random.normal(0, 0.5, sound_times.shape)

sound_times = np.array([
    [100.767, 164.229, 214.850, 270.065],
    [92.453, 112.220, 169.362, 196.583],
    [75.560, 110.696, 156.936, 188.020],
    [94.653, 141.409, 196.517, 258.985],
    [78.600, 86.216, 118.443, 126.669],
    [67.274, 166.270, 175.482, 266.871],
    [103.738, 163.024, 206.789, 210.306]
])
sound_times_with_error = add_random_error(sound_times)

# Calculate distance between two points using latitude and longitude
def haversine_distance(lon1, lat1, lon2, lat2):
    delta_lon = (lon2 - lon1) * lon_conversion
    delta_lat = (lat2 - lat1) * lat_conversion
    return np.sqrt(delta_lon ** 2 + delta_lat ** 2)

# Calculate the time delay from each device to a potential explosion point using latitude and longitude
def time_delay(device, explosion_point, explosion_time):
    distance = haversine_distance(device[0], device[1], explosion_point[0], explosion_point[1])
    height_diff = abs(device[2] - explosion_point[2])
    total_distance = np.sqrt(distance ** 2 + height_diff ** 2)
    time_diff = total_distance / sound_speed
    return explosion_time + time_diff

# Fitness function: evaluate the total error of the sound arrival times and add penalty for time constraints
def fitness(solution, sound_times):
    explosion_times = solution[:4]
    explosion_positions = solution[4:].reshape((4, 3))
    
    total_error = 0
    penalty = 0

    # Check the conditions for each explosion time
    for i in range(4):
        # Ensure explosion time is greater than zero
        if explosion_times[i] <= 0:
            penalty += 1000  # Large penalty for negative or zero times
        
        # Ensure explosion time is less than the minimum arrival time for each device
        for j in range(7):
            min_arrival_time = np.min(sound_times[j])
            if explosion_times[i] >= min_arrival_time:
                penalty += 1000  # Large penalty for exceeding the minimum arrival time

    # Calculate RMSE for the sound arrival times
    for i in range(7):  # For each device
        for j in range(4):  # For each explosion
            predicted_time = time_delay(device_locations[i], explosion_positions[j], explosion_times[j])
            total_error += (predicted_time - sound_times[i, j]) ** 2  # Squared error
    
    # Add penalty if the time differences between explosions exceed 5 seconds
    for i in range(4):
        for j in range(i + 1, 4):
            if abs(explosion_times[i] - explosion_times[j]) > 5:
                penalty += 1000  # Arbitrary large penalty value for violation

    return np.sqrt(total_error) + penalty  # RMSE with penalty

# Initial population generation
def generate_initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        explosion_times = np.random.uniform(1, 200, 4)  # Random initial times, starting from 1 to ensure positive values
        explosion_positions = np.random.uniform([110.0, 27.0, 500], [111.0, 28.0, 900], (4, 3))  # Random positions in lat/lon/height
        solution = np.concatenate([explosion_times, explosion_positions.flatten()])
        population.append(solution)
    return population

# Evolve population function
def evolve_population(population, elite_size=1, generation_num=1, max_generations=500):
    # Sort population by fitness
    population = sorted(population, key=lambda x: fitness(x, sound_times_with_error))

    # Elite preservation: retain the best solutions
    new_population = population[:elite_size]

    # Adaptive mutation rate
    mutation_rate = max(0.1, 1.0 - (generation_num / max_generations))  # Decaying mutation rate

    # Crossover and mutation
    while len(new_population) < len(population):
        parents = random.sample(population[:len(population) // 2], 2)  # Select from best half
        cross_point = random.randint(1, len(parents[0]) - 1)
        child = np.concatenate([parents[0][:cross_point], parents[1][cross_point:]])

        # Mutation
        if random.random() < mutation_rate:
            mutation_index = random.randint(0, len(child) - 1)
            child[mutation_index] += np.random.uniform(-1, 1)

        new_population.append(child)

    return new_population

# Particle Swarm Optimization
class Particle:
    def __init__(self, dimensions):
        self.position = np.random.uniform(1, 200, dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

def particle_swarm_optimization(best_solution, num_particles=50, max_iterations=1000, w=0.5, c1=1.5, c2=1.5, sound_times_with_error=None):
    dimensions = len(best_solution)
    swarm = [Particle(dimensions) for _ in range(num_particles)]

    global_best_position = best_solution.copy()
    global_best_fitness = fitness(global_best_position, sound_times_with_error)

    for iteration in range(max_iterations):
        for particle in swarm:
            # Update particle's fitness
            particle_fitness = fitness(particle.position, sound_times_with_error)

            # Update personal best
            if particle_fitness < particle.best_fitness:
                particle.best_position = particle.position.copy()
                particle.best_fitness = particle_fitness

            # Update global best
            if particle_fitness < global_best_fitness:
                global_best_position = particle.position.copy()
                global_best_fitness = particle_fitness

        for particle in swarm:
            # Update velocity
            particle.velocity = (
                w * particle.velocity +
                c1 * np.random.rand() * (particle.best_position - particle.position) +
                c2 * np.random.rand() * (global_best_position - particle.position)
            )

            # Update position
            particle.position += particle.velocity

        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Best fitness: {global_best_fitness}")

    return global_best_position

# Genetic algorithm with elite preservation and PSO
def genetic_algorithm_with_pso(pop_size=300, generations=1000, elite_size=5, sound_times_with_error=None):
    population = generate_initial_population(pop_size)

    for generation in range(generations):
        population = evolve_population(population, elite_size, generation, generations)

        # Track the best solution found
        best_solution = min(population, key=lambda x: fitness(x, sound_times_with_error))
        best_fitness = fitness(best_solution, sound_times_with_error)
        
        print(f"Generation {generation}: Best fitness = {best_fitness}")

        # If fitness is good enough, stop early
        if best_fitness < 100:
            break
    
    # After genetic algorithm finishes, perform PSO for local search
    best_solution = particle_swarm_optimization(best_solution, sound_times_with_error=sound_times_with_error)
    
    return best_solution
    
# Test the model with new synthetic data
def simulate_new_data():
    new_device_locations = np.array([
        [110.500, 27.600, 750],
        [110.600, 27.700, 760],
        [110.700, 27.800, 770],
        [110.800, 27.900, 780],
        [110.900, 28.000, 790],
        [111.000, 28.100, 800],
        [111.100, 28.200, 810]
    ])
    
    # Generate synthetic sound times based on new locations
    new_sound_times = np.random.uniform(50, 250, (7, 4)) + add_random_error(np.zeros((7, 4)))
    
    return new_device_locations, new_sound_times

# Run the optimized genetic algorithm with PSO and error correction
new_device_locations, new_sound_times = simulate_new_data()

best_solution = genetic_algorithm_with_pso(sound_times_with_error=new_sound_times)

# Extract the results
explosion_times = best_solution[:4]
explosion_positions = best_solution[4:].reshape((4, 3))

# Calculate final error
total_error = fitness(best_solution, new_sound_times)

# Output the final explosion times and positions with error
print("Best explosion times:", explosion_times)
print("Best explosion positions (longitude, latitude, height):", explosion_positions)
print("Total error (RMSE):", total_error)