import numpy as np

# Constants
sound_speed = 340  # m/s (sound speed in air)
lat_conversion = 111263  # meters per degree of latitude
lon_conversion = 97304   # meters per degree of longitude

# Best explosion positions (longitude, latitude, height in meters)
explosion_positions = np.array([
    [110.24881132, 27.71875567, 830.17912749],
    [110.34201478, 27.76222241, 782.72874308],
    [110.41989654, 27.70311125, 495.09025456],
    [110.51613294, 27.9112601, 797.64233002]
])

# Best explosion times in seconds
explosion_times = np.array([11.97872243, 16.97330964, 16.38970079, 16.7949202])

# Convert explosion positions (longitude, latitude) to meters (local approximation)
explosion_positions_m = explosion_positions.copy()
explosion_positions_m[:, 0] *= lon_conversion  # Convert longitude to meters
explosion_positions_m[:, 1] *= lat_conversion  # Convert latitude to meters

# Simulate 7 devices with random positions (longitude, latitude, height in meters)
np.random.seed(42)  # For reproducibility
device_longitudes = np.random.uniform(110.2, 110.6, 7)  # Random longitudes (degrees)
device_latitudes = np.random.uniform(27.65, 27.95, 7)   # Random latitudes (degrees)
device_heights = np.random.uniform(400, 850, 7)         # Random heights (meters)

# Create device positions (longitude, latitude, height)
device_positions = np.vstack([device_longitudes, device_latitudes, device_heights]).T

# Convert device positions (longitude, latitude) to meters
device_positions_m = device_positions.copy()
device_positions_m[:, 0] *= lon_conversion  # Convert longitude to meters
device_positions_m[:, 1] *= lat_conversion  # Convert latitude to meters

# Calculate the predicted sound arrival times for each device and explosion
def calculate_arrival_times(device_positions_m, explosion_positions_m, explosion_times):
    arrival_times = []
    for device_position in device_positions_m:
        times = []
        for i in range(len(explosion_positions_m)):
            # Calculate the distance between the device and the explosion in meters
            distance = np.linalg.norm(device_position - explosion_positions_m[i])
            # Calculate the time delay (distance / speed of sound)
            time_delay = distance / sound_speed
            # Calculate the sound arrival time at the device
            arrival_time = explosion_times[i] + time_delay
            times.append(arrival_time)
        arrival_times.append(times)
    return np.array(arrival_times)

# Calculate the sound arrival times at each device
arrival_times = calculate_arrival_times(device_positions_m, explosion_positions_m, explosion_times)

# Output the simulated device positions and arrival times
print("Simulated device positions (longitude, latitude, height in degrees and meters):")
print(device_positions)

print("\nSimulated sound arrival times (t1, t2, t3, t4 in seconds):")
print(arrival_times)
