import pandas as pd
import numpy as np

# Number of data points to generate
num_samples = 5000
num_exceptions = 1000  # Number of exception cases to introduce
num_violations = 500   # Number of rule-violating cases to introduce

# Define ranges for each feature
remaining_battery_range = (0, 100)  # percentage
drain_rate_range = (2, 10)  # percentage per unit distance or time
remaining_range_range = (0, 400)  # km
estimated_time_left_range = (10, 600)  # minutes
time_to_station_range = (1, 60)  # minutes
distance_to_station_range = (0.1, 30)  # km

# Generate random values within the defined ranges
np.random.seed(42)  # for reproducibility
remaining_battery = np.random.uniform(*remaining_battery_range, num_samples)
drain_rate = np.random.uniform(*drain_rate_range, num_samples)
remaining_range = np.random.uniform(*remaining_range_range, num_samples)
estimated_time_left = np.random.uniform(*estimated_time_left_range, num_samples)
time_to_station = np.random.uniform(*time_to_station_range, num_samples)
distance_to_station = np.random.uniform(*distance_to_station_range, num_samples)

# Add noise to each feature
noise_factor = 0.05
remaining_battery += np.random.normal(0, noise_factor * (remaining_battery_range[1] - remaining_battery_range[0]), num_samples)
drain_rate += np.random.normal(0, noise_factor * (drain_rate_range[1] - drain_rate_range[0]), num_samples)
remaining_range += np.random.normal(0, noise_factor * (remaining_range_range[1] - remaining_range_range[0]), num_samples)
estimated_time_left += np.random.normal(0, noise_factor * (estimated_time_left_range[1] - estimated_time_left_range[0]), num_samples)
time_to_station += np.random.normal(0, noise_factor * (time_to_station_range[1] - time_to_station_range[0]), num_samples)
distance_to_station += np.random.normal(0, noise_factor * (distance_to_station_range[1] - distance_to_station_range[0]), num_samples)

# Ensure values stay within their respective ranges
remaining_battery = np.clip(remaining_battery, *remaining_battery_range)
drain_rate = np.clip(drain_rate, *drain_rate_range)
remaining_range = np.clip(remaining_range, *remaining_range_range)
estimated_time_left = np.clip(estimated_time_left, *estimated_time_left_range)
time_to_station = np.clip(time_to_station, *time_to_station_range)
distance_to_station = np.clip(distance_to_station, *distance_to_station_range)

# Select random indices for exception cases and rule-violating cases
exception_indices = np.random.choice(num_samples, num_exceptions, replace=False)
violation_indices = np.random.choice(num_samples, num_violations, replace=False)

# Introduce exception cases
remaining_battery[exception_indices[:num_exceptions//5]] = np.random.uniform(90, 100, num_exceptions//5)  # Extremely high battery
drain_rate[exception_indices[num_exceptions//5:2*num_exceptions//5]] = np.random.uniform(8, 10, num_exceptions//5)  # High drain rate
remaining_range[exception_indices[2*num_exceptions//5:3*num_exceptions//5]] = np.random.uniform(350, 400, num_exceptions//5)  # High remaining range
estimated_time_left[exception_indices[3*num_exceptions//5:4*num_exceptions//5]] = np.random.uniform(500, 600, num_exceptions//5)  # Long estimated time left
time_to_station[exception_indices[4*num_exceptions//5:]] = np.random.uniform(55, 60, num_exceptions//5)  # Long time to station
distance_to_station[exception_indices[4*num_exceptions//5:]] = np.random.uniform(25, 30, num_exceptions//5)  # Long distance to station

# Introduce rule-violating cases
remaining_battery[violation_indices[:num_violations//5]] = np.random.uniform(0, 10, num_violations//5)  # Low battery but low priority
drain_rate[violation_indices[num_violations//5:2*num_violations//5]] = np.random.uniform(2, 4, num_violations//5)  # Low drain rate but high priority
remaining_range[violation_indices[2*num_violations//5:3*num_violations//5]] = np.random.uniform(0, 50, num_violations//5)  # Low remaining range but low priority
estimated_time_left[violation_indices[3*num_violations//5:4*num_violations//5]] = np.random.uniform(10, 50, num_violations//5)  # Short estimated time but low priority
time_to_station[violation_indices[4*num_violations//5:]] = np.random.uniform(1, 5, num_violations//5)  # Short time to station but low priority
distance_to_station[violation_indices[4*num_violations//5:]] = np.random.uniform(0.1, 2, num_violations//5)  # Short distance to station but low priority

# Calculate priority based on the logical rules provided
# Using arbitrary weights to combine the effects into a single priority score
priority = (
    -0.3 * remaining_battery +  # more remaining battery -> less priority
    0.4 * drain_rate +          # more drain rate -> more priority
    -0.3 * remaining_range +    # more remaining range -> less priority
    -0.2 * estimated_time_left +# more estimated time left -> less priority
    0.2 * (1 / (time_to_station + 1)) +  # less time to station -> more priority
    0.2 * (1 / (distance_to_station + 1))  # less distance to station -> more priority
)

# Normalize priority to be between 0 and 1
priority = (priority - priority.min()) / (priority.max() - priority.min())

# Create DataFrame
data = pd.DataFrame({
    'car_id': np.arange(1, num_samples + 1),
    'remaining_battery': remaining_battery,
    'drain_rate': drain_rate,
    'remaining_range': remaining_range,
    'estimated_time_left': estimated_time_left,
    'time_to_station': time_to_station,
    'distance_to_station': distance_to_station,
    'priority': priority
})

# Display first few rows of the dataset
print(data.head())

# Save to CSV using a raw string for the file path
file_path = r'C:\Users\Diya\projects\ev_charging_data_with_exceptions.csv'
data.to_csv(file_path, index=False)

print(f'Dataset saved to {file_path}')
