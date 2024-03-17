import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

Well_count = 5 # Number of wells
Total_steam_flow = 1000 # Total steam flow at particular time
Dryness = [0.182, 0.204, 0.101, 0.059, 0.137] # Dryness from excel spreadsheet

# Parameters of the Weibull distribution
shape = 2  # Shape parameter
scale = 1  # Scale parameter

# Generate random samples from the Weibull distribution
# Note: weibull_min.rvs() generates random variates.
# You may also use weibull_min.pdf() for probability density function.
samples = weibull_min.rvs(shape, scale=scale, size=1000)

# Scale and shift the samples to fit between 0 and 1
min_val = samples.min()
max_val = samples.max()
scaled_samples = (samples - min_val) / (max_val - min_val)

def random_allocation(scaled_samples, num_allocations):
    allocations = []
    remaining = 1.0
    for _ in range(num_allocations - 1):
        # Ensure that allocation is greater than 0 and less than or equal to remaining
        allocation = np.random.uniform(0, remaining)
        allocation = min(allocation, remaining)
        allocations.append(allocation)
        remaining -= allocation
    allocations.append(remaining)
    return allocations

def mass_calc(Wells, allocations, Steam_Total, Dryness):
    Steam_wells = np.zeros(Wells)
    Mass = np.zeros(Wells)
    for i in range(Wells):
        Steam_wells[i] = allocations[i] * Steam_Total

    for i in range(Wells):
        Mass[i] = Steam_wells[i] / Dryness[i]

    return Steam_wells, Mass

# ==============================================================================
# Running the main script
# ==============================================================================

allocations = random_allocation(scaled_samples, Well_count)
Steam_wells, Mass = mass_calc(Well_count, allocations, Total_steam_flow, Dryness)

print(allocations)
print(Steam_wells)
print(Mass)

Well_1 = np.zeros(1000)
Well_2 = np.zeros(1000)
Well_3 = np.zeros(1000)
Well_4 = np.zeros(1000)
Well_5 = np.zeros(1000)

for i in range(1000):
    allocations = random_allocation(scaled_samples, Well_count)
    Steam_wells, Mass = mass_calc(Well_count, allocations, Total_steam_flow, Dryness)
    Well_1[i] = Mass[0]
    Well_2[i] = Mass[1]
    Well_3[i] = Mass[2]
    Well_4[i] = Mass[3]
    Well_5[i] = Mass[4]

CI = np.percentile(Well_1, [2.5,97.5])

plt.hist(Well_1, bins=20, density=True, alpha=0.6, color='b', label='Data')
plt.axvline(CI[0], color='r', linestyle='--', label='95% CI')
plt.axvline(CI[1], color='r', linestyle='--')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Well 1 Histogram with 95% Confidence Interval')
plt.legend()
plt.show()

CI = np.percentile(Well_2, [2.5,97.5])

plt.hist(Well_2, bins=20, density=True, alpha=0.6, color='b', label='Data')
plt.axvline(CI[0], color='r', linestyle='--', label='95% CI')
plt.axvline(CI[1], color='r', linestyle='--')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Well 2 Histogram with 95% Confidence Interval')
plt.legend()
plt.show()

CI = np.percentile(Well_3, [2.5,97.5])

plt.hist(Well_3, bins=20, density=True, alpha=0.6, color='b', label='Data')
plt.axvline(CI[0], color='r', linestyle='--', label='95% CI')
plt.axvline(CI[1], color='r', linestyle='--')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Well 3 Histogram with 95% Confidence Interval')
plt.legend()
plt.show()

CI = np.percentile(Well_4, [2.5,97.5])

plt.hist(Well_4, bins=20, density=True, alpha=0.6, color='b', label='Data')
plt.axvline(CI[0], color='r', linestyle='--', label='95% CI')
plt.axvline(CI[1], color='r', linestyle='--')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Well 4 Histogram with 95% Confidence Interval')
plt.legend()
plt.show()

CI = np.percentile(Well_5, [2.5,97.5])

plt.hist(Well_5, bins=20, density=True, alpha=0.6, color='b', label='Data')
plt.axvline(CI[0], color='r', linestyle='--', label='95% CI')
plt.axvline(CI[1], color='r', linestyle='--')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Well 5 Histogram with 95% Confidence Interval')
plt.legend()
plt.show()