import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# Adjust number of wells
wells = 5

# ==============================================================================
# Example of generating allocation so it's between 0 and 1
# ==============================================================================

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

print("Original range:", min_val, "-", max_val)
print("Scaled range:", scaled_samples.min(), "-", scaled_samples.max())
print(scaled_samples[1:10])

# Plot the scaled samples as a histogram
plt.hist(scaled_samples, bins=30, density=True, alpha=0.6, color='g')

plt.xlabel('Allocation Sample')
plt.ylabel('Probability Density')
plt.title('Histogram showing probability density of allocation')
plt.grid(True)
plt.show()

# ==============================================================================
# Generating allocations for multiple wells
# ==============================================================================

# Function to randomly allocate values until the sum reaches 1
def random_allocation(scaled_samples, num_allocations):
    allocations = []
    remaining = 1.0
    for _ in range(num_allocations - 1):
        allocation = np.random.choice(scaled_samples)
        allocation = min(allocation, remaining)
        allocations.append(allocation)
        remaining -= allocation
    allocations.append(remaining)
    return allocations

# Perform 5 random allocations
allocations = random_allocation(scaled_samples, wells)

# Print the allocations
print("Allocations:", allocations)
print("Sum of allocations:", sum(allocations))

# ==============================================================================
# Determine total steam flow for each well
# ==============================================================================

# Set an arbitrary steam value
Steam_Total = 1000
Steam_wells = np.zeros(wells)

for i in range(wells):
    Steam_wells[i] = allocations[i]*Steam_Total

print(Steam_wells)

# ==============================================================================
# Determine mass flow for each well
# ==============================================================================

Mass = np.zeros(wells)

for i in range(wells):
    Mass[i] = Steam_wells[i]/np.random.uniform(0,1,1)

print(Mass)