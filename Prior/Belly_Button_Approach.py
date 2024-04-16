########################################################################################################################
# Attempt at mimicking 'belly button' allocation approach to our problem
########################################################################################################################
from scipy.stats import dirichlet
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statistics

########################################################################################################################
# Function used to assign allocation to each well
########################################################################################################################

def Histogram_Probability_Density_Plots(Well_Steamflow_Means, confidence = 0.95):
    Allocations = dirichlet(Well_Steamflow_Means).rvs(1000)
    # Extract columns
    columns = Allocations.T
    Means = np.zeros((1, len(Well_Steamflow_Means)))
    for j in range(len(Well_Steamflow_Means)):
        Means[0, j] = statistics.mean(Allocations[:, j])  # Adjusted indexing and storage

    # Plot probability distribution and histogram for each column
    for i, column in enumerate(columns):
        plt.figure(figsize=(10, 5))

        # Probability density plot
        plt.subplot(1, 2, 1)
        sns.kdeplot(column, fill=True)
        plt.title(f'Well {i + 1} - Probability Distribution')
        plt.xlabel('Allocation Made')

        # Histogram plot
        plt.subplot(1, 2, 2)
        sns.histplot(column, kde=False, bins=10)
        plt.title(f'Well {i + 1} - Histogram')
        plt.xlabel('Allocation Made')

        # Add confidence interval for histogram
        lower_bound = np.percentile(column, (1 - confidence) * 100 / 2)
        upper_bound = np.percentile(column, (1 + confidence) * 100 / 2)
        plt.axvline(lower_bound, color='r', linestyle='--', label=f'{int(confidence * 100)}% Confidence Interval')
        plt.axvline(upper_bound, color='r', linestyle='--')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return Means


########################################################################################################################
# Function used to display potential allocations in a aesthetic way
########################################################################################################################

def Colour_Plot(Steam_flows, samples=20):
    s = np.random.default_rng().dirichlet((Steam_flows), samples).transpose()

    plt.barh(range(samples), s[0], label="Well 1")
    plt.barh(range(samples), s[1], left=s[0], color='g', label="Well 2")
    plt.barh(range(samples), s[2], left=s[0] + s[1], color='r', label="Well 3")
    plt.barh(range(samples), s[3], left=s[0] + s[1] + s[2], color='y', label="Well 4")
    plt.barh(range(samples), s[4], left=s[0] + s[1] + s[2] + s[3], color='k', label="Well 5")
    plt.title("Lengths of Strings")

    plt.legend()
    plt.show()

########################################################################################################################
# Generate relevant steam allocations for each well
########################################################################################################################

def Steam_Flow(Well_Allocations, Steam_Flow_Total):

    Steam_flows = Well_Allocations*Steam_Flow_Total

    return Steam_flows

########################################################################################################################
# Estimate mass flow coming from each well
########################################################################################################################

########################################################################################################################
# Script runs from here
########################################################################################################################

# Well_Steamflow_Means = [60,30,22.5]
Well_proportions = [42, 46.7, 11, 3.6, 26.4]  # For the 5 wells, input their relevant contribution for known data.
Mean_Allocations = Histogram_Probability_Density_Plots(Well_proportions)
print(np.sum(Mean_Allocations))

Steam_flows = Steam_Flow(Mean_Allocations,1000)
print(Steam_flows)

# Colour_Plot(Well_proportions, samples=20)