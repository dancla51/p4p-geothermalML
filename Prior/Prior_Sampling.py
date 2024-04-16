# imports
import numpy as np
import scipy.stats as s
import matplotlib.pyplot as plt

def generateInitialAllocations(n, nx, data_points):
    # Take data from data points to get initial guess of allocations
    well_steamflow_sums = np.zeros(nx)
    well_amnt_tfts = np.zeros(nx)
    for dp in data_points:
        well = dp['Well']
        well_steamflow_sums[well] += dp['SteamFlow']
        well_amnt_tfts[well] += 1
    well_steamflow_means = np.divide(well_steamflow_sums, well_amnt_tfts)
    well_initial_allocations = well_steamflow_means / sum(well_steamflow_means)

    return well_initial_allocations


# Expected Value of Beta dist = a/(a+b) , mode = (a-1)/(a+b-2) , if increase a and b, then decrease variance

# Generate distributions based on initial allocations
def generateBetaDists(nx, initial_allocs, surety):
    # Higher surety = lower variance (hyperparameter)
    dists = []
    # For each well, set up a beta distribution
    for i in range(nx):
        dists.append(s.beta(initial_allocs[i] * surety, surety - initial_allocs[i] * surety))

    return dists

def generateAllocations(n, nx, data_points, surety, Generated_CI):
    # Find initial allocations
    initial_allocs = generateInitialAllocations(n, nx, data_points)
    # Generate Distributions
    dists = generateBetaDists(nx, initial_allocs, surety)

    # Generate actual allocations
    x = np.zeros([n, nx])
    for i in range(n):
        # For certain time, get alloc for all wells based on distributions, then scale
        tmp = np.zeros(nx)
        for j in range(nx):
            lower_CI, upper_CI = Generated_CI[j][0], Generated_CI[j][1]
            while tmp[j]<lower_CI or tmp[j]>upper_CI or tmp[j]==0:
                tmp[j] = dists[j].rvs(1)
        for j in range(nx):
            x[i, j] = tmp[j] / sum(tmp)

    # Artificially set allocations to 'optimal'
    #     for i in range(n):
    #         x[i,:] = initial_allocs

    return x, dists

# Use 3rd (beta) distribution for drynesses (all wells have same dist atm)

def generateDryness(n, data_points):
    # Treat dryness as constant for now
    d = np.zeros([n, nx])

    # Take data from data points to get mean dryness
    well_dryness_sums = np.zeros(nx)
    dryness_amnt_tfts = np.zeros(nx)
    for dp in data_points:
        well = dp['Well']
        well_dryness_sums[well] += dp['Dryness']
        dryness_amnt_tfts[well] += 1
    well_dryness_means = np.divide(well_dryness_sums, dryness_amnt_tfts)

    # Assign to dryness matrix
    for i in range(n):
        d[i, :] = well_dryness_means

    return d

def generateSteamFlow(x, S_total):
    # x is n rows * xn columns
    # S_total is n * 1
    S = np.copy(x)
    for i in range(len(x)):
        S[i] = x[i] * S_total[i]

    return S


def BayesianPriorBeta(n, nx, S_total, data_points, surety):
    """
    This function takes in the ouput dimensions, known data, and hyperparameter surety, and returns the allocations,
    steam flows, mass flows, and the distributions used to generate these allocations. Dryness is CONSTANT, taken from data
    inputted. Allocations are generated using a beta distribution, whose mean value is the allocation found from inputted
    data. (Averaging occurs if there is more than one data point per well.)
    Parameters:
        n = number of months to run for
        nx = number of wells
        S_total = total steam flow over n months
        data_points = list of dictionaries representing TFT data points.
        surety = hyperparameter of function. Increasing this decreases the variance of our beta distributions.
    """
    # generate results with randomness
    x, dists = generateAllocations(n, nx, data_points, surety)
    d = generateDryness(n, data_points)
    S = generateSteamFlow(x, S_total)
    M = np.divide(S, d)

    return x, S, M, dists


def Beta_Dist_CI_Generation_Plots(n, nx, data_points, surety, Confidence_Interval=0.95,show=True):
    # Find initial allocations
    initial_allocs = generateInitialAllocations(n, nx, data_points)
    # Generate Distributions
    dists = generateBetaDists(nx, initial_allocs, surety)

    confidence_interval = []
    # Create subplots for each beta distribution
    fig, axes = plt.subplots(nrows=nx, ncols=1, figsize=(8, 6))

    # Iterate over each beta distribution
    for i, dist in enumerate(dists):
        ax = axes[i]  # Get the current axis

        # Generate 1000 random samples
        random_samples = dist.rvs(size=1000)

        # Calculate confidence interval
        ci = dist.interval(Confidence_Interval)
        lower_ci, upper_ci = ci
        confidence_interval.append(ci)

        # Plot histogram
        ax.hist(random_samples, bins=30, density=True, alpha=0.5, color='b', label=f'Well {i + 1} Beta Distribution')

        # Add vertical lines for confidence interval
        ax.axvline(lower_ci, color='r', linestyle='--')
        ax.axvline(upper_ci, color='r', linestyle='--')

        # Add labels for confidence interval
        ax.text(0.05, 0.95, f'Lower CI: {lower_ci:.2f}\nUpper CI: {upper_ci:.2f}', transform=ax.transAxes, color='r',
                fontsize=10, va='top')

        # Add labels and title
        ax.set_xlabel('Allocation')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Well {i + 1} Beta Distribution with Confidence Interval')
        ax.legend()

    # Adjust layout and show plot
    plt.tight_layout()
    if show==True:
        plt.show()


    return confidence_interval

# Declare input data
iterations = 40
# n = number of months
n = 10
# nx = number of wells
nx = 3
# Total Steam Flow
S_total = np.array([[100], [100], [100], [110], [120], [130], [150], [180], [200], [100]])
# Each data point can be defined by dictionary of mass flow, dryness, steam flow (and enthalpy excluded for now)
data_points = [
    {'Well': 0, 'Month': 2, 'Dryness': 0.3, 'MassFlow': 200., 'SteamFlow': 60.},
    {'Well': 1, 'Month': 7, 'Dryness': 0.1, 'MassFlow': 300., 'SteamFlow': 30.},
    {'Well': 2, 'Month': 4, 'Dryness': 0.25, 'MassFlow': 100., 'SteamFlow': 25.},
    {'Well': 2, 'Month': 5, 'Dryness': 0.20, 'MassFlow': 100., 'SteamFlow': 20.}
]
# declare hyperparameter surety (decreases variance)
surety = 400

########################################################################################################################
# All functions and such are ran below
########################################################################################################################

# CI = Beta_Dist_CI_Generation_Plots(n, nx, data_points, surety,Confidence_Interval=0.95,show=False)
# print(CI)
# print(CI[0][0])
#
# x,dist = generateAllocations(n,nx,data_points,surety,CI)

allocations = generateInitialAllocations(n, nx, data_points)
