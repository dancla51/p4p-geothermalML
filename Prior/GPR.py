########################################################################################################################
# Attempt at applying Gaussian Process Regression (GPR) to our problem
########################################################################################################################
from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Exponentiation
from sklearn.kernel_ridge import KernelRidge

########################################################################################################################
# Create 1-D GPR model
########################################################################################################################

def quadratic_kernel(X1, X2):
    return (np.dot(X1, X2.T) + 1) ** 2

def GPR_Model(X,y,training_indices):
    """
    Fits a Gaussian Process Regressor on data and displays the results with varying levels of noise
    :param X: 1D vector containing the x values of observed data (e.g. months)
    :param y: 1D vector containing the y values of observed data (e.g. steam flow)
    :param Random_Samples: The number of observations to train on.
    :param Noise_Level: Determines the width of confidence intervals produced in final plot

    Note - When creating training data does this without replacement so Random_Samples must not exceed the length of X or y
    """
    # Data plotted for existing x and y values
    plt.scatter(X, y, color='blue', label='Data Points', zorder=3)
    plt.legend()
    plt.plot(X, y)
    plt.xlabel("Month")
    plt.ylabel("Steam flow (kg/s)")
    plt.title("Plot of actual data")
    plt.show()

    # GPR applied to specified well steam flow data
    X_train, y_train = X[training_indices], y[training_indices]

    # Use the remaining data for testing
    test_indices = np.delete(np.arange(len(y)), training_indices)
    X_test, y_test = X[test_indices], y[test_indices]

    kernel = Exponentiation(RationalQuadratic(), exponent=2)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-1)
    gaussian_process.fit(X_train,y_train)

    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

    plt.plot(X, y, label="Actual data", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Training Observations")
    plt.scatter(X_test, y_test, label="Test Observations", color='red', marker='x')
    plt.plot(X, mean_prediction, label="Mean prediction")

    plt.fill_between(
        X.ravel(),
        mean_prediction - 2 * std_prediction,
        mean_prediction + 2 * std_prediction,
        alpha=0.5,
        label=r"95% Training confidence interval",
    )

    plt.legend()
    plt.xlabel("Month")
    plt.ylabel("Steam flow (kg/s)")
    plt.title("Gaussian process regression on noise-free dataset")
    plt.show()

########################################################################################################################
# Main script begins here
########################################################################################################################
# This is for Well 9 from the example data

X = np.linspace(start=1, stop=12, num=12).reshape(-1, 1)
y = np.array([19.42,19.94,19.92,19.96,20.03,20.13,20.25,20.42,20.63,20.9,21.25,21.76])
training_indices = [6, 8, 9]

GPR_Model(X,y,training_indices)

# Months = np.arange(1,24)
# values = [
#     23.95, 48.62, 51.55, 53.64, 53.48, 55.34, 55.89, 56.12, 56.48, 56.71,
#     60.9, 74.77, 75.56, 76.2, 77.15, 78.39, 80.02, 82.11, 84.82, 88.29,
#     92.6, 98.04, 102.29
# ]
# # Create a NumPy array
# steam_total = np.array(values)
# # Data for well1_steam
# values = [
#     10.9, 22.12, 21.73, 21.43, 21.19, 21.0, 20.92, 20.91, 20.94, 20.92,
#     20.53, 20.18, 20.19, 20.34, 20.59, 20.96, 21.5, 22.28, 23.4, 25.02,
#     27.15, 29.96, 30.84
# ]
#
# # Create a NumPy array
# well1_steam = np.array(values)
#
# values = [
#     13.04, 26.5, 26.17, 25.96, 25.87, 25.86, 26.03, 26.32, 26.69, 26.99,
#     26.8, 26.64, 27.0, 27.57, 28.26, 29.1, 30.11, 31.31, 32.75, 34.4, 36.32,
#     38.59, 41.45
# ]
#
# # Create a NumPy array
# well2_steam = np.array(values)
#
# values = [
#     0, 0, 3.65, 6.25, 6.21, 6.09, 6.03, 6, 5.97, 5.94, 5.87, 5.76, 5.69, 5.65,
#     5.63, 5.61, 5.59, 5.58, 5.58, 5.58, 5.58, 5.58, 5.59
# ]
#
# # Create a NumPy array
# well3_steam = np.array(values)
#
# values = [
#     0, 0, 0, 0, 0.21, 2.39, 2.91, 2.89, 2.88, 2.86, 2.82, 2.77, 2.74, 2.72,
#     2.71, 2.7, 2.69, 2.68, 2.67, 2.66, 2.66, 2.65, 2.65
# ]
#
# # Create a NumPy array
# well4_steam = np.array(values)
#
# # Data for well5_steam
# values = [
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.89, 19.42, 19.94, 19.92, 19.96, 20.03,
#     20.13, 20.25, 20.42, 20.63, 20.9, 21.25, 21.76
# ]
#
# # Create a NumPy array
# well5_steam = np.array(values)
#
# Actually_known_months = np.array([5,8,10,12,15])
# Actuall_known_steams = np.array([21.19,6,26.99,19.42,2.71])
#
# # Plotting the data
# plt.plot(Months, steam_total, label="Total Steam Flow")
# plt.plot(Months, well1_steam, label='Well 1 Steam Flow')
# plt.plot(Months, well2_steam, label='Well 2 Steam Flow')
# plt.plot(Months, well3_steam, label='Well 3 Steam Flow')
# plt.plot(Months, well4_steam, label='Well 4 Steam Flow')
# plt.plot(Months, well5_steam, label='Well 5 Steam Flow')
# plt.scatter(Actually_known_months,Actuall_known_steams, label="Collected Steam Flow")
#
# # Shade in the region between the first and second non-zero steam values vertically
# plt.axvspan(np.where(well1_steam != 0)[0][0] + 1, np.where(well1_steam != 0)[0][1] + 1, color='gray', alpha=0.3)
# plt.axvspan(np.where(well2_steam != 0)[0][0] + 1, np.where(well2_steam != 0)[0][1] + 1, color='gray', alpha=0.3)
# plt.axvspan(np.where(well3_steam != 0)[0][0] + 1, np.where(well3_steam != 0)[0][1] + 1, color='gray', alpha=0.3)
# plt.axvspan(np.where(well4_steam != 0)[0][0] + 1, np.where(well4_steam != 0)[0][1] + 1, color='gray', alpha=0.3)
# plt.axvspan(np.where(well5_steam != 0)[0][0] + 1, np.where(well5_steam != 0)[0][1] + 1, color='gray', alpha=0.3)
#
# # Adding labels and title
# plt.xlabel('Month')
# plt.ylabel('Steam flow (kg/s)')
# plt.title('Steam flow against month for total and individual wells')
# plt.legend()
#
# # Show plot
# plt.show()
