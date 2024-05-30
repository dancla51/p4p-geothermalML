########################################################################################################################
# Attempt at applying Gaussian Process Regression (GPR) to our problem
########################################################################################################################
from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Exponentiation
from scipy.interpolate import interp1d
from numpy import arange, array, exp

########################################################################################################################
# Create 1-D GPR model
########################################################################################################################

def quadratic_kernel(X1, X2):
    return (np.dot(X1, X2.T) + 1) ** 2

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(list(map(pointwise, array(xs))))

    return ufunclike

def GPR_Model(X,y,training_indices):
    """
    Fits a Gaussian Process Regressor on data and displays the results with varying levels of noise
    :param X: 1D vector containing the x values of observed data (e.g. months)
    :param y: 1D vector containing the y values of observed data (e.g. steam flow)
    :param Random_Samples: The number of observations to train on.
    :param Noise_Level: Determines the width of confidence intervals produced in final plot

    Note - When creating training data does this without replacement so Random_Samples must not exceed the length of X or y
    """
    # # Data plotted for existing x and y values
    # plt.scatter(X, y, color='blue', label='Data Points', zorder=3)
    # plt.legend()
    # plt.plot(X, y)
    # plt.xlabel("Month")
    # plt.ylabel("Steam flow (kg/s)")
    # plt.title("Plot of actual data")
    # plt.show()

    # GPR applied to specified well steam flow data
    X_train, y_train = X[training_indices], y[training_indices]

    # Use the remaining data for testing
    test_indices = np.delete(np.arange(len(y)), training_indices)
    X_test, y_test = X[test_indices], y[test_indices]

    kernel = Exponentiation(RationalQuadratic(), exponent=2)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-10)
    gaussian_process.fit(X_train,y_train)

    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

    f_i = interp1d(X_train.flatten(), y_train.flatten())
    f_x = extrap1d(f_i)
    extrapolated_points = f_x([11, 12])
    x_values_extra = np.array([11, 12])

    plt.plot(X, y, label="Actual data", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Training Observations")
    plt.scatter(x_values_extra, extrapolated_points, label="Extrapolated Values")
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
# reversed_arr = y[::-1]
training_indices = [4,6,8]

GPR_Model(X,y,training_indices)
