import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import random as r

def GPR_Model(X, y_target_well, TI_target_well, y_other_well, TI_other_well, total_steam, LS_bounds=(1e-1, 10), nu=1.5, alpha=1e-10):
    """
    Fits a Gaussian Process Regressor on data and returns the mean and standard deviation of predictions
    :param X: 1D vector containing the x values of observed data (e.g. months)
    :param y_target_well: 1D vector containing the y values of observed data for the target well (e.g. steam flow)
    :param TI_target_well: The training indices of observations to train on for the target well.
    :param y_other_well: 1D vector containing the y values of observed data for other wells
    :param TI_other_well: The training indices of observations to train on for other wells.
    :param LS_bounds: Length scale bounds for the Matern kernel
    :param nu: Parameter for the Matern kernel
    :param alpha: Regularization parameter for GaussianProcessRegressor
    :return: mean_prediction, std_prediction
    """
    X_train, y_train = X[TI_target_well], y_target_well[TI_target_well]

    kernel = 1 * Matern(length_scale_bounds=LS_bounds, nu=nu)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=alpha)

    y_additional = []
    X_additional = []
    if isinstance(TI_other_well, np.ndarray) and TI_other_well.size != 0:
        for i in range(len(y_other_well)):
            if i in TI_other_well:
                y_additional.append(total_steam[i] - y_other_well[i])
                X_additional.append(i + 1)

        # Convert additional data lists to numpy arrays
        y_additional = np.array(y_additional)
        X_additional = np.array(X_additional).reshape(-1, 1)

        # Combine the original training data with the additional data
        X_train = np.concatenate([X_train.reshape(-1, 1), X_additional])
        y_train = np.concatenate([y_train, y_additional])

    # Fit the Gaussian Process model
    gaussian_process.fit(X_train, y_train)

    # Make predictions
    mean_prediction, std_prediction = gaussian_process.predict(X.reshape(-1, 1), return_std=True)

    return mean_prediction, std_prediction

# Define wells data
wells_data = {
    9: {
        'months': np.linspace(start=1, stop=12, num=12).reshape(-1, 1),
        'y': np.array([19.42, 19.94, 19.92, 19.96, 20.03, 20.13, 20.25, 20.42, 20.63, 20.9, 21.25, 21.76]),
        'training_indices': [],
        'color': 'blue'
    },
    3: {
        'months': np.linspace(start=1, stop=12, num=12).reshape(-1, 1),
        'y': np.array([26.64, 27, 27.57, 28.26, 29.1, 30.11, 31.31, 32.75, 34.4, 36.32, 38.59, 41.45]),
        'training_indices': np.array([2,5,9]),
        'color': 'red'
    }
}

# Total steam flow data
total_steam_flow = {
    'months': np.linspace(start=1, stop=12, num=12).reshape(-1, 1),
    'y': np.array([46.06, 46.94, 47.49, 48.22, 49.13, 50.24, 51.56, 53.17, 55.03, 57.22, 59.84, 63.21])
}

X = total_steam_flow['months']
y_target_well = wells_data[3]['y']
TI_target_well = wells_data[3]['training_indices']
y_other_well = wells_data[9]['y']
TI_other_well = wells_data[9]['training_indices']
total_steam = total_steam_flow['y']

mean_pred_3, std_pred_3 = GPR_Model(X, y_target_well, TI_target_well, y_other_well, TI_other_well, total_steam)

y_target_well = wells_data[9]['y']
TI_target_well = wells_data[9]['training_indices']
y_other_well = wells_data[3]['y']
TI_other_well = wells_data[3]['training_indices']

mean_pred_9, std_pred_9 = GPR_Model(X, y_target_well, TI_target_well, y_other_well, TI_other_well, total_steam)

def realizations(total_steam_flow, month, mean_pred_well, std_pred_well, min_LB, max_UB):
    lower_bound = mean_pred_well[month] - 1.96 * std_pred_well[month]
    upper_bound = mean_pred_well[month] + 1.96 * std_pred_well[month]

    while True:
        well_realization = r.uniform(lower_bound, upper_bound)
        other_well_realization = total_steam_flow[month] - well_realization
        if other_well_realization >= min_LB[month] and other_well_realization <= max_UB[month]:
            return other_well_realization, well_realization


min_LB = mean_pred_9 - 1.96*std_pred_9
max_UB = mean_pred_9 + 1.96*std_pred_9

Realization_well_3 = np.zeros(12)
Realization_well_9 = np.zeros(12)

for i in range(12):
    Real_well_9, Real_well_3 = realizations(total_steam, i, mean_pred_3, std_pred_3, min_LB, max_UB)
