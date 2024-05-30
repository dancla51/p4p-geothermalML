# import numpy as np
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
#
# # Define the model function
# def model_func(x, a, b, c):
#     return a * x**2 + b * x + c
#
# # Define the data
# X = np.linspace(start=1, stop=12, num=12)
# y = np.array([19.42,19.94,19.92,19.96,20.03,20.13,20.25,20.42,20.63,20.9,21.25,21.76])
# training_indices = [0,1,2,3,4,5,6,7,8,9,10]
#
# X_train = X[training_indices]
# y_train = y[training_indices]
#
# # Perform curve fitting
# popt, pcov = curve_fit(model_func, X_train, y_train)
#
# # Get the parameters
# a, b, c = popt
#
# # Predicted values
# y_pred = model_func(X, a, b, c)
#
# # Plotting
# plt.scatter(X, y, label='Data')
# plt.plot(X, y_pred, color='red', label='Fitted curve')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.title('Curve Fitting')
# plt.show()

from scipy.interpolate import interp1d
import numpy as np
from numpy import arange, array, exp

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

# x = arange(0,10)
# y = exp(-x/3.0)

X = np.linspace(start=1, stop=11, num=11)
y = np.array([19.42,19.94,19.92,19.96,20.03,20.13,20.25,20.42,20.63,20.9,21.25])

f_i = interp1d(X, y)
f_x = extrap1d(f_i)

print(f_x([11,15]))