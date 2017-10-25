import numpy as np
import matplotlib.pyplot as plt

iterations = 1500
alpha = 0.01

def read_data():
    data = np.loadtxt('restaurant_profit.in', delimiter=',')
    X = data[:, 0]
    y = data[:, 1]

    return X, y

def plot_best_fit(X, y, theta):

    plt.scatter(X, y, marker='x')
    plt.xlabel("Population of city in 10.000s")
    plt.ylabel("Profit in $10.000s")

    plt.plot(X, theta[0] + theta[1] * X)
    plt.show()

def gradient_descent(X, y, nr_iterations):

    m = X.shape[0]
    theta = np.zeros(2)
    for _ in range(nr_iterations):
        theta_zero = theta[0] - alpha/m * sum(theta[0] + theta[1] * X - y)
        theta_one = theta[1] - alpha/m * sum((theta[0] + theta[1] * X  - y) * X)

        theta[0] = theta_zero
        theta[1] = theta_one

    return theta


X, y = read_data()
plot_best_fit(X, y, gradient_descent(X, y, iterations))
