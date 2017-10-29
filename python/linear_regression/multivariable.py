import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iterations = 35
alpha = 0.1

def read_data():
    data = np.loadtxt('data/housing_prices.in', delimiter=',')
    X = data[:, [0,1]]
    y = data[:, 2]
    y.shape = (y.shape[0], 1)
    return X, y

def normalize(X):
    return (X - X.mean(0))/X.std(0)

def add_xzero(X):
    return np.hstack((np.ones((X.shape[0],1)), X))

def gradient_descent(X, y):
    theta = np.zeros((X.shape[1],1))
    m = X.shape[0]
    cost = []
    for _ in range(iterations):
        X_transpose = np.transpose(X)
        cost_deriv = (alpha/m) * np.dot(X_transpose, np.dot(X, theta) - y)
        theta = theta - cost_deriv

        cost_func = np.sum(np.square(np.dot(X, theta) - y))/(2 * m)
        cost.append(cost_func)
    
    return theta, cost

def plot_cost_function(cost):

    plt.plot(cost)
    plt.xlabel("Iterations")
    plt.ylabel("Cost function")

    plt.show()

X, y = read_data()
X = add_xzero(normalize(X))
theta, cost = gradient_descent(X, y)
plot_cost_function(cost)

