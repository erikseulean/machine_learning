import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from math import exp

alpha = 0.0001
iterations = 20000000

def read_data():
    data = np.loadtxt('data/student_grades.in', delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, 2]

    return X, y

def plot_data(X, y, point1, point2):
    X_ones = [(elem[0], elem[1]) for i, elem in enumerate(X) if y[i]]
    X_zeroes = [(elem[0], elem[1]) for i, elem in enumerate(X) if not y[i]]
    
    plt.scatter(*zip(*X_ones), marker="x")
    plt.scatter(*zip(*X_zeroes), marker="o")

    plt.plot(point1, point2)

    plt.show()

def gradient_descent(X, y):
    m = X.shape[0]
    theta = np.zeros(X.shape[1])
    cost_values = []

    for _ in range(iterations):
        scores = np.dot(X, theta)
        hypothesis = sigmoid(scores)

        # update theta
        theta = theta - (alpha/m) * np.dot(np.transpose(X), hypothesis - y)
        cost = -1/m * np.sum(np.dot(y.T, np.log(hypothesis)) + np.dot((1 - y).T, np.log(1 - hypothesis)))
        cost_values.append(cost)
    print(theta)
    return cost_values

def plot_cost_function(cost):

    plt.plot(cost)
    plt.xlabel("Iterations")
    plt.ylabel("Cost function")

    plt.show()


def add_intercept(X):
    return np.hstack((np.ones((X.shape[0],1)), X))

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def decision_boundary(X, y, theta):
    x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 2]) + 2])
    y = -1/theta[2] * (theta[0] + x * theta[1])
    return x, y
    

X, y = read_data()

#cost = gradient_descent(add_intercept(X), y)
#plot_cost_function(cost)

theta  = np.array([-25.161, 0.206, 0.201])
point1, point2 = decision_boundary(add_intercept(X), y, theta)
plot_data(X, y, point1, point2)

