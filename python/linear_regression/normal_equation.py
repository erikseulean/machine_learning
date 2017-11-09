import numpy as np

pinv = np.linalg.pinv

def read_data():
    data = np.loadtxt('data/housing_prices.in', delimiter=',')
    X = data[:, [0,1]]
    y = data[:, 2]
    y.shape = (y.shape[0], 1)
    return X, y

def add_xzero(X):
    return np.hstack((np.ones((X.shape[0],1)), X))

def normal_equation(X, y):
    X_transpose = np.transpose(X)
    theta = np.dot(np.dot(pinv(np.dot(X_transpose, X)), X_transpose), y)
    return theta


X, y = read_data()
X = add_xzero(X)
theta = normal_equation(X, y)
print(theta)