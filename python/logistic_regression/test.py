import numpy as np


a = np.array([[1,2], [3,4]])

a = np.c_[a, [5,6]]
print(a)