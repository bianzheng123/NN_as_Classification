import numpy as np

a = np.array([])
b = np.array([[1, 2], [3, 4]])
a = np.append(a[np.newaxis, :], b, axis=0)
print(a)
