import numpy as np

miu = 10
sigma = 10

base = np.random.normal(loc=miu, scale=sigma,
                            size=(100, 5))
lsh_miu = 0
lsh_sigma = 32
lsh_r = 1

norm = np.linalg.norm(base, axis=1)
# print(norm)
norm_div = np.max(norm)
# print(norm_div)
base_normlize = base / norm_div
a = np.random.normal(loc=lsh_miu, scale=lsh_sigma, size=5)
proj_result = np.dot(base_normlize, a)
b = np.random.random() * lsh_r
arr = np.floor((proj_result + b) / lsh_r) % 16
print(arr)