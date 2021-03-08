import _init_paths
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

fname = 'normalsmall_4_nn_4_partition_knn_'

x = np.loadtxt('data/result/%s/recall_l.txt' % fname)
x = x[10]
num_bins = 5

# the histogram of the data
plt.hist(x, bins=num_bins, density=True)

plt.xlabel('Smarts')
plt.ylabel('Probability density')

plt.show()
plt.savefig('%s1024.png' % fname)