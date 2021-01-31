import os
import time
import numpy as np


def exec_partition():
    start = time.time()
    os.system(
        'mpirun -n 8 /home/bianzheng/software/KaHIP/deploy/parhip '
        '/home/bianzheng/NN_as_Classification/pytest/data/graph_data/graph_siftsmall.graph --k 256 '
        '--save_partition --preconfiguration fastsocial')
    end = time.time()
    print("time consumed: %d" % (end - start))


def calc_distribution():
    partition = np.loadtxt('/home/bianzheng/NN_as_Classification/pytest/tmppartition.txtp')
    arr = []
    for cluster_i in range(256):
        cls_l = np.argwhere(partition == cluster_i).reshape(-1)
        arr.append(len(cls_l))
    print(arr)


exec_partition()
# calc_distribution()
