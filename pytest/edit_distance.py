import _init_paths
from util.vecs import vecs_io
from util import groundtruth
import numpy as np
import Levenshtein
import torch
import time
import multiprocessing


def edit_distance(x):
    a, B = x
    return [Levenshtein.distance(a, b) for b in B]


def all_pair_distance(A, B, n_thread):
    def all_pair(A, B, n_thread):
        with multiprocessing.Pool(n_thread) as pool:
            start_time = time.time()
            edit = list(pool.imap(edit_distance, zip(A, [B for _ in A])), )
            return np.array(edit)

    if len(A) < len(B):
        return all_pair(B, A, n_thread).T
    else:
        return all_pair(A, B, n_thread)


def gnd_string(base, query, k):
    dist = all_pair_distance(query, base, multiprocessing.cpu_count() // 4 * 3)
    dist_torch = torch.IntTensor(dist)
    distance, idx = torch.topk(dist_torch, k, largest=False, dim=1)
    return distance.numpy(), idx.numpy()


def get_gnd(base, query, k, metrics='l2'):
    if metrics == 'l2':
        base_dim = base.shape[1]
        index = faiss.IndexFlatL2(base_dim)
        index.add(base)
        gnd_distance, gnd_idx = index.search(query, k)
        print("search")
        return gnd_idx
    elif metrics == 'string':
        gnd_distance, gnd_idx = gnd_string(base, query, k)
        return gnd_idx
    raise Exception("not support the metrics")


def read_txt(dire, leng=None):
    with open(dire, "r") as f:
        txt = f.read().split('\n')[:-1]
        if leng is not None and leng < len(txt) and leng != -1:
            txt = txt[:leng]
    return txt


def string_gnd():
    query = ['156  ', '4  ']
    base = ['123', '4']

    distance, idx = groundtruth.gnd_string(base, query, 2)
    print(distance)
    print(idx)


def get_max_length(dire):
    with open(dire, 'r') as f:
        text = f.read().split('\n')[:-1]

        # print(max([len(_) + 1 for _ in text]))
        print(len(text))


# dire = "/home/zhengbian/Dataset/uniref/uniref.txt"
# get_max_length(dire)
# dire = "/home/zhengbian/Dataset/uniref/unirefquery.txt"
# get_max_length(dire)
# string_gnd()
total = read_txt('/home/zhengbian/Dataset/uniref/uniref.txt')
base = total[:1000]
query = total[:1000]
gnd = get_gnd(base, query, 100, 'string')

test = vecs_io.ivecs_read('/home/zhengbian/NN_as_Classification/data/dataset/unirefsmall_10/base_base_gnd.ivecs')[0]
same = True

for i in range(len(gnd)):
    for j in range(len(gnd[0])):
        if gnd[i][j] != test[i][j]:
            # print("%d %d" % (i, j))
            same = False
print(gnd)
print(test)
if same:
    print("same")
else:
    print("not same")
