import faiss
import numpy as np
import Levenshtein
import torch
import time
import multiprocessing

"""
get the gnd according to faiss
input:base, query, k
output: numpy file
"""


def edit_distance(x):
    a, B = x
    return [Levenshtein.distance(a, b) for b in B]


def all_pair_distance(A, B, n_thread, progress=True):

    def all_pair(A, B, n_thread):
        with multiprocessing.Pool(n_thread) as pool:
            start_time = time.time()
            edit = list(pool.imap(edit_distance, zip(A, [B for _ in A])))
            if progress:
                print("# Calculate edit distance time: {}".format(time.time() - start_time))
            return np.array(edit)

    if len(A) < len(B):
        return all_pair(B, A, n_thread).T
    else:
        return all_pair(A, B, n_thread)


def gnd_string(base, query, k):
    dist = all_pair_distance(query, base, multiprocessing.cpu_count(), progress=False)
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
