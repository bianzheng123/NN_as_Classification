import faiss
import numpy as np
import Levenshtein
import torch
import time
import multiprocessing
from multiprocessing.managers import BaseManager

"""
get the gnd according to faiss
input:base, query, k
output: numpy file
"""


def edit_distance(single_query, base):
    return [Levenshtein.distance(single_query, item) for item in base]


def gnd_parallel(obj, idx):
    query, base, total_process = obj.get_share_data()
    res_l = []
    for i in range(idx, len(query), total_process):
        if i % 100 == 0:
            print(i)
        res_l.append(edit_distance(query[i], base))
    return res_l


class GndString:
    def __init__(self, query, base, total_process):
        self.total_process = total_process
        self.query = query
        self.base = base

    def get_share_data(self):
        return self.query, self.base, self.total_process


def all_pair_distance(query, base, n_thread):
    manager = BaseManager()
    manager.register('GndString', GndString)
    manager.start()
    parallel_obj = manager.GndString(query, base, n_thread)
    res_l = []
    pool = multiprocessing.Pool(n_thread)
    for i in range(n_thread):
        res = pool.apply_async(gnd_parallel, args=(parallel_obj, i))
        res_l.append(res)
    pool.close()
    pool.join()

    gnd = [0] * len(query)
    print("complete parallel part")
    for i, tmp_res in enumerate(res_l, 0):
        tmp_res = tmp_res.get()
        for j in range(len(tmp_res)):
            gnd[i + j * n_thread] = tmp_res[j]

    return gnd


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
