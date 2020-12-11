import faiss
import numpy as np
import copy
from util import dir_io

"""
get the gnd according to faiss
input:base, query, k
output: numpy file
"""


def get_gnd_numpy(base, query, k, save_dir=None):
    base_dim = base.shape[1]
    index = faiss.IndexFlatL2(base_dim)
    index.add(base)
    gnd_distance, gnd_idx = index.search(query, k)
    print("search")
    if save_dir is not None:
        dir_io.save_file(save_dir)
        np.save(save_dir, gnd_idx)
    return gnd_idx
