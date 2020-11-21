import faiss
import numpy as np

"""
通过相似性搜索得到gnd文件
输入: base数组, query数组, k(多少个临近的结果)
输出: numpy文件
"""


def get_gnd_numpy(base, query, k, save_dir=None):
    base_dim = base.shape[1]
    index = faiss.IndexFlatL2(base_dim)
    index.add(base)
    gnd_distance, gnd_idx = index.search(query, k)
    if save_dir is not None:
        np.save(save_dir, gnd_idx)
    return gnd_idx
