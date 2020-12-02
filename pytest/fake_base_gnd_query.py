import _init_paths
import faiss
import os
import numpy as np

base = np.array([[1, 2], [1, 5], [2, 2], [3, 5], [4, 5], [5, 10], [10, 15], [2, 2]])
query = np.array([[1, 2], [5, 5]])
learn = np.array([[1, 2], [10, 10]])
k = 2
base = base.astype(np.float32)
query = query.astype(np.float32)
learn = learn.astype(np.float32)

base_dim = base.shape[1]
index = faiss.IndexFlatL2(base_dim)
index.add(base)
gnd_distance, gnd_idx = index.search(query, k)

save_dir = '/home/bz/NN_as_Classification/data/test_2/'
os.system('mkdir %s' % save_dir)
np.save(save_dir + 'base.npy', base)
np.save(save_dir + 'query.npy', query)
np.save(save_dir + 'learn.npy', learn)
np.save(save_dir + 'gnd.npy', gnd_idx)
