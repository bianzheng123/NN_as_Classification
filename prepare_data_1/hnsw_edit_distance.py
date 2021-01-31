import random
from time import time
import nmslib

f = 1000 # length of vectors
seqs = []
for i in range(800):
    v = [random.randint(0, 2) for z in range(f)]
    print(v)
    seqs.append(''.join(map(str, v)))

t=time()
print("Made data. Now time to build index.")
index = nmslib.init(method='hnsw', space='leven', data_type=nmslib.DataType.OBJECT_AS_STRING, dtype=nmslib.DistType.INT)
index.addDataPointBatch(seqs)
index.createIndex(print_progress=True)
print("Index building time:", time()-t)
t = time()
NNqueryresult = index.knnQueryBatch(seqs, k=10)
print("Query time:", time()-t)