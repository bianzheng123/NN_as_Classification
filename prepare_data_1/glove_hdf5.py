import numpy as np
import h5py
import _init_paths
import faiss


def prepare_data(config):
    # read data
    hdfFile = h5py.File('../glove-200-angular.hdf5', 'r')
    pass


if __name__ == '__main__':
    data_config = {
        "k": 10,
        "base_base_gnd_k": 150,
        "data_fname": "glove",
        "source_data_dir": "/home/bianzheng/Dataset/sift",
        "project_dir": "/home/bianzheng/NN_as_Classification",
        "query_len": -1
    }
    prepare_data(data_config)




def prt(name):
    print(name)


hdfFile.visit(prt)

distances = hdfFile.get('distances')
print(distances.shape)
test = hdfFile.get('test')
print(test.shape)
neighbors = hdfFile.get('neighbors')
print(neighbors.shape)
train = hdfFile.get('train')
print(train.shape)
hdfFile.close()
