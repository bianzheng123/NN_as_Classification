import numpy as np
import h5py
import _init_paths
from util import dir_io, groundtruth
from util.vecs import vecs_io


def print_data_config_type(data_dir):
    hdfFile = h5py.File(data_dir, 'r')

    def prt(name):
        print(name)

    # print the dataset name in hdf5
    hdfFile.visit(prt)
    distances = hdfFile.get('distances')
    print('distances', distances.shape)
    test = hdfFile.get('test')
    print('test', test.shape)
    neighbors = hdfFile.get('neighbors')
    print('neighbors', neighbors.shape)
    train = hdfFile.get('train')
    print('train', train.shape)
    hdfFile.close()


def normalization(vectors):
    vecs_module = np.linalg.norm(vectors, axis=1)
    vecs_module = vecs_module[:, np.newaxis]
    vectors = vectors / vecs_module
    return vectors


def prepare_data(config):
    data_dir = '%s/data/dataset/%s' % (
        config['project_dir'], config['data_fname'])
    print("data_dir", data_dir)
    config['data_dir'] = data_dir

    dir_io.delete_dir_if_exist(data_dir)
    dir_io.mkdir(data_dir)

    # read data
    hdfFile = h5py.File(config['source_data_dir'], 'r')

    base_info = config['file']['base']
    base = hdfFile.get(base_info['name'])
    if base_info['length'] != -1 and base_info['length'] < len(base):
        base = base[:base_info['length']]
    if config['normalization']:
        base = normalization(base)
        print("normalize base")
    base = base.astype(np.float32)
    save_base_dir = '%s/base.fvecs' % data_dir
    vecs_io.fvecs_write(save_base_dir, base)
    print("save base")

    query_info = config['file']['query']
    query = hdfFile.get(query_info['name'])
    if query_info['length'] != -1 and query_info['length'] < len(query):
        query = query[:query_info['length']]
    if config['normalization']:
        query = normalization(query)
        print("normalize query")
    query = query.astype(np.float32)
    save_query_dir = '%s/query.fvecs' % data_dir
    vecs_io.fvecs_write(save_query_dir, query)
    print("save query")

    save_gnd_dir = '%s/gnd-%d.ivecs' % (data_dir, config['k'])
    gnd = groundtruth.get_gnd(base, query, config['k'])
    vecs_io.ivecs_write(save_gnd_dir, gnd)
    print("save gnd")

    base_base_gnd_npy_dir = '%s/base_base_gnd-%d.ivecs' % (config['data_dir'], config['base_base_gnd_k'])
    base_base_gnd = groundtruth.get_gnd(base, base, max(config['base_base_gnd_k'], config['k']))
    vecs_io.ivecs_write(base_base_gnd_npy_dir, base_base_gnd)
    print("save base_base_gnd for the training set preparation")

    print("base:", base.shape)
    print("query:", query.shape)
    print("gnd:", gnd.shape)
    print("base_base_gnd:", base_base_gnd.shape)
    hdfFile.close()


if __name__ == '__main__':
    data_config = {
        "k": 50,
        "base_base_gnd_k": 150,
        "data_fname": "deep_big",
        "source_data_dir": "/home/zhengbian/Dataset/deep-image-96-angular.hdf5",
        # "source_data_dir": "/home/zhengbian/Dataset/lastfm-64-dot.hdf5",
        "file": {
            'base': {
                "name": "train",
                "length": 1000000000,
            },
            'query': {
                "name": "test",
                "length": 1000
            }
        },
        "project_dir": "/home/zhengbian/NN_as_Classification",
        "normalization": True
    }
    prepare_data(data_config)
    # print_data_config_type(data_config['source_data_dir'])
