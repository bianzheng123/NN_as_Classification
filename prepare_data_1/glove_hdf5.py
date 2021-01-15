import numpy as np
import h5py
import _init_paths
from util import dir_io
from util.vecs import vecs_util


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


def prepare_data(config):
    data_dir = '%s/data/dataset/%s_%d' % (
        config['project_dir'], config['data_fname'], config['k'])
    print("data_dir", data_dir)
    config['data_dir'] = data_dir

    dir_io.delete_dir_if_exist(data_dir)
    dir_io.mkdir(data_dir)

    # read data
    hdfFile = h5py.File(config['source_data_dir'], 'r')

    base_info = config['file']['base']
    base = hdfFile.get(base_info['name'])
    if base_info['length'] != -1:
        base = base[:base_info['length']]
    save_base_dir = '%s/base.npy' % data_dir
    dir_io.save_numpy(save_base_dir, base)
    print("extract base")

    query_info = config['file']['query']
    query = hdfFile.get(query_info['name'])
    if query_info['length'] != -1:
        query = query[:query_info['length']]
    save_base_dir = '%s/query.npy' % data_dir
    dir_io.save_numpy(save_base_dir, query)
    print("extract query")

    learn = base
    save_base_dir = '%s/learn.npy' % data_dir
    dir_io.save_numpy(save_base_dir, learn)
    print("extract learn")

    gnd_npy_dir = '%s/gnd.npy' % data_dir
    gnd = vecs_util.get_gnd_numpy(base, query, config['k'], gnd_npy_dir)
    print("extract gnd")

    base_base_gnd_npy_dir = '%s/base_base_gnd.npy' % config['data_dir']
    base_base_gnd = vecs_util.get_gnd_numpy(base, base, config['base_base_gnd_k'], base_base_gnd_npy_dir)
    print("extract base_base_gnd for the training set preparation")

    print(base.shape, query.shape, learn.shape, gnd.shape, base_base_gnd.shape)
    hdfFile.close()


if __name__ == '__main__':
    data_config = {
        "k": 10,
        "base_base_gnd_k": 150,
        # "data_fname": "glove",
        "data_fname": "deep",
        # "source_data_dir": "/home/bianzheng/Dataset/glove-200-angular.hdf5",
        "source_data_dir": "/home/bianzheng/Dataset/deep-image-96-angular.hdf5",
        "file": {
            'base': {
                "name": "train",
                "length": 1000000,
            },
            'query': {
                "name": "test",
                "length": 1000
            }
        },
        "project_dir": "/home/bianzheng/NN_as_Classification"
    }
    prepare_data(data_config)
    # print_data_config_type(data_config['source_data_dir'])
