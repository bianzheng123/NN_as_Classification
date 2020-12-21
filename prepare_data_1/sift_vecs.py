import numpy as np
import _init_paths
import os
from util.vecs import vecs_io, vecs_util
from util import dir_io

'''
extract vecs file and output numpy file
'''


def vecs2numpy(fname, new_file_name, file_type, file_len=None):
    vectors = None
    if file_type == 'bvecs':
        vectors, dim = vecs_io.bvecs_read_mmap(fname)
    elif file_type == 'ivecs':
        vectors, dim = vecs_io.ivecs_read_mmap(fname)
    elif file_type == 'fvecs':
        vectors, dim = vecs_io.fvecs_read_mmap(fname)
    if file_len is not None and file_len != -1:
        vectors = vectors[:file_len]
    vectors = vectors.astype(np.float32)
    dir_io.save_numpy(new_file_name, vectors)
    return vectors


'''
make directory, extract base, query and gnd
'''


def convert_data_type(config):
    dir_io.mkdir(config['data_dir'])
    print("create directory")

    base_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['base'])
    base_npy_dir = '%s/%s' % (config['data_dir'], 'base.npy')
    base = vecs2numpy(base_dir, base_npy_dir, config['source_data_type']['base'])
    print("extract base")

    query_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['query'])
    query_npy_dir = '%s/%s' % (config['data_dir'], 'query.npy')
    query = vecs2numpy(query_dir, query_npy_dir, config['source_data_type']['query'], config['query_len'])
    print("extract query")

    learn_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['learn'])
    learn_npy_dir = '%s/%s' % (config['data_dir'], 'learn.npy')
    learn = vecs2numpy(learn_dir, learn_npy_dir, config['source_data_type']['learn'])
    print("extract learn")

    gnd_npy_dir = '%s/%s' % (config['data_dir'], 'gnd.npy')
    gnd = vecs_util.get_gnd_numpy(base, query, config['k'], gnd_npy_dir)
    print("extract gnd")

    base_base_gnd_npy_dir = '%s/%s' % (config['data_dir'], 'base_base_gnd.npy')
    # print(base_npy_dir)
    # print(query_npy_dir)
    # print(gnd_npy_dir)
    base_base_gnd = vecs_util.get_gnd_numpy(base, base, config['base_base_gnd_k'], base_base_gnd_npy_dir)
    print("extract base_base_gnd for the training set preparation")

    print(base.shape, query.shape, gnd.shape, learn.shape, base_base_gnd.shape)
    return base, query, gnd, learn, base_base_gnd


def prepare_data(config):
    data_dir = '%s/data/%s_%d' % (
        config['project_dir'], config['data_fname'], config['k'])
    print("data_dir", data_dir)
    config['data_dir'] = data_dir

    dir_io.delete_dir_if_exist(data_dir)
    '''
    dataset preparation
    make directory, extract base, query, learn, gnd
    '''
    convert_data_type(config)


if __name__ == '__main__':
    data_config = {
        "k": 10,
        "base_base_gnd_k": 150,
        "data_fname": "sift",
        "source_data_dir": "/home/bianzheng/Dataset/sift",
        "source_data_type": {
            "base": "fvecs",
            "query": "fvecs",
            "learn": "fvecs"
        },
        "source_data_fname": {
            "base": "sift_base.fvecs",
            "query": "sift_query.fvecs",
            "learn": "sift_learn.fvecs"
        },
        "project_dir": "/home/bianzheng/NN_as_Classification",
        "query_len": -1
    }
    prepare_data(data_config)
