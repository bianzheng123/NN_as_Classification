import numpy as np
import _init_paths
import os
from util.vecs import vecs_io
from util import dir_io, groundtruth

'''
extract vecs file and output numpy file
'''


def vecs2vecs(old_dir, new_dir, old_file_type, new_file_type, file_len=None):
    vectors = None
    if old_file_type == 'bvecs':
        vectors, dim = vecs_io.bvecs_read_mmap(old_dir)
    elif old_file_type == 'ivecs':
        vectors, dim = vecs_io.ivecs_read_mmap(old_dir)
    elif old_file_type == 'fvecs':
        vectors, dim = vecs_io.fvecs_read_mmap(old_dir)
    if file_len is not None and file_len < len(vectors) and file_len != -1:
        vectors = vectors[:file_len]
    vectors = vectors.astype(np.float32)

    if new_file_type == 'bvecs':
        vecs_io.bvecs_write(new_dir, vectors)
    elif new_file_type == 'ivecs':
        vecs_io.ivecs_write(new_dir, vectors)
    elif new_file_type == 'fvecs':
        vecs_io.fvecs_write(new_dir, vectors)
    return vectors


'''
make directory, extract base, query and gnd
'''


def convert_data_type(config):
    dir_io.mkdir(config['data_dir'])
    print("create directory")

    base_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['base'])
    base_save_dir = '%s/%s' % (config['data_dir'], 'base.fvecs')
    base = vecs2vecs(base_dir, base_save_dir, config['source_data_type']['base'], 'fvecs')
    print("extract base")

    query_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['query'])
    query_save_dir = '%s/%s' % (config['data_dir'], 'query.fvecs')
    query = vecs2vecs(query_dir, query_save_dir, config['source_data_type']['query'], 'fvecs',
                      file_len=config['query_len'])
    print("extract query")

    gnd = groundtruth.get_gnd(base, query, config['k'])
    gnd_save_dir = '%s/%s' % (config['data_dir'], 'gnd.ivecs')
    vecs_io.ivecs_write(gnd_save_dir, gnd)
    print("extract gnd")

    base_base_gnd = groundtruth.get_gnd(base, base, config['base_base_gnd_k'])
    base_base_gnd_save_dir = '%s/%s' % (config['data_dir'], 'base_base_gnd.ivecs')
    vecs_io.ivecs_write(base_base_gnd_save_dir, base_base_gnd)
    print("extract base_base_gnd for the training set preparation")

    print("base: ", base.shape)
    print("query: ", query.shape)
    print("gnd: ", gnd.shape)
    print("base_base_gnd: ", base_base_gnd.shape)
    return base, query, gnd, base_base_gnd


def prepare_data(config):
    data_dir = '%s/data/dataset/%s_%d' % (
        config['project_dir'], config['data_fname'], config['k'])
    print("data_dir", data_dir)
    config['data_dir'] = data_dir

    dir_io.delete_dir_if_exist(data_dir)
    '''
    dataset preparation
    make directory, extract base, query, gnd
    '''
    convert_data_type(config)


if __name__ == '__main__':
    data_config = {
        "k": 10,
        "base_base_gnd_k": 150,
        "data_fname": "siftsmall",
        "source_data_dir": "/home/zhengbian/Dataset/siftsmall",
        "source_data_type": {
            "base": "fvecs",
            "query": "fvecs"
        },
        "source_data_fname": {
            "base": "siftsmall_base.fvecs",
            "query": "siftsmall_query.fvecs"
        },
        "project_dir": "/home/zhengbian/NN_as_Classification",
        "query_len": -1
    }
    prepare_data(data_config)
