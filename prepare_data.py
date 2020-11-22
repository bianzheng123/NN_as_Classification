import json
import numpy as np
import os
from util.vecs import vecs_io, vecs_util

'''
提取vecs, 输出numpy文件
'''


def vecs2numpy(fname, new_file_name, file_type, file_len=None):
    if file_type == 'bvecs':
        vectors, dim = vecs_io.bvecs_read_mmap(fname)
    elif file_type == 'ivecs':
        vectors, dim = vecs_io.ivecs_read_mmap(fname)
    elif file_type == 'fvecs':
        vectors, dim = vecs_io.fvecs_read_mmap(fname)
    if file_len is not None:
        vectors = vectors[:file_len]
    vectors = vectors.astype(np.float32)
    np.save(new_file_name, vectors)
    return vectors


'''
创建文件夹, 提取base, query, gnd
'''


def convert_data_type(config):
    os.system("mkdir %s" % (config['data_dir']))
    print("创建文件夹")

    base_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['base'])
    base_npy_dir = '%s/%s' % (config['data_dir'], 'base.npy')
    base = vecs2numpy(base_dir, base_npy_dir, config['source_data_type']['base'])
    print("提取base")

    query_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['query'])
    query_npy_dir = '%s/%s' % (config['data_dir'], 'query.npy')
    query = vecs2numpy(query_dir, query_npy_dir, config['source_data_type']['query'])
    print("提取query")

    learn_dir = '%s/%s' % (config['source_data_dir'], config['source_data_fname']['learn'])
    learn_npy_dir = '%s/%s' % (config['data_dir'], 'learn.npy')
    learn = vecs2numpy(learn_dir, learn_npy_dir, config['source_data_type']['learn'])
    print("提取learn")

    gnd_npy_dir = '%s/%s' % (config['data_dir'], 'gnd.npy')
    # print(base_npy_dir)
    # print(query_npy_dir)
    # print(gnd_npy_dir)
    gnd = vecs_util.get_gnd_numpy(base, query, config['k'], gnd_npy_dir)
    print("提取gnd")
    print(base.shape, query.shape, gnd.shape, learn.shape)
    return base, query, gnd, learn


def delete_dir_if_exist(directory):
    if os.path.isdir(directory):
        command = 'rm -rf %s' % directory
        print(command)
        os.system(command)


if __name__ == '__main__':
    # 设置两个配置文件, 方便批量执行
    with open('config/prepare_data/config.json', 'r') as f:
        config = json.load(f)

    data_dir = '%s/data/%s_%d' % (
        config['project_dir'], config['data_fname'], config['k'])
    print("data_dir", data_dir)
    config['data_dir'] = data_dir

    delete_dir_if_exist(data_dir)
    '''
    数据准备
    创建文件夹, 提取base, query, learn, gnd
    '''
    convert_data_type(config)
