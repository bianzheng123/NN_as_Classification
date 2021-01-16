import numpy as np
import _init_paths
import Levenshtein
import os
from util.vecs import vecs_io, vecs_util
from util import dir_io
import numba as nb
import time


def read_txt(dire, leng=None):
    with open(dire, "r") as f:
        txt = f.read().split('\n')[:-1]
        if leng is not None and leng < len(txt) and leng != -1:
            txt = txt[:leng]
    return txt


def text_count(config):
    base_dir = '%s/%s' % (config['source_data_dir'], config['source_data']['base']['name'])
    base = read_txt(base_dir)
    print(len(base))
    query_dir = '%s/%s' % (config['source_data_dir'], config['source_data']['query']['name'])
    query = read_txt(query_dir)
    print(len(query))


def prepare_data(config):
    data_dir = '%s/data/dataset/%s_%d' % (
        config['project_dir'], config['data_fname'], config['k'])
    print("data_dir", data_dir)

    dir_io.delete_dir_if_exist(data_dir)
    '''
    dataset preparation
    make directory, extract base, query, gnd
    '''

    dir_io.mkdir(data_dir)
    print("create directory")

    base_dir = '%s/%s' % (config['source_data_dir'], config['source_data']['base']['name'])
    base = read_txt(base_dir, leng=config['source_data']['base']['len'])
    base_save_dir = '%s/%s' % (data_dir, 'base.txt')
    dir_io.save_array_txt(base_save_dir, base, '%s')
    print("save base")

    query_dir = '%s/%s' % (config['source_data_dir'], config['source_data']['query']['name'])
    query = read_txt(query_dir, leng=config['source_data']['query']['len'])
    query_save_dir = '%s/%s' % (data_dir, 'query.txt')
    dir_io.save_array_txt(query_save_dir, query, '%s')
    print("save query")

    start = time.time()
    gnd = vecs_util.get_gnd(base, query, config['k'], metrics="string")
    gnd_save_dir = '%s/%s' % (data_dir, 'gnd.ivecs')
    vecs_io.ivecs_write(gnd_save_dir, gnd)
    end = time.time()
    print("save gnd, time:", end - start)

    start = time.time()
    base_base_gnd = vecs_util.get_gnd(base, base, config['base_base_gnd_k'], metrics="string")
    base_base_gnd_save_dir = '%s/%s' % (data_dir, 'base_base_gnd.ivecs')
    vecs_io.ivecs_write(base_base_gnd_save_dir, base_base_gnd)
    end = time.time()
    print("save base_base_gnd for the training set preparation, time:", end - start)

    print("base: ", len(base))
    print("query: ", len(query))
    print("gnd: ", gnd.shape)
    print("base_base_gnd: ", base_base_gnd.shape)
    return base, query, gnd, base_base_gnd


if __name__ == '__main__':
    data_config = {
        "k": 10,
        "base_base_gnd_k": 150,
        "data_fname": "unirefsmall",
        "source_data_dir": "/home/zhengbian/Dataset/uniref",
        "source_data": {
            "base": {
                "name": "uniref.txt",
                "len": 1000
            },
            "query": {
                "name": "unirefquery.txt",
                "len": 10
            }
        },
        "project_dir": "/home/zhengbian/NN_as_Classification"
    }
    prepare_data(data_config)
    # text_count(data_config)
