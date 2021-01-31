import numpy as np
import _init_paths
import Levenshtein
import os
from util.vecs import vecs_io
from util import dir_io, send_email, groundtruth
import numba as nb
import time
import math


def read_txt(dire, leng=None, word_len=None):
    with open(dire, "r") as f:
        txt = f.read().split('\n')[:-1]
        if leng is not None and leng < len(txt) and leng != -1:
            txt = txt[:leng]
        if word_len is not None:
            for i in range(len(txt)):
                if len(txt[i]) > word_len:
                    txt[i] = txt[i][:word_len]
                    print(len(txt[i]))
    return txt


def words2vector(word_l, dimension):
    vectors = []
    for word in word_l:
        vecs = [ord(_) for _ in word]
        tmp_num = ord('-')
        len_word = len(word)
        if len_word < dimension:
            vecs += [tmp_num] * (dimension - len_word)
        elif len_word > dimension:
            vecs = vecs[:dimension]
            print(len(vecs))
        vectors.append(vecs)
    return np.array(vectors, dtype=np.int)


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

    dataset_dir = '%s/%s' % (config['source_data_dir'], config['source_data']['name'])
    dataset = read_txt(dataset_dir, word_len=config['padding_length'])
    base = dataset[:config['source_data']['base_len']]

    query = dataset[-config['source_data']['query_len']:]

    start = time.time()
    gnd = groundtruth.get_gnd(base, query, config['k'], metrics="string")
    gnd_save_dir = '%s/%s' % (data_dir, 'gnd.ivecs')
    vecs_io.ivecs_write(gnd_save_dir, gnd)
    end = time.time()
    print("save gnd, time:", end - start)
    print("gnd: ", gnd.shape)
    del gnd

    start = time.time()
    batch_size = 100
    print("batch size %d" % batch_size)
    base_base_gnd = np.empty(shape=(len(base), config['base_base_gnd_k']), dtype=np.int)
    for i in range(0, len(base), batch_size):
        if i + batch_size > len(base):
            tmp_query = base[i:]
        else:
            tmp_query = base[i: i + batch_size]

        seg_base_base_gnd = groundtruth.get_gnd(base, tmp_query, config['base_base_gnd_k'], metrics="string")
        if i + batch_size > len(base):
            base_base_gnd[i:] = seg_base_base_gnd
        else:
            base_base_gnd[i: i + batch_size] = seg_base_base_gnd
        if i % 10 == 0:
            print(i)
    base_base_gnd_save_dir = '%s/%s' % (data_dir, 'base_base_gnd.ivecs')
    vecs_io.ivecs_write(base_base_gnd_save_dir, base_base_gnd)
    end = time.time()
    print("save base_base_gnd for the training set preparation, time:", end - start)
    print("base_base_gnd: ", base_base_gnd.shape)
    del base_base_gnd

    base_save_dir = '%s/%s' % (data_dir, 'base.ivecs')
    base = words2vector(base, config['padding_length'])
    vecs_io.ivecs_write(base_save_dir, base)
    print("save base")
    print("base: ", base.shape)

    query_save_dir = '%s/%s' % (data_dir, 'query.ivecs')
    query = words2vector(query, config['padding_length'])
    vecs_io.ivecs_write(query_save_dir, query)
    print("save query")
    print("query: ", query.shape)


if __name__ == '__main__':
    data_config = {
        "k": 10,
        "base_base_gnd_k": 100,
        "padding_length": 5000,
        "data_fname": "unirefsmall",
        "source_data_dir": "/home/zhengbian/Dataset/uniref",
        "source_data": {
            # 400000 in total
            "name": "uniref.txt",
            "base_len": 1000,
            "query_len": 100
        },
        "project_dir": "/home/zhengbian/NN_as_Classification"
    }
    prepare_data(data_config)
    # text_count(data_config)
    # send_email.send('success for ' + data_config["data_fname"])
