import numpy as np
import _init_paths
from util.vecs import vecs_io
from util import dir_io, send_email, groundtruth
import numba as nb
import time
import multiprocessing
import math
import nmslib


# get the alphabet and the data
def read_txt(dire, n_character=None, text_len=None, word_len=None):
    with open(dire, "r") as f:
        txt = f.read().split('\n')[:-1]

    alphabet = set()
    if n_character is not None:
        exit_flag = False
        for row in txt:
            for c in row:
                if c not in alphabet:
                    alphabet.add(c)
                if len(alphabet) == n_character:
                    exit_flag = True
                if exit_flag:
                    break
            if exit_flag:
                break

    if text_len is not None and text_len < len(txt) and text_len != -1:
        txt = txt[:text_len]
    if word_len is not None:
        for i in range(len(txt)):
            if len(txt[i]) > word_len:
                txt[i] = txt[i][:word_len]

    if n_character is not None:
        return txt, alphabet

    return txt


# convert the string to one hot encoding and padding
def words2vector(word_l, padding_len, alphabet):
    vectors = []
    for word in word_l:
        coding_l = []
        for char in alphabet:
            tmp = [1 if _ == char else 0 for _ in word]
            if len(tmp) < padding_len:
                tmp += [0] * (padding_len - len(tmp))
            coding_l.append(tmp)
        coding_l = np.array(coding_l).reshape(-1)
        vectors.append(coding_l)
    return np.array(vectors, dtype=np.int)


def text_count(config):
    base_dir = '%s/%s' % (config['source_data_dir'], config['source_data']['base']['name'])
    base = read_txt(base_dir)
    print(len(base))
    query_dir = '%s/%s' % (config['source_data_dir'], config['source_data']['query']['name'])
    query = read_txt(query_dir)
    print(len(query))


def get_base_base_gnd(base, config):
    index = nmslib.init(method='hnsw', space='leven', data_type=nmslib.DataType.OBJECT_AS_STRING,
                        dtype=nmslib.DistType.INT)
    index.addDataPointBatch(base)
    index.createIndex(print_progress=True)
    res = index.knnQueryBatch(base, k=config['base_base_gnd_k'],
                              num_threads=multiprocessing.cpu_count())  # 0: index, 1: distance
    base_base_gnd = []
    min_val = config['base_base_gnd_k']
    for i in range(len(res)):
        min_val = min(min_val, len(res[i][0]))
    print("true base_base_gnd k %d" % min_val)
    for i in range(len(res)):
        base_base_gnd.append(res[i][0][:min_val])
    base_base_gnd = np.array(base_base_gnd)
    return base_base_gnd, min_val


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
    dataset, alphabet = read_txt(dataset_dir, word_len=config['padding_length'], n_character=config['n_character'])
    print(alphabet)

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
    base_base_gnd, n_base_base_gnd = get_base_base_gnd(base, config)
    base_base_gnd_save_dir = '%s/%s' % (data_dir, 'base_base_gnd.ivecs')
    vecs_io.ivecs_write(base_base_gnd_save_dir, base_base_gnd)
    end = time.time()
    print("save base_base_gnd for the training set preparation, time:", end - start)
    print("base_base_gnd: ", base_base_gnd.shape)
    del base_base_gnd

    # encoding and padding
    base_save_dir = '%s/%s' % (data_dir, 'base.ivecs')
    start = time.time()
    base = words2vector(base, config['padding_length'], alphabet)
    print("wrods2vector time consume %d" % (time.time() - start))
    vecs_io.ivecs_write(base_save_dir, base)
    print("save base")
    print("base: ", base.shape)

    query_save_dir = '%s/%s' % (data_dir, 'query.ivecs')
    query = words2vector(query, config['padding_length'], alphabet)
    vecs_io.ivecs_write(query_save_dir, query)
    print("save query")
    print("query: ", query.shape)

    description_dir = '%s/%s' % (data_dir, 'readme.txt')
    ptr = dir_io.write_ptr(description_dir)
    ptr.write('the max base_base_gnd is %d\n' % n_base_base_gnd)
    ptr.close()


if __name__ == '__main__':
    data_config = {
        "k": 10,
        "n_character": 24,
        "base_base_gnd_k": 500,
        "padding_length": 5000,
        "data_fname": "uniref",
        "source_data_dir": "/home/zhengbian/Dataset/uniref",
        "source_data": {
            # 400000 in total
            "name": "uniref.txt",
            "base_len": 399000,
            "query_len": 1000
        },
        "project_dir": "/home/zhengbian/NN_as_Classification"
    }
    prepare_data(data_config)
    # text_count(data_config)
    # send_email.send('success for ' + data_config["data_fname"])
