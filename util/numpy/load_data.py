import numpy as np


def load_data_npy(config):
    data_dir = config['data_dir']
    base_dir = '%s/base.npy' % data_dir
    base = np.load(base_dir)

    query_dir = '%s/query.npy' % data_dir
    query = np.load(query_dir)

    gnd_dir = '%s/gnd.npy' % data_dir
    gnd = np.load(gnd_dir)

    learn_dir = '%s/learn.npy' % data_dir
    learn = np.load(learn_dir)

    return base, query, learn, gnd
