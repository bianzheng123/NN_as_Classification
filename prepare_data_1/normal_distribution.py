import _init_paths
from util import dir_io, groundtruth
from util.vecs import vecs_io
import numpy as np

np.random.seed(123)


def prepare_data(config):
    save_dir = '%s/data/dataset/%s_%d' % (config['save_dir'], config['data_fname'], config['k'])
    dir_io.delete_dir_if_exist(save_dir)
    dir_io.mkdir(save_dir)
    base = np.random.normal(loc=config['miu'], scale=data_config['sigma'],
                            size=(config['base']['length'], config['base']['dim']))
    base_save_dir = '%s/base.fvecs' % save_dir
    vecs_io.fvecs_write(base_save_dir, base)

    query = np.random.normal(loc=config['miu'], scale=data_config['sigma'],
                             size=(config['query']['length'], config['query']['dim']))
    query_save_dir = '%s/query.fvecs' % save_dir
    vecs_io.fvecs_write(query_save_dir, query)

    base = base.astype(np.float32)
    query = query.astype(np.float32)

    gnd = groundtruth.get_gnd(base, query, config['k'])
    gnd_save_dir = '%s/gnd.ivecs' % save_dir
    vecs_io.ivecs_write(gnd_save_dir, gnd)

    base_base_gnd = groundtruth.get_gnd(base, base, config['base_base_gnd_k'])
    base_base_gnd_save_dir = '%s/base_base_gnd.ivecs' % save_dir
    vecs_io.ivecs_write(base_base_gnd_save_dir, base_base_gnd)

    print("base", base.shape)
    print("query", query.shape)
    print("gnd", gnd.shape)
    print("base_base_gnd", base_base_gnd.shape)


if __name__ == '__main__':
    data_config = {
        'data_fname': 'normal',
        'miu': 0,
        'sigma': 100,
        'save_dir': '/home/zhengbian/NN_as_Classification',
        'base': {
            'length': 1000000,
            'dim': 2
        },
        'query': {
            'length': 1000,
            'dim': 2
        },
        'base_base_gnd_k': 150,
        'k': 10
    }
    prepare_data(data_config)
