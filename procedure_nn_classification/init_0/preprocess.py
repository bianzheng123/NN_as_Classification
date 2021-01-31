from procedure_nn_classification.init_0 import multiple_model
from util import dir_io
import time
import numpy as np
import copy


def preprocess(base, config):
    dir_io.mkdir(config['program_train_para_dir'])
    start_time = time.time()
    multiple_model_ins = factory(config)
    model_l, model_intermediate = multiple_model_ins.preprocess(base)

    end_time = time.time()
    model_intermediate['total_time'] = end_time - start_time
    return model_l, model_intermediate


def factory(config):
    _type = config['dataset_partition']['type']
    if _type == 'kmeans_multiple':
        return multiple_model.MultipleKMeans(config)
    elif _type == 'kmeans_independent':
        return multiple_model.IndependentKMeans(config)
    elif _type == 'knn' or _type == 'hnsw':
        return multiple_model.MultipleLearnOnGraph(config)
    elif _type == 'random_hash' or _type == 'lsh':
        return multiple_model.MultipleHash(config)
    raise Exception('do not support the type of partition')
