from procedure_nn_classification.dataset_partition_1 import hash, kmeans
from procedure_pq_nn.dataset_partition_1 import pq_learn_on_graph
from util import dir_io
import time
import numpy as np
import copy


def preprocess(config):
    dir_io.mkdir(config['program_train_para_dir'])
    start_time = time.time()
    model_ins = factory(config['dataset_partition']['type'])

    model_l = []
    for i in range(config['n_instance']):
        tmp_config = copy.deepcopy(config['dataset_partition'])
        tmp_config['type'] = config['dataset_partition']['type']
        tmp_config['classifier_number'] = i
        tmp_config['kahip_dir'] = config['kahip_dir']
        tmp_config['save_dir'] = '%s/Classifier_%d' % (
            config['program_train_para_dir'], tmp_config['classifier_number'])
        tmp_config['n_cluster'] = config['n_cluster']
        dir_io.mkdir(tmp_config['save_dir'])
        tmp_model = model_ins(tmp_config)
        model_l.append(tmp_model)

    end_time = time.time()
    model_intermediate = {
        'total_time': end_time - start_time
    }
    return model_l, model_intermediate


def factory(_type):
    if _type == 'kmeans_independent':
        return kmeans.IndependentKMeans
    elif _type == 'knn' or _type == 'hnsw':
        return pq_learn_on_graph.PQLearnOnGraph
    elif _type == 'random_hash':
        return hash.RandomHash
    elif _type == 'lsh':
        return hash.LocalitySensitiveHash
    raise Exception('do not support the type of partition')
