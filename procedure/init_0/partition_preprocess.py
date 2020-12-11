from procedure.init_0.kmeans import multiple_kmeans, independent_kmeans
from procedure.init_0.learn_on_graph import multiple_learn_on_graph
import os
import time


def preprocess(base, config):
    program_train_para_dir = config['program_train_para_dir']
    os.system("sudo mkdir %s" % program_train_para_dir)
    n_cluster = config['n_cluster']
    kahip_dir = config['kahip_dir']
    partition_model_l = []
    start_time = time.time()
    for entity_number, tmp_config in enumerate(config['independent_config'], 1):
        tmp_config['program_train_para_dir'] = program_train_para_dir
        tmp_config['n_cluster'] = n_cluster
        tmp_config['kahip_dir'] = kahip_dir
        tmp_config['entity_number'] = entity_number
        multiple_model = factory(tmp_config)
        tmp_model_l = multiple_model.preprocess(base)
        for model in tmp_model_l:
            partition_model_l.append(model)
    end_time = time.time()
    intermediate_result = {
        'time': end_time - start_time
    }
    return partition_model_l, intermediate_result


def factory(config):
    _type = config['type']
    if _type == 'kmeans':
        _specific_type = config['specific_type']
        # multiple_kmeans_batch, independent_kmeans, multiple_kmeans_self_impl
        if _specific_type == 'multiple':
            return multiple_kmeans.MultipleKMeans(config)
        elif _specific_type == 'independent':
            return independent_kmeans.IndependentKMeans(config)
    elif _type == 'learn_on_graph':
        return multiple_learn_on_graph.MultipleLearnOnGraph(config)
    raise Exception('do not support the type of partition')
