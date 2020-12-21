from procedure_nn_classification.init_0.kmeans import multiple_kmeans, independent_kmeans
from procedure_nn_classification.init_0.learn_on_graph import multiple_learn_on_graph
from util import dir_io
import time


def preprocess(base, config):
    program_train_para_dir = config['program_train_para_dir']
    dir_io.mkdir(program_train_para_dir)
    n_cluster = config['n_cluster']
    kahip_dir = config['kahip_dir']
    partition_model_l = []
    model_intermediate_m = {}
    start_time = time.time()
    for entity_number, tmp_config in enumerate(config['independent_config'], 1):
        tmp_config['program_train_para_dir'] = program_train_para_dir
        tmp_config['n_cluster'] = n_cluster
        tmp_config['kahip_dir'] = kahip_dir
        tmp_config['entity_number'] = entity_number
        multiple_model = factory(tmp_config)
        tmp_model_l, model_intermediate = multiple_model.preprocess(base)
        model_intermediate_m[model_intermediate['signature']] = model_intermediate['intermediate']
        for model in tmp_model_l:
            partition_model_l.append(model)
    end_time = time.time()
    model_intermediate_m['total_time'] = end_time - start_time
    return partition_model_l, model_intermediate_m


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
