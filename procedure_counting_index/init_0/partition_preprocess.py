from procedure_counting_index.init_0 import preprocess_lsh, preprocess_kmeans, preprocess_kmeans_multiple
from util import dir_io
import time


def preprocess(base, config):
    program_train_para_dir = config['program_train_para_dir']
    dir_io.mkdir(program_train_para_dir)
    n_cluster = config['n_cluster']
    partition_model_l = []
    model_intermediate_m = {}
    start_time = time.time()
    for entity_number, tmp_config in enumerate(config['independent_config'], 1):
        tmp_config['program_train_para_dir'] = program_train_para_dir
        tmp_config['n_cluster'] = n_cluster
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
    if _type == 'kmeans_independent':
        return preprocess_kmeans.PreprocessKMeans(config)
    elif _type == 'kmeans_multiple':
        return preprocess_kmeans_multiple.PreprocessKMeansMultiple(config)
    elif _type == 'e2lsh':
        return preprocess_lsh.PreprocessLSH(config)
    raise Exception('do not support the type of partition')
