from procedure_counting_index.init_0 import preprocess_lsh, preprocess_kmeans, preprocess_kmeans_multiple
import os
import time


def preprocess(base, config):
    program_train_para_dir = config['program_train_para_dir']
    os.system("mkdir %s" % program_train_para_dir)
    n_cluster = config['n_cluster']
    partition_model_l = []
    start_time = time.time()
    for entity_number, tmp_config in enumerate(config['independent_config'], 1):
        tmp_config['program_train_para_dir'] = program_train_para_dir
        tmp_config['n_cluster'] = n_cluster
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
        return preprocess_kmeans.PreprocessKMeans(config)
    elif _type == 'kmeans_multiple':
        return preprocess_kmeans_multiple.PreprocessKMeansMultiple(config)
    elif _type == 'e2lsh':
        return preprocess_lsh.PreprocessLSH(config)
    raise Exception('do not support the type of partition')
