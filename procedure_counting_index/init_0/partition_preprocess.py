from procedure_counting_index.init_0 import preprocess_model
from util import dir_io
import time


def preprocess(base, config):
    program_train_para_dir = config['program_train_para_dir']
    dir_io.mkdir(program_train_para_dir)

    start_time = time.time()
    ds_partition_config = config['dataset_partition']
    ds_partition_config['program_train_para_dir'] = program_train_para_dir
    ds_partition_config['n_cluster'] = config['n_cluster']
    ds_partition_config['n_instance'] = config['n_instance']
    multiple_model = factory(ds_partition_config)
    model_l, model_intermediate = multiple_model.preprocess(base)

    end_time = time.time()
    model_intermediate['total_time'] = end_time - start_time
    return model_l, model_intermediate


def factory(config):
    _type = config['type']
    if _type == 'kmeans_independent':
        return preprocess_model.PreprocessKMeansIndependent(config)
    elif _type == 'kmeans_multiple':
        return preprocess_model.PreprocessKMeansMultiple(config)
    elif _type == 'e2lsh':
        return preprocess_model.PreprocessLSH(config)
    raise Exception('do not support the type of partition')
