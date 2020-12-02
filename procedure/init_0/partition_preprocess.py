from procedure.init_0.kmeans import multiple_kmeans_batch, independent_kmeans, multiple_kmeans_self_impl
from procedure.init_0.learn_on_graph import multiple_learn_on_graph
import os


def preprocess(base, config):
    program_train_para_dir = config['program_train_para_dir']
    os.system("mkdir %s" % program_train_para_dir)
    n_cluster = config['n_cluster']
    kahip_dir = config['kahip_dir']
    partition_model_l = []
    for entity_number, tmp_config in enumerate(config['independent_config'], 1):
        tmp_config['program_train_para_dir'] = program_train_para_dir
        tmp_config['n_cluster'] = n_cluster
        tmp_config['kahip_dir'] = kahip_dir
        tmp_config['entity_number'] = entity_number
        multiple_model = factory(tmp_config)
        tmp_model_l = multiple_model.preprocess(base)
        for model in tmp_model_l:
            partition_model_l.append(model)
    print("finish preprocess")

    return partition_model_l


def factory(config):
    _type = config['type']
    if _type == 'kmeans':
        _specific_type = config['specific_type']
        # multiple_kmeans_batch, independent_kmeans, multiple_kmeans_self_impl
        if _specific_type == 'multiple_batch':
            return multiple_kmeans_batch.MultipleKMeansBatch(config)
        elif _specific_type == 'independent':
            return independent_kmeans.IndependentKMeans(config)
        elif _specific_type == 'multiple_self_impl':
            return multiple_kmeans_self_impl.MultipleKMeansSelfImpl(config)
    elif _type == 'learn_on_graph':
        return multiple_learn_on_graph.MultipleLearnOnGraph(config)
    raise Exception('partition类型不支持')
