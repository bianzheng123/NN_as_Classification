from procedure.dataset_partition_1.kmeans import multiple_kmeans
from procedure.dataset_partition_1.learn_on_graph import multiple_learn_on_graph
import os

'''
输入base, 输出partition的文本信息
'''


def partition(base, config):
    os.system("mkdir %s" % config['program_train_para_dir'])
    dataset_partition_config = config['dataset_partition']
    mutual_attribute_config = config['mutual_attribute']
    partition_result_l = []
    for i in range(config['n_config_entity']):
        tmp_config = dataset_partition_config[i]
        tmp_config['program_train_para_dir'] = config['program_train_para_dir']
        tmp_config['n_cluster'] = mutual_attribute_config[i]['n_cluster']
        tmp_config['n_instance'] = mutual_attribute_config[i]['n_instance']
        tmp_config['kahip_dir'] = config['kahip_dir']
        # 让partition_ins知道自己是几号, 多个模型需要分别建立文件夹
        tmp_config['entity_number'] = i + 1
        partition_ins = partition_factory(tmp_config)
        partition_info_l = partition_ins.partition(base)
        partition_ins.save()
        for info in partition_info_l:
            partition_result_l.append(info)
    return partition_result_l


def partition_factory(config):
    _type = config['type']
    if _type == 'kmeans':
        return multiple_kmeans.MultipleKMeans(config)
    elif _type == 'learn_on_graph':
        return multiple_learn_on_graph.MultipleLearnOnGraph(config)
    raise Exception('partition类型不支持')
