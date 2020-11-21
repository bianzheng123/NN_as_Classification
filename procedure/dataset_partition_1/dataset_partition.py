from procedure. dataset_partition_1 import kmeans
'''
输入base, 输出partition的文本信息
'''


def partition(base, config):
    dataset_partition_config = config['dataset_partition']
    partition_result_l = []
    for i in range(config['n_classifier']):
        save_dir = '%s/Classifier_%d' % (config['program_train_para_dir'], i + 1)
        tmp_config = dataset_partition_config[i]
        tmp_config['save_dir'] = save_dir
        # 让partition_ins知道自己是几号
        tmp_config['classifier_number'] = i + 1
        partition_ins = partition_factory(tmp_config)
        partition_info = partition_ins.partition(base)
        partition_result_l.append(partition_info)
    return partition_result_l


def partition_factory(config):
    _type = config['type']
    if _type == 'kmeans':
        return kmeans.KMeans(config)
    raise Exception('partition类型不支持')
