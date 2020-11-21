from procedure.prepare_train_sample_2 import neighbor


def prepare_train(base, partition_info_l, config):
    prepare_train_config = config['prepare_train']
    trainset_result_l = []
    for i in range(config['n_classifier']):
        save_dir = '%s/Classifier_%d' % (config['program_train_para_dir'], i + 1)
        tmp_config = prepare_train_config[i]
        tmp_config['save_dir'] = save_dir
        tmp_config['n_cluster'] = config['n_cluster_l'][i]
        tmp_config['classifier_number'] = i + 1
        trainset_ins = data_node_factory(tmp_config)
        # 将来如果需要使用learn数据, 这个方法就多加一个learn变量
        trainset_info = trainset_ins.prepare(base, partition_info_l[i])
        trainset_ins.save()
        trainset_result_l.append(trainset_info)
    return trainset_result_l


def data_node_factory(config):
    _type = config['type']
    if _type == 'neighbor':
        return neighbor.NeighborDataNode(config)
    raise Exception('准备训练数据类型不支持')
