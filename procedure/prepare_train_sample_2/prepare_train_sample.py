from procedure.prepare_train_sample_2 import neighbor


def prepare_train(base, partition_info, config):
    save_dir = '%s/Classifier_%d_%d' % (
        config['program_train_para_dir'], config['entity_number'], config['classifier_number'])
    config['save_dir'] = save_dir
    trainset_ins = data_node_factory(config)
    # 将来如果需要使用learn数据, 这个方法就多加一个learn变量
    trainset_info = trainset_ins.prepare(base, partition_info)
    # trainset_ins.save()
    return trainset_info


def data_node_factory(config):
    _type = config['type']
    if _type == 'neighbor':
        return neighbor.NeighborDataNode(config)
    raise Exception('准备训练数据类型不支持')
