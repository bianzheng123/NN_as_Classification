from procedure_nn_classification.prepare_train_sample_2 import neighbor


def prepare_train(base, base_base_gnd, partition_info, config):
    save_dir = '%s/Classifier_%d_%d' % (
        config['program_train_para_dir'], config['entity_number'], config['classifier_number'])
    config['save_dir'] = save_dir
    trainset_ins = data_node_factory(config)
    # if need the learn data, add a new learn variable here in this function
    trainset_info = trainset_ins.prepare(base, base_base_gnd, partition_info)
    # trainset_ins.save()
    return trainset_info


def data_node_factory(config):
    _type = config['type']
    if _type == 'neighbor':
        return neighbor.NeighborDataNode(config)
    raise Exception('do not support the type of training data preparation')
