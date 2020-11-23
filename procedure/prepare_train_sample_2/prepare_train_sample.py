from procedure.prepare_train_sample_2 import neighbor


def prepare_train(base, partition_info_l, config):
    prepare_train_config = config['prepare_train']
    mutual_attribute_config = config['mutual_attribute']
    trainset_result_l = []
    iter_idx = 0
    for i in range(config['n_config_entity']):
        for j in range(mutual_attribute_config[i]['n_instance']):
            save_dir = '%s/Classifier_%d_%d' % (config['program_train_para_dir'], i + 1, j + 1)
            tmp_config = prepare_train_config[i]
            tmp_config['save_dir'] = save_dir
            tmp_config['n_cluster'] = mutual_attribute_config[i]['n_cluster']
            tmp_config['entity_number'] = i + 1
            tmp_config['classifier_number'] = j + 1
            trainset_ins = data_node_factory(tmp_config)
            # 将来如果需要使用learn数据, 这个方法就多加一个learn变量
            trainset_info = trainset_ins.prepare(base, partition_info_l[iter_idx])
            trainset_ins.save()
            trainset_result_l.append(trainset_info)
            iter_idx += 1
    return trainset_result_l


def data_node_factory(config):
    _type = config['type']
    if _type == 'neighbor':
        return neighbor.NeighborDataNode(config)
    raise Exception('准备训练数据类型不支持')
