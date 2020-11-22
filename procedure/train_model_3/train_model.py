from procedure.train_model_3 import neural_network


def train_model(base, trainset_l, config):
    train_model_config = config['train_model']
    mutual_attribute_config = config['mutual_attribute']
    train_model_ins_l = []
    for i in range(config['n_classifier']):
        save_dir = '%s/Classifier_%d' % (config['program_train_para_dir'], i + 1)
        tmp_config = train_model_config[i]
        tmp_config['save_dir'] = save_dir
        tmp_config['n_cluster'] = mutual_attribute_config[i]['n_cluster']
        tmp_config['classifier_number'] = i + 1
        train_model_ins = train_model_factory(tmp_config)
        # 将来如果需要使用learn数据, 这个方法就多加一个learn变量
        train_model_ins.train(base, trainset_l[i])
        train_model_ins.save()
        train_model_ins_l.append(train_model_ins)
    return train_model_ins_l


def train_model_factory(config):
    _type = config['type']
    if _type == 'neural_network':
        return neural_network.NeuralNetwork(config)
    raise Exception('准备训练数据类型不支持')
