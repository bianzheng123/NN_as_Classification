from procedure.train_eval_model_3 import neural_network


def train_eval_model(base, query, label_map, trainset, config):
    save_dir = '%s/Classifier_%d_%d' % (
        config['program_train_para_dir'], config['entity_number'], config['classifier_number'])
    config['save_dir'] = save_dir
    train_model_ins = train_model_factory(config)
    # 将来如果需要使用learn数据, 这个方法就多加一个learn变量
    train_model_ins.train(base, trainset)
    eval_result, intermediate_config = train_model_ins.eval(query, label_map)
    train_model_ins.save()
    return eval_result, intermediate_config


def train_model_factory(config):
    _type = config['type']
    if _type == 'neural_network':
        return neural_network.NeuralNetwork(config)
    raise Exception('准备训练数据类型不支持')
