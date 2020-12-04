from procedure.train_eval_model_3 import neural_network
from procedure.result_integrate_4 import result_integrate
import numpy as np


def train_eval_model(base, query, label_map, trainset, config):
    save_dir = '%s/Classifier_%d_%d' % (
        config['program_train_para_dir'], config['entity_number'], config['classifier_number'])
    config['save_dir'] = save_dir
    train_model_ins = train_model_factory(config)
    # 将来如果需要使用learn数据, 这个方法就多加一个learn变量
    train_model_ins.train(base, trainset)
    eval_result, intermediate_config = train_model_ins.eval(query, label_map)
    # train_model_ins.save()
    return eval_result, intermediate_config


'''
统计出总的score_table
'''


def save_score_table(score_table_l, config):
    result_score_table = np.zeros(score_table_l[0].shape)
    shape = score_table_l[0].shape
    for tmp_score_table in score_table_l:
        for i in range(shape[0]):
            for j in range(shape[1]):
                result_score_table[i][j] += tmp_score_table[i][j]
    # 保存
    total_score_table_dir = '%s/total_score_table.txt' % config['program_train_para_dir']
    np.savetxt(total_score_table_dir, result_score_table, fmt='%.3f')
    print('save score table success')


def train_model_factory(config):
    _type = config['type']
    if _type == 'neural_network':
        return neural_network.NeuralNetwork(config)
    raise Exception('准备训练数据类型不支持')
