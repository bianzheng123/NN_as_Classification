from procedure_nn_classification.train_eval_model_3 import neural_network
import numpy as np
from util import dir_io
import time


def train_eval_model(base, query, trainset, config):
    save_dir = '%s/Classifier_%d_%d' % (
        config['program_train_para_dir'], config['entity_number'], config['classifier_number'])
    config['save_dir'] = save_dir
    train_model_ins = train_model_factory(config)
    # if use the learn dataset, add the learn variable in here
    train_model_ins.train(base, trainset)
    eval_result, intermediate_config = train_model_ins.eval(query)
    # train_model_ins.save()
    return eval_result, intermediate_config


def train_model_factory(config):
    _type = config['type']
    if _type == 'neural_network':
        return neural_network.NeuralNetwork(config)
    raise Exception('do not support the type of training data')


'''
count the total score_table from all the score table in each classifier
'''


def integrate_save_score_table_total(cluster_score_l, label_map_l, config, save=False):
    start_time = time.time()
    shape = cluster_score_l[0].shape

    total_score_table_dir = '%s/total_score_table.txt' % config['program_train_para_dir']

    score_table = np.zeros(shape=(shape[0], config['n_item']), dtype=np.float32)
    for i in range(shape[0]):
        for k, tmp_cluster_score in enumerate(cluster_score_l, 0):
            label_map = label_map_l[k]
            for j in range(shape[1]):
                score_item_idx_l = label_map[j]
                score_table[i][score_item_idx_l] += tmp_cluster_score[i][j].item()

    if save:
        dir_io.save_array_txt(total_score_table_dir, score_table, '%.3f')
    end_time = time.time()
    intermediate = {
        'time': end_time - start_time
    }
    print('save score table success')
    return score_table, intermediate


'''
count the total score_table from all the score table in each classifier
'''


def integrate_save_score_table_single(cluster_score_l, label_map_l, config):
    start_time = time.time()
    shape = cluster_score_l[0].shape

    total_score_table_dir = '%s/total_score_table.txt' % config['program_train_para_dir']

    f_ptr = dir_io.write_ptr(total_score_table_dir)

    for i in range(shape[0]):
        result_score_single_query = np.zeros(shape=config['n_item'], dtype=np.float32)
        for k, tmp_cluster_score in enumerate(cluster_score_l, 0):
            label_map = label_map_l[k]
            for j in range(shape[1]):
                score_item_idx_l = label_map[j]
                result_score_single_query[score_item_idx_l] += tmp_cluster_score[i][j].item()
        str_line = ' '.join('%.3f' % score for score in result_score_single_query)
        f_ptr.write(str_line + '\n')

    f_ptr.close()
    end_time = time.time()
    intermediate = {
        'time': end_time - start_time
    }
    print('save score table success')
    return intermediate
