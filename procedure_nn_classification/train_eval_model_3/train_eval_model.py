from procedure_nn_classification.train_eval_model_3 import neural_network
import numpy as np
from util import dir_io
import time
import os
import multiprocessing
from multiprocessing.managers import BaseManager


def train_eval_model(base, query, trainset, config):
    save_dir = '%s/Classifier_%d' % (
        config['program_train_para_dir'], config['classifier_number'])
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
count the score_table for query
'''


def score_table_parallel(obj, start_idx):
    cluster_score_l, label_l, total_process, n_item = obj.get_share_data()
    score_table = {}
    # iteration for every classifier
    for i in range(start_idx, cluster_score_l[0].shape[0], total_process):
        if i % 50 == 0: print("get score table " + str(i))
        tmp_score_table = np.zeros(shape=n_item, dtype=np.float32)
        for k, tmp_cluster_score in enumerate(cluster_score_l, 0):
            tmp_label_map = label_l[k]
            # iteration for every cluster
            for j in range(cluster_score_l[0].shape[1]):
                score_item_idx_l = tmp_label_map[j]
                for _ in score_item_idx_l:
                    tmp_score_table[_] += tmp_cluster_score[i][j]
                score_table[i] = tmp_score_table
    print("finish parallel")
    return score_table


class IntegrateScoreTable:
    def __init__(self, cluster_score_l, label_l, n_item, total_process):
        self.score_table = np.zeros(shape=(cluster_score_l[0].shape[0], n_item), dtype=np.float32)
        self.cluster_score_l = cluster_score_l
        self.label_l = label_l
        self.total_process = total_process
        self.n_item = n_item

    def get_share_data(self):
        return self.cluster_score_l, self.label_l, self.total_process, self.n_item


'''
count the total score_table from all the score table in each classifier
'''


def integrate_save_score_table_parallel(cluster_score_l, label_l, config, save=False):
    n_process = 8
    n_pool_process = 8
    start_time = time.time()

    manager = BaseManager()
    manager.register('IntegrateScoreTable', IntegrateScoreTable)
    manager.start()
    parallel_obj = manager.IntegrateScoreTable(cluster_score_l, label_l, config['n_item'], n_process)
    res_l = []
    pool = multiprocessing.Pool(n_pool_process)
    for i in range(n_process):
        res = pool.apply_async(score_table_parallel, args=(parallel_obj, i))
        res_l.append(res)
    pool.close()
    pool.join()

    score_table = [0] * cluster_score_l[0].shape[0]
    for tmp_res in res_l:
        tmp_res = tmp_res.get()
        for idx in tmp_res:
            score_table[idx] = tmp_res[idx]

    score_table = np.array(score_table, dtype=np.float32)

    if save:
        total_score_table_dir = '%s/total_score_table.npy' % config['program_train_para_dir']
        dir_io.save_numpy(total_score_table_dir, score_table)
    end_time = time.time()
    intermediate = {
        'time': end_time - start_time
    }
    print('save score table success')
    return score_table, intermediate


'''
count the total score_table from all the score table in each classifier
'''


def integrate_save_score_table_total(cluster_score_l, label_l, config, save=False):
    start_time = time.time()
    shape = cluster_score_l[0].shape

    total_score_table_dir = '%s/total_score_table_total.txt' % config['program_train_para_dir']

    score_table = np.zeros(shape=(shape[0], config['n_item']), dtype=np.float32)
    # iteration for every query
    for i in range(shape[0]):
        # iterate for every classifier
        for k, tmp_cluster_score in enumerate(cluster_score_l, 0):
            label_map = label_l[k]
            # iterate for every cluster
            for j in range(shape[1]):
                score_item_idx_l = label_map[j]
                score_table[i][score_item_idx_l] += tmp_cluster_score[i][j]

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


def integrate_save_score_table_single(cluster_score_l, label_l, config):
    start_time = time.time()
    shape = cluster_score_l[0].shape

    total_score_table_dir = '%s/total_score_table.txt' % config['program_train_para_dir']

    f_ptr = dir_io.write_ptr(total_score_table_dir)

    for i in range(shape[0]):
        result_score_single_query = np.zeros(shape=config['n_item'], dtype=np.float32)
        for k, tmp_cluster_score in enumerate(cluster_score_l, 0):
            label_map = label_l[k]
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
