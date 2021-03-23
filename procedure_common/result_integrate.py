import torch
import numpy as np
import json
from util import dir_io
import time
import numba as nb


def integrate(score_table_ptr_l, gnd, config):
    # long_term_config, short_term_config, short_term_config_before_run, intermediate_result, total_score_table
    dir_io.mkdir(config['program_result_dir'])
    recall_l = []
    iter_idx = 0
    while True:
        end_of_file = False
        # get the total recall for each query
        total_score_arr = None
        for score_table_ptr in score_table_ptr_l:
            line = score_table_ptr.readline()
            if not line or line == '':
                end_of_file = True
                break
            tmp_score_table = np.array([float(number) for number in line.split(' ')])
            if total_score_arr is None:
                total_score_arr = tmp_score_table
            else:
                total_score_arr += tmp_score_table
        if end_of_file:
            break

        efsearch_recall_l = evaluate(total_score_arr, config['efSearch_l'], gnd[iter_idx], config['k'])
        recall_l.append(efsearch_recall_l)

        iter_idx += 1
    print('get all the recall')
    # transpose makes the same efsearch in every row of recall
    recall_l = np.array(recall_l).transpose()

    result_n_candidate_recall = []
    for i, efSearch in enumerate(config['efSearch_l'], 0):
        recall_avg = np.mean(recall_l[i])
        result_item = {
            'n_candidate': efSearch,
            "recall": recall_avg
        }
        result_n_candidate_recall.append(result_item)
        print('recall: {}, n_candidates: {}'.format(recall_avg, efSearch))

    dir_io.save_json(config['program_result_dir'], 'result.json', result_n_candidate_recall)

    recall_l_save_dir = '%s/recall_l.txt' % config['program_result_dir']
    dir_io.save_array_txt(recall_l_save_dir, recall_l, '%.3f')


def integrate_single(score_table, gnd, config):
    torch.set_num_threads(12)
    start_time = time.time()
    # long_term_config, short_term_config, short_term_config_before_run, intermediate_result, total_score_table
    dir_io.mkdir(config['program_result_dir'])
    recall_l = []
    print("start evaluate")
    for i, score_arr in enumerate(score_table, 0):
        if i % 50 == 0: print("evaluate " + str(i))
        efsearch_recall_l = evaluate(score_arr, config['efSearch_l'], gnd[i], config['k'])
        recall_l.append(efsearch_recall_l)
    print('get all the recall')
    # transpose makes the same efsearch in every row of recall
    recall_l = np.array(recall_l).transpose()

    result_n_candidate_recall = []
    for i, efSearch in enumerate(config['efSearch_l'], 0):
        recall_avg = np.mean(recall_l[i])
        result_item = {
            'n_candidate': efSearch,
            "recall": recall_avg
        }
        result_n_candidate_recall.append(result_item)
        print('recall: {}, n_candidates: {}'.format(recall_avg, efSearch))

    dir_io.save_json(config['program_result_dir'], 'result.json', result_n_candidate_recall)

    recall_l_save_dir = '%s/recall_l.txt' % config['program_result_dir']
    dir_io.save_array_txt(recall_l_save_dir, recall_l, '%.3f')
    end_time = time.time()
    intermediate = {
        'time': end_time - start_time
    }
    return intermediate


'''
evaluate for a single query
'''


def evaluate(score_table, efSearch_l, gnd, k):
    # get recall in different efSearch
    recall_l = []
    arg_idx = score_table.argsort(kind='stable')
    total_candidate_index = arg_idx[-efSearch_l[-1]:].tolist()
    total_candidate_index.reverse()
    for efSearch in efSearch_l:
        candidate_index = total_candidate_index[:efSearch]
        # count recall for every single query
        gnd_set = set(gnd)
        efsearch_recall = count_recall_single(candidate_index, gnd_set, k)
        recall_l.append(efsearch_recall)
    # print("end count recall")
    return recall_l


def count_recall_single(predict_idx, gnd, k):
    matches = [index for index in predict_idx if index in gnd]
    recall = float(len(matches)) / k
    return recall
