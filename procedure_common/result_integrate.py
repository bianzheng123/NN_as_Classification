import torch
import numpy as np
import json
from util import dir_io


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


'''
score_table: the score_table for a single query
'''


def evaluate(score_table, efSearch_l, gnd, k):
    # get recall in different efSearch
    recall_l = []
    score_table = torch.from_numpy(score_table)

    total_candidate_index = torch.topk(score_table, dim=0, largest=True, k=efSearch_l[-1])[1]
    for efSearch in efSearch_l:
        candidate_index = total_candidate_index[:efSearch]
        # count recall for every single query
        efsearch_recall = count_recall_single(candidate_index, gnd, k)
        recall_l.append(efsearch_recall)

    return recall_l


def count_recall_single(predict_idx, gnd, k):
    gnd = gnd[:k]
    predict_idx = predict_idx.numpy()
    matches = [index for index in predict_idx if index in gnd]
    recall = float(len(matches)) / k
    return recall
