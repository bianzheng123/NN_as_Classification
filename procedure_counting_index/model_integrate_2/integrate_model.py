import numpy as np
import os
import torch
from util import dir_io
from procedure_counting_index.util import io


def integrate(score_table_ptr_l, gnd, config):
    # long_term_config, short_term_config, short_term_config_before_run, intermediate_result, total_score_table
    os.system('mkdir %s' % config['program_result_dir'])
    # os.system('sudo mkdir %s' % config['program_result_dir'])
    recall_l = []
    iter_idx = 0
    while True:
        end_of_file = False
        # get the total recall for each query
        total_score_arr = None
        if iter_idx == 1000:
            break
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

    io.save_json(config['program_result_dir'], 'result.json', result_n_candidate_recall)

    recall_l_save_dir = '%s/recall_l.txt' % config['program_result_dir']
    # dir_io.save_file(recall_l_save_dir)
    np.savetxt(recall_l_save_dir, recall_l, fmt='%.3f')


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


def integrate_save_score_table(predict_cluster_l, label_map_l, config):
    total_score_table_dir = '%s/total_score_table.txt' % config['program_train_para_dir']
    # dir_io.save_file(total_score_table_dir)
    f_ptr = open(total_score_table_dir, "w")

    for i in range(predict_cluster_l[0].shape[0]):  # the length of query, means for every query
        result_score_single_query = np.zeros(shape=config['n_item'], dtype=np.int)
        for j in range(len(predict_cluster_l)):  # for every cluster
            pred_cluster = predict_cluster_l[j][i]
            for k in label_map_l[j][pred_cluster]:  # for every item in pred_cluster
                result_score_single_query[k] += 1
        str_line = ' '.join('%d' % score for score in result_score_single_query)
        f_ptr.write(str_line + '\n')

    f_ptr.close()

    print('save score table success')
