import torch
import numpy as np
import json
import os


def integrate(score_table_l, gnd, config):
    # long_term_config, short_term_config, short_term_config_before_run, intermediate_result, total_score_table
    os.system('mkdir %s' % config['program_result_dir'])
    total_score_table = np.zeros(score_table_l[0].shape)
    for tmp_score_table in score_table_l:
        for i in range(len(tmp_score_table)):
            for j in range(len(tmp_score_table[i])):
                total_score_table[i][j] += tmp_score_table[i][j]
    print('add total score complete')
    evaluate(total_score_table, config['efSearch_l'], gnd, config['program_result_dir'], config['k'])


def evaluate(score_table, efSearch_l, gnd, save_json_dir, k):
    recall_l = []
    score_table = torch.from_numpy(score_table)
    for i, efSearch in enumerate(efSearch_l, 0):
        largest_table_number_l, candidate_index = torch.topk(score_table, dim=1, largest=True, k=efSearch)
        # 计算每一个query的recall
        k_recall_l = count_recall_multiple(candidate_index, gnd, k)
        recall_l.append(k_recall_l)

    recall_l = np.array(recall_l)
    result_n_candidate_recall = []
    for i, efSearch in enumerate(efSearch_l, 0):
        recall_avg = np.mean(recall_l[i])
        result_item = {
            'n_candidate': efSearch,
            "recall": recall_avg
        }
        result_n_candidate_recall.append(result_item)
        print('recall: {}, n_candidates: {}'.format(recall_avg, efSearch))

    save_json(save_json_dir, 'result.json', result_n_candidate_recall)


'''
计算多个recall
'''


def count_recall_multiple(predict_idx_l, gnd_l, k):
    gnd_l = gnd_l[:, :k]
    recall_l = []
    for i in range(len(gnd_l)):
        predict_idx = predict_idx_l[i].numpy()
        gnd = gnd_l[i]
        matches = [index for index in predict_idx if index in gnd]
        recall = float(len(matches)) / k
        recall_l.append(recall)
    return recall_l


def save_json(save_dir, result_fname, json_file):
    with open('%s/%s' % (save_dir, result_fname), 'w') as f:
        json.dump(json_file, f)


def save_config(config):
    save_dir = '%s/config' % config['save_dir']
    os.system('mkdir %s' % save_dir)

    long_config = config['long_term_config']
    short_config = config['short_term_config']
    short_config_before_run = config['short_term_config_before_run']
    intermediate_result = config['intermediate_result']

    save_json(save_dir, 'long_term_config.json', long_config)
    save_json(save_dir, 'short_term_config.json', short_config)
    save_json(save_dir, 'short_term_config_before_run.json', short_config_before_run)
    save_json(config['save_dir'], 'intermediate_result.json', intermediate_result)
    print('save program: %s' % config['program_fname'])
