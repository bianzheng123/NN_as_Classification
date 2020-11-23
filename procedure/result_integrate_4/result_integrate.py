import torch
import numpy as np
import json
import os


def integrate(query, gnd, train_model_ins_l, partition_info_l, config):
    os.system('mkdir %s' % config['program_result_dir'])
    result_integrate_config = config['result_integrate']
    mutual_attribute_config = config['mutual_attribute']
    score_table = np.zeros((query.shape[0], config['n_item']))
    for i in range(config['n_total_classifier']):
        label_map = partition_info_l[i][1]
        distribution = train_model_ins_l[i].eval(query)
        # 找到预测分布中最大的cluster
        score_l, large_idx = torch.topk(distribution, dim=1, largest=True, k=1)
        # 对每一个query加分
        for j in range(query.shape[0]):
            score_partition_idx = large_idx[j][0].item()
            score_item_idx_l = label_map[score_partition_idx]
            for item_idx in score_item_idx_l:
                score_table[j][score_item_idx_l] += score_l[j][0].item()

    evaluate(score_table, config['efSearch_l'], gnd, config['program_result_dir'], config['k'])


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
    print('save fname: %s' % result_fname)
    with open('%s/%s' % (save_dir, result_fname), 'w') as f:
        json.dump(json_file, f)
