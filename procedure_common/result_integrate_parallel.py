import torch
import numpy as np
import json
from util import dir_io
import time
import numba as nb
import multiprocessing
from multiprocessing.managers import BaseManager


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


class IntegrateResult:
    def __init__(self, efSearch_l, gnd, k, n_process):
        self.efSearch_l = efSearch_l
        self.gnd = gnd
        self.k = k
        self.n_process = n_process
        print("success")
        self.recall_l = np.empty(shape=(score_table.shape[0], len(efSearch_l)), dtype=np.float32)

    def evaluate(self, start_idx):
        n_query = self.score_table.shape[0]
        for i in range(start_idx, n_query, self.n_process):
            efsearch_recall_l = evaluate(self.score_table[i], self.efSearch_l, self.gnd[i], self.k)
            self.recall_l[i, :] = np.array(efsearch_recall_l)

    def get_recall_l(self):
        return self.recall_l


def evaluate_parallel(score_table, parallel_obj, start_idx):
    parallel_obj.evaluate(start_idx)
    print("end parallel evaluate")


def integrate_single(score_table, gnd, config):
    start_time = time.time()
    # long_term_config, short_term_config, short_term_config_before_run, intermediate_result, total_score_table
    dir_io.mkdir(config['program_result_dir'])
    print("start recall evaluate")
    n_process = 4
    n_pool_process = 4

    manager = BaseManager()
    manager.register('IntegrateResult', IntegrateResult)
    manager.start()
    print(config['efSearch_l'])
    print(score_table.dtype)
    '''
    解决方案: 对score_table分成几份, 然后进行并行
    '''
    score_table = score_table.reshape(-1)
    score_table = multiprocessing.Array('f', score_table)
    print("share arr")
    parallel_obj = manager.IntegrateResult(config['efSearch_l'], gnd, config['k'], n_process)
    pool = multiprocessing.Pool(n_pool_process)
    for i in range(n_process):
        pool.apply_async(evaluate_parallel, args=(score_table, parallel_obj, i))
    pool.close()
    pool.join()

    recall_l = parallel_obj.get_recall_l()

    print('finish recall evaluate')
    # transpose makes the same efsearch in every row of recall
    recall_l = np.array(recall_l).transpose()

    result_n_candidate_recall = []
    for i, efSearch in enumerate(config['efSearch_l'], 0):
        recall_avg = np.mean(recall_l[i])
        recall_avg = '{:.3f}'.format(recall_avg)
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


@nb.jit(nopython=True)
def resort_indices(val, total_candidate_index, score_table):
    # sort according to the number of indices
    start_same_ptr = 0
    end_same_ptr = 1
    len_val = val.shape[-1]
    while True:
        if val[end_same_ptr] == val[start_same_ptr]:
            end_same_ptr += 1
            if end_same_ptr == len_val:
                n_same = end_same_ptr - start_same_ptr
                for i, score in enumerate(score_table, 0):
                    if score == val[start_same_ptr]:
                        total_candidate_index[start_same_ptr] = i
                        start_same_ptr += 1
                        n_same -= 1
                        if n_same == 0:
                            break
                break
        else:
            n_same = end_same_ptr - start_same_ptr
            for i, score in enumerate(score_table, 0):
                if score == val[start_same_ptr]:
                    total_candidate_index[start_same_ptr] = i
                    start_same_ptr += 1
                    n_same -= 1
                    if n_same == 0:
                        break
            start_same_ptr = end_same_ptr
            end_same_ptr += 1
    return total_candidate_index


'''
score_table: the score_table for a single query
'''


def evaluate(score_table, efSearch_l, gnd, k):
    # get recall in different efSearch
    recall_l = np.empty(shape=(len(efSearch_l)), dtype=np.float32)
    # recall_l = []
    val, total_candidate_index = torch.topk(torch.from_numpy(score_table), dim=0, largest=True, k=efSearch_l[-1])

    total_candidate_index = resort_indices(val.numpy(), total_candidate_index.numpy(), score_table)
    # print(val, total_candidate_index)
    for i, efSearch in enumerate(efSearch_l, 0):
        candidate_index = total_candidate_index[:efSearch]
        # count recall for every single query
        efsearch_recall = count_recall_single(candidate_index, gnd, k)
        recall_l[i] = efsearch_recall
        # recall_l.append(efsearch_recall)
    return recall_l


def count_recall_single(predict_idx, gnd, k):
    gnd = gnd[:k]
    # predict_idx = predict_idx.numpy()
    matches = [index for index in predict_idx if index in gnd]
    recall = float(len(matches)) / k
    return recall
