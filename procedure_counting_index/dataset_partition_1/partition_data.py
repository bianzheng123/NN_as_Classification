import numpy as np
from util import dir_io
import multiprocessing
import time

'''
input base and output the text information of partition
'''


def partition(base, model):
    partition_info, model_info = model.partition(base)
    # model.save()
    return partition_info, model_info


def integrate_save_score_table_total(predict_cluster_l, label_map_l, config, save):
    start_time = time.time()

    query_len = predict_cluster_l[0].shape[0]
    score_table = np.zeros(shape=(query_len, config['n_item']), dtype=np.int)
    for i in range(query_len):  # the length of query, means for every query
        for j in range(len(predict_cluster_l)):  # for every cluster
            pred_cluster = predict_cluster_l[j][i]
            for k in label_map_l[j][pred_cluster]:  # for every item in pred_cluster
                score_table[i][k] += 1

    total_score_table_dir = '%s/total_score_table.txt' % config['program_train_para_dir']
    if save:
        dir_io.save_array_txt(total_score_table_dir, score_table, '%d')

    end_time = time.time()
    intermediate = {
        'time': end_time - start_time
    }
    print('save score table success')
    return score_table, intermediate


def get_score_table_single_query(x):
    predict_cluster, label_map_l, n_item = x
    score_table = np.zeros(shape=n_item, dtype=np.int)
    for i, cls_idx in enumerate(predict_cluster, 0):
        for k in label_map_l[i][cls_idx]:  # for every item in pred_cluster
            score_table[k] += 1
    return score_table


def integrate_save_score_table_total_parallel(predict_cluster_l, label_map_l, config, save):
    start_time = time.time()

    query_len = predict_cluster_l[0].shape[0]
    predict_cluster_l = np.array(predict_cluster_l)
    # 2 dimension array, first is the query index, second is the predicted cluster index
    query_predict_cluster = predict_cluster_l.transpose()

    with multiprocessing.Pool(multiprocessing.cpu_count() // 10 * 9) as pool:
        score_table = list(
            pool.imap(get_score_table_single_query, zip(query_predict_cluster, [label_map_l for _ in range(query_len)],
                                                        [config['n_item'] for _ in range(query_len)]))
        )

    total_score_table_dir = '%s/total_score_table.txt' % config['program_train_para_dir']
    if save:
        dir_io.save_array_txt(total_score_table_dir, score_table, '%d')

    end_time = time.time()
    intermediate = {
        'time': end_time - start_time
    }
    print('save score table success')
    return score_table, intermediate


def f(x):
    a, B = x
    return [Levenshtein.distance(a, b) for b in B]


def all_pair_distance(A, B, n_thread, progress=True):
    bar = tqdm if progress else lambda iterable, total, desc: iterable

    def all_pair(A, B, n_thread):
        with Pool(n_thread) as pool:
            start_time = time.time()
            edit = list(
                bar(
                    pool.imap(f, zip(A, [B for _ in A])),
                    total=len(A),
                    desc="# edit distance {}x{}".format(len(A), len(B)),
                ))
            if progress:
                print("# Calculate edit distance time: {}".format(time.time() - start_time))
            return np.array(edit)

    if len(A) < len(B):
        return all_pair(B, A, n_thread).T
    else:
        return all_pair(A, B, n_thread)
