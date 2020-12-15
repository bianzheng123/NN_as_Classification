import numpy as np
from util import dir_io

'''
input base and output the text information of partition
'''


def partition(base, model):
    partition_info, model_info = model.partition(base)
    # model.save()
    return partition_info, model_info


def integrate_save_score_table(predict_cluster_l, label_map_l, config):
    total_score_table_dir = '%s/total_score_table.txt' % config['program_train_para_dir']

    f_ptr = dir_io.write_ptr(total_score_table_dir)

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
