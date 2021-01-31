import numpy as np
from util.vecs import vecs_io
from procedure_counting_index.dataset_partition_1 import partition_data
from procedure_counting_index.init_0 import partition_preprocess
from procedure_common import result_integrate
import time
import json
from util import dir_io
import math


def run(long_term_config_dir, short_term_config_dir):
    np.random.seed(123)
    with open(long_term_config_dir, 'r') as f:
        long_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config_before_run = json.load(f)

    save_train_para = False
    total_start_time = time.time()

    data_dir = '%s/data/dataset/%s_%d' % (
        long_term_config['project_dir'], long_term_config['data_fname'], long_term_config['k'])
    load_data_config = {
        'data_dir': data_dir
    }
    # load data
    data = vecs_io.read_data_l2(load_data_config)
    base = data[0]
    query = data[1]
    gnd = data[2]

    program_fname = '%s_%d_count_%d_%s_%s' % (
        long_term_config['data_fname'], short_term_config['n_cluster'], short_term_config['n_instance'],
        short_term_config['dataset_partition']['type'], short_term_config['specific_fname'])
    if short_term_config['dataset_partition']['type'] == 'e2lsh':
        program_fname = '%s_%d_count_%d_%s_%s' % (
            long_term_config['data_fname'], short_term_config['n_cluster'] * short_term_config['n_cluster'],
            short_term_config['n_instance'] // 2,
            short_term_config['dataset_partition']['type'], short_term_config['specific_fname'])

    # classification
    program_train_para_dir = '%s/data/train_para/%s' % (
        long_term_config['project_dir'], program_fname)
    program_result_dir = '%s/data/result/%s' % (
        long_term_config['project_dir'], program_fname)

    dir_io.delete_dir_if_exist(program_train_para_dir)
    dir_io.delete_dir_if_exist(program_result_dir)

    partition_preprocess_config = {
        "dataset_partition": short_term_config['dataset_partition'],
        'n_cluster': short_term_config['n_cluster'],
        'n_instance': short_term_config['n_instance'],
        "program_train_para_dir": program_train_para_dir,
    }
    model_l, preprocess_intermediate = partition_preprocess.preprocess(base, partition_preprocess_config)

    predict_cluster_l = []
    intermediate_result_l = []
    label_map_l = []
    for model in model_l:
        # partition_info = (labels, label_map)
        partition_info, model_info = partition_data.partition(base, model)
        partition_intermediate = model_info[1]

        # predict all the query
        pred_cluster, predict_intermediate = model.predict(query)

        intermediate = {
            "ins_id": model_info[0],
            'dataset_partition': partition_intermediate,
            'predict': predict_intermediate,
        }
        intermediate_result_l.append(intermediate)
        predict_cluster_l.append(pred_cluster)
        label_map = partition_info[1]
        label_map_l.append(label_map)

    save_classifier_config = {
        'program_train_para_dir': program_train_para_dir,
        'n_item': base.shape[0]
    }
    # integrate the cluster_score_l and label_map_l to get the score_table and store the score_table in /train_para
    score_table, integrate_intermediate = partition_data.integrate_save_score_table_total(predict_cluster_l,
                                                                                          label_map_l,
                                                                                          save_classifier_config,
                                                                                          save_train_para)

    result_integrate_config = {
        'k': long_term_config['k'],
        'program_result_dir': program_result_dir,
        'efSearch_l': [2 ** i for i in range(1 + int(math.log2(len(base))))],
    }
    count_recall_intermediate = result_integrate.integrate_single(score_table, gnd, result_integrate_config)
    total_end_time = time.time()

    intermediate_result_final = {
        'total_time_consume': total_end_time - total_start_time,
        'preprocess': preprocess_intermediate,
        'classifier': intermediate_result_l,
        'integrate': integrate_intermediate,
        'count_recall': count_recall_intermediate
    }

    # save intermediate and configure file
    if save_train_para:
        save_config_config = {
            'long_term_config': long_term_config,
            'short_term_config': short_term_config,
            'short_term_config_before_run': short_term_config_before_run,
            'intermediate_result': intermediate_result_final,
            'save_dir': program_train_para_dir,
            'program_fname': program_fname
        }
        dir_io.save_config(save_config_config)

    save_config_config = {
        'long_term_config': long_term_config,
        'short_term_config': short_term_config,
        'short_term_config_before_run': short_term_config_before_run,
        'intermediate_result': intermediate_result_final,
        'save_dir': program_result_dir,
        'program_fname': program_fname
    }
    dir_io.save_config(save_config_config)
