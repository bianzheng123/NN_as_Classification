from util.vecs import vecs_io
from procedure_nn_classification.init_0 import preprocess
from procedure_nn_classification.dataset_partition_1 import base_partition
from procedure_nn_classification import prepare_train_sample_2
from procedure_nn_classification.train_eval_model_3 import train_eval_model
from procedure_common import result_integrate
import json
import numpy as np
import time
from util import dir_io
import math


def run(long_term_config_dir, short_term_config_dir, topk):
    np.random.seed(123)
    with open(long_term_config_dir, 'r') as f:
        long_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config_before_run = json.load(f)

    save_train_para = True
    total_start_time = time.time()

    # load data
    data_dir = '%s/data/dataset/%s' % (
        long_term_config['project_dir'], long_term_config['data_fname'])
    load_data_config = {
        'data_dir': data_dir
    }
    if long_term_config['distance_metric'] == 'l2':
        base, query, gnd, base_base_gnd = vecs_io.read_data_l2(load_data_config)
    elif long_term_config['distance_metric'] == 'string':
        base, query, gnd, base_base_gnd = vecs_io.read_data_string(load_data_config)
    gnd = gnd[:, :topk]
    print(gnd.shape)
    del load_data_config

    program_fname = '%s_%d_%d_nn_%d_%s_%s' % (
        long_term_config['data_fname'], topk, short_term_config['n_cluster'],
        short_term_config['n_instance'],
        short_term_config['dataset_partition']['type'], short_term_config['specific_fname'])

    program_train_para_dir = '%s/data/train_para/%s' % (
        long_term_config['project_dir'], program_fname)
    program_result_dir = '%s/data/result/%s' % (long_term_config['project_dir'], program_fname)

    dir_io.delete_dir_if_exist(program_train_para_dir)
    dir_io.delete_dir_if_exist(program_result_dir)

    partition_preprocess_config = {
        "dataset_partition": short_term_config['dataset_partition'],
        'n_cluster': short_term_config['n_cluster'],
        'n_instance': short_term_config['n_instance'],
        'kahip_dir': long_term_config['kahip_dir'],
        "program_train_para_dir": program_train_para_dir,
        'distance_metric': long_term_config['distance_metric']
    }
    # get the initial classifier object for each method
    # initial the centroid for joint model e.g. kmeans multiple
    model_l, preprocess_intermediate = preprocess.preprocess(base, partition_preprocess_config)
    del partition_preprocess_config

    cluster_score_l = []
    intermediate_result_l = []
    label_l = []
    dataset_partition_para = None
    for model in model_l:
        partition_info, model_info, dataset_partition_para = base_partition.partition(base, model, base_base_gnd,
                                                                                      dataset_partition_para)
        partition_intermediate = model_info[1]

        prepare_train_config = short_term_config['prepare_train_sample']
        prepare_train_config['n_cluster'] = short_term_config['n_cluster']
        prepare_train_config['program_train_para_dir'] = program_train_para_dir
        prepare_train_config['classifier_number'] = model_info[0]
        # to prepare for the training set and evaluating set
        trainset, prepare_train_intermediate = prepare_train_sample_2.prepare_train(base, base_base_gnd, partition_info,
                                                                                    prepare_train_config)

        train_model_config = short_term_config['train_model']
        train_model_config['n_cluster'] = short_term_config['n_cluster']
        train_model_config['classifier_number'] = model_info[0]
        train_model_config['program_train_para_dir'] = program_train_para_dir
        train_model_config['distance_metric'] = long_term_config['distance_metric']
        train_model_config['data_fname'] = long_term_config['data_fname']
        train_model_config['n_input'] = long_term_config['dimension']
        train_model_config['dataset_partition_method'] = short_term_config["dataset_partition"]['type']
        if 'n_character' in long_term_config:
            train_model_config['n_character'] = long_term_config['n_character']

        cluster_score, train_eval_intermediate = train_eval_model.train_eval_model(base, query, trainset,
                                                                                   train_model_config)
        del trainset
        intermediate = {
            "ins_id": model_info[0],
            'dataset_partition': partition_intermediate,
            'prepare_train_sample': prepare_train_intermediate,
            'train_eval_model': train_eval_intermediate
        }
        intermediate_result_l.append(intermediate)
        cluster_score_l.append(cluster_score)
        label = partition_info[1]
        label_l.append(label)
    del query, dataset_partition_para, model_l

    dir_io.save_numpy(program_train_para_dir + '/cls_score_l.npy', np.array(cluster_score_l))
    dir_io.save_numpy(program_train_para_dir + '/label_l.npy', np.array(label_l, dtype=object))

    save_classifier_config = {
        'program_train_para_dir': program_train_para_dir,
        'n_item': base.shape[0]
    }
    # integrate the cluster_score_l and label_l to get the score_table and store the score_table in /train_para
    score_table, integrate_intermediate = train_eval_model.integrate_save_score_table_parallel(cluster_score_l,
                                                                                               label_l,
                                                                                               save_classifier_config,
                                                                                               save=save_train_para)
    # if the memory exceed, can change the method to integrate_save_score_table_single
    del label_l, cluster_score_l

    result_integrate_config = {
        'k': topk,
        'program_result_dir': program_result_dir,
        'efSearch_l': [2 ** i for i in range(1 + int(math.log2(len(base))))],
    }
    del base
    count_recall_intermediate = result_integrate.integrate_single(score_table, gnd, result_integrate_config)
    del result_integrate_config, gnd
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
