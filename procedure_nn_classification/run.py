from util.numpy import load_data
from procedure_nn_classification.init_0 import partition_preprocess
from procedure_nn_classification.dataset_partition_1 import dataset_partition
from procedure_nn_classification.prepare_train_sample_2 import prepare_train_sample
from procedure_nn_classification.train_eval_model_3 import train_eval_model
from procedure_common import result_integrate
import json
import numpy as np
import time
from util import dir_io
import math


def run(long_term_config_dir, short_term_config_dir):
    with open(long_term_config_dir, 'r') as f:
        long_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config_before_run = json.load(f)

    save_train_para = True
    total_start_time = time.time()

    # load data
    data_dir = '%s/data/%s_%d' % (
        long_term_config['project_dir'], long_term_config['data_fname'], long_term_config['k'])
    load_data_config = {
        'data_dir': data_dir
    }
    base, query, learn, gnd, base_base_gnd = load_data.load_data_npy(load_data_config)
    del load_data_config

    program_train_para_dir = '%s/train_para/%s' % (long_term_config['project_dir'], short_term_config['program_fname'])
    program_result_dir = '%s/result/%s' % (long_term_config['project_dir'], short_term_config['program_fname'])

    dir_io.delete_dir_if_exist(program_train_para_dir)
    dir_io.delete_dir_if_exist(program_result_dir)

    partition_preprocess_config = {
        "independent_config": short_term_config['independent_config'],
        'n_cluster': short_term_config['n_cluster'],
        'kahip_dir': long_term_config['kahip_dir'],
        "program_train_para_dir": program_train_para_dir,
    }
    # get the initial classifier object for each method
    # preprocess to get enable parallelization, however, when use the Multiprocessor, some bug happen
    model_l, preprocess_intermediate = partition_preprocess.preprocess(base, partition_preprocess_config)
    del partition_preprocess_config

    cluster_score_l = []
    intermediate_result_l = []
    label_l = []
    dataset_partition_para = None
    for model in model_l:
        partition_info, model_info, dataset_partition_para = dataset_partition.partition(base, model,
                                                                                         dataset_partition_para)
        partition_intermediate = model_info[1]

        prepare_train_config = short_term_config['prepare_train_sample']
        prepare_train_config['n_cluster'] = short_term_config['n_cluster']
        prepare_train_config['program_train_para_dir'] = program_train_para_dir
        prepare_train_config['classifier_number'] = model_info[0]["classifier_number"]
        prepare_train_config['entity_number'] = model_info[0]["entity_number"]
        # to prepare for the training set and evaluating set
        trainset, prepare_train_intermediate = prepare_train_sample.prepare_train(base, base_base_gnd, partition_info,
                                                                                  prepare_train_config)

        train_model_config = short_term_config['train_model']
        train_model_config['n_cluster'] = short_term_config['n_cluster']
        train_model_config['classifier_number'] = model_info[0]['classifier_number']
        train_model_config['entity_number'] = model_info[0]['entity_number']
        train_model_config['program_train_para_dir'] = program_train_para_dir

        cluster_score, train_eval_intermediate = train_eval_model.train_eval_model(base, query, trainset,
                                                                                   train_model_config)
        del trainset
        intermediate = {
            "ins_id": '%d_%d' % (model_info[0]["entity_number"], model_info[0]["classifier_number"]),
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
        'k': long_term_config['k'],
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
            'program_fname': short_term_config['program_fname']
        }
        dir_io.save_config(save_config_config)

    save_config_config = {
        'long_term_config': long_term_config,
        'short_term_config': short_term_config,
        'short_term_config_before_run': short_term_config_before_run,
        'intermediate_result': intermediate_result_final,
        'save_dir': program_result_dir,
        'program_fname': short_term_config['program_fname']
    }
    dir_io.save_config(save_config_config)
