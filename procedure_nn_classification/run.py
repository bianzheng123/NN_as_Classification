from util.numpy import load_data
from procedure_nn_classification.init_0 import partition_preprocess
from procedure_nn_classification.dataset_partition_1 import dataset_partition
from procedure_nn_classification.prepare_train_sample_2 import prepare_train_sample
from procedure_nn_classification.train_eval_model_3 import train_eval_model
import json
import numpy as np
import time
from util import dir_io

from torch.utils.data import Dataset, DataLoader, TensorDataset


def run(long_term_config_dir, short_term_config_dir):
    with open(long_term_config_dir, 'r') as f:
        long_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config_before_run = json.load(f)

    total_start_time = time.time()

    # load data
    data_dir = '%s/data/%s_%d' % (
        long_term_config['project_dir'], long_term_config['data_fname'], long_term_config['k'])
    load_data_config = {
        'data_dir': data_dir
    }
    base, query, learn, gnd, base_base_gnd = load_data.load_data_npy(load_data_config)

    program_train_para_dir = '%s/train_para/%s' % (long_term_config['project_dir'], short_term_config['program_fname'])

    dir_io.delete_dir_if_exist(program_train_para_dir)

    partition_preprocess_config = {
        "independent_config": short_term_config['independent_config'],
        'n_cluster': short_term_config['n_cluster'],
        'kahip_dir': long_term_config['kahip_dir'],
        "program_train_para_dir": program_train_para_dir,
    }
    # get the initial classifier object for each method
    # preprocess to get enable parallelization, however, when use the Multiprocessor, some bug happen
    model_l, preprocess_intermediate = partition_preprocess.preprocess(base, partition_preprocess_config)

    cluster_score_l = []
    intermediate_result_l = []
    label_map_l = []
    for model in model_l:
        partition_info, model_info = dataset_partition.partition(base, model)
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
        intermediate = {
            "ins_id": '%d_%d' % (model_info[0]["entity_number"], model_info[0]["classifier_number"]),
            'dataset_partition': partition_intermediate,
            'prepare_train_sample': prepare_train_intermediate,
            'train_eval_model': train_eval_intermediate
        }
        intermediate_result_l.append(intermediate)
        cluster_score_l.append(cluster_score)
        label_map = partition_info[1]
        label_map_l.append(label_map)

    save_classifier_config = {
        'program_train_para_dir': program_train_para_dir,
        'n_item': base.shape[0]
    }
    # integrate the cluster_score_l and label_map_l to get the score_table and store the score_table in /train_para
    train_eval_model.integrate_save_score_table(cluster_score_l, label_map_l, save_classifier_config)

    total_end_time = time.time()
    intermediate_result_final = {
        'total_time_consume': total_end_time - total_start_time,
        'preprocess': preprocess_intermediate,
        'classifier': intermediate_result_l
    }

    # save intermediate and configure file
    save_config_config = {
        'long_term_config': long_term_config,
        'short_term_config': short_term_config,
        'short_term_config_before_run': short_term_config_before_run,
        'intermediate_result': intermediate_result_final,
        'save_dir': program_train_para_dir,
        'program_fname': short_term_config['program_fname']
    }
    dir_io.save_config(save_config_config)
