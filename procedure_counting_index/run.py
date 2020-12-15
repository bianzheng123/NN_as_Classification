import numpy as np
from util.numpy import load_data
from procedure_counting_index.util import io
from procedure_counting_index.dataset_partition_1 import partition_data
from procedure_counting_index.init_0 import partition_preprocess
from procedure_counting_index.model_integrate_2 import integrate_model
import time
import json
from util import dir_io


def run(long_term_config_dir, short_term_config_dir):
    with open(long_term_config_dir, 'r') as f:
        long_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config_before_run = json.load(f)

    total_start_time = time.time()

    data_dir = '%s/data/%s_%d' % (
        long_term_config['project_dir'], long_term_config['data_fname'], long_term_config['k'])
    load_data_config = {
        'data_dir': data_dir
    }
    # load data
    data = load_data.load_data_npy(load_data_config)
    base = data[0]
    query = data[1]

    # classification
    program_train_para_dir = '%s/train_para/%s_counting_index' % (long_term_config['project_dir'], short_term_config['program_fname'])

    dir_io.delete_dir_if_exist(program_train_para_dir)

    partition_preprocess_config = {
        "independent_config": short_term_config['independent_config'],
        'n_cluster': short_term_config['n_cluster'],
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
            "ins_id": '%d_%d' % (model_info[0]["entity_number"], model_info[0]["classifier_number"]),
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
    integrate_model.integrate_save_score_table(predict_cluster_l, label_map_l, save_classifier_config)

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
    io.save_config(save_config_config)
