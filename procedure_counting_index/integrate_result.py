import json
from procedure_counting_index.util import io
from procedure_counting_index.model_integrate_2 import integrate_model
import os
from util import read_data, dir_io
from util.numpy import load_data


# the same as the nn_classification
def integrate_result(config_dir):
    with open(config_dir, 'r') as f:
        config = json.load(f)

    project_train_para_dir = '%s/train_para' % config['project_dir']

    # get the result for every classifier and its config file
    long_term_config_m = {}
    short_term_config_m = {}
    short_term_config_before_run_m = {}
    intermediate_result_m = {}
    score_table_ptr_l = []

    classifier_fname_l = config['classifier_fname_l']
    for classifier_fname in classifier_fname_l:
        tmp_dir = '%s/train_para/%s_counting_index' % (config['project_dir'], classifier_fname)
        # print(tmp_dir)
        tmp_intermediate_result = read_data.get_score_table_intermediate_config(tmp_dir)

        long_term_config_m[classifier_fname] = tmp_intermediate_result[0]
        short_term_config_m[classifier_fname] = tmp_intermediate_result[1]
        short_term_config_before_run_m[classifier_fname] = tmp_intermediate_result[2]
        intermediate_result_m[classifier_fname] = tmp_intermediate_result[3]
        score_table_ptr_l.append(tmp_intermediate_result[4])
        # long_term_config, short_term_config, short_term_config_before_run, intermediate_result, total_score_table

    print('integrate config complete')

    # load gnd
    data_dir = '%s/data/%s_%d/gnd.npy' % (
        config['project_dir'], config['data_fname'], config['k'])
    gnd = load_data.load_single_data_npy(data_dir)

    print('load gnd complete')

    program_result_dir = '%s/result/%s_counting_index' % (config['project_dir'], config['program_fname'])
    dir_io.delete_dir_if_exist(program_result_dir)
    result_integrate_config = {
        'k': config['k'],
        'program_result_dir': program_result_dir,
        'efSearch_l': config['efSearch_l'],
    }
    integrate_model.integrate(score_table_ptr_l, gnd, result_integrate_config)

    save_config_config = {
        'long_term_config': long_term_config_m,
        'short_term_config': short_term_config_m,
        'short_term_config_before_run': short_term_config_before_run_m,
        'intermediate_result': intermediate_result_m,
        'save_dir': program_result_dir,
        'program_fname': config['program_fname']
    }
    io.save_config(save_config_config)
