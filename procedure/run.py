from util.numpy import load_data
from procedure.init_0 import partition_preprocess
from procedure.dataset_partition_1 import dataset_partition
from procedure.prepare_train_sample_2 import prepare_train_sample
from procedure.train_eval_model_3 import train_eval_model
from procedure.result_integrate_4 import result_integrate
import json
import numpy as np
import os
from util import dir_io

from torch.utils.data import Dataset, DataLoader, TensorDataset


def train_eval(long_term_config_dir, short_term_config_dir):
    with open(long_term_config_dir, 'r') as f:
        long_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config_before_run = json.load(f)

    # 加载数据
    data_dir = '%s/data/%s_%d' % (
        long_term_config['project_dir'], long_term_config['data_fname'], long_term_config['k'])
    load_data_config = {
        'data_dir': data_dir
    }
    base, query, learn, gnd, base_base_gnd = load_data.load_data_npy(load_data_config)

    # 将代码变成每一个分类器都单独运行，而不是批量执行某一个步骤
    # 执行切分的不可并行化结果
    program_train_para_dir = '%s/train_para/%s' % (long_term_config['project_dir'], short_term_config['program_fname'])

    dir_io.delete_dir_if_exist(program_train_para_dir)

    partition_preprocess_config = {
        "independent_config": short_term_config['independent_config'],
        'n_cluster': short_term_config['n_cluster'],
        'kahip_dir': long_term_config['kahip_dir'],
        "program_train_para_dir": program_train_para_dir,
    }
    model_l = partition_preprocess.preprocess(base, partition_preprocess_config)

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
        # 准备训练的代码将要完成
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
    # cluster_score_l和label_map_l整合成score_table并保存
    train_eval_model.integrate_save_score_table(cluster_score_l, label_map_l, save_classifier_config)

    # 保存中间结果以及配置文件
    save_config_config = {
        'long_term_config': long_term_config,
        'short_term_config': short_term_config,
        'short_term_config_before_run': short_term_config_before_run,
        'intermediate_result': intermediate_result_l,
        'save_dir': program_train_para_dir,
        'program_fname': short_term_config['program_fname']
    }
    result_integrate.save_config(save_config_config)
